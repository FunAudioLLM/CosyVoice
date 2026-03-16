""" Example Usage
    CUDA_VISIBLE_DEVICES=0 \
        python3 infer_cosyvoice3_token2wav.py \
            --output-dir $output_dir \
            --llm-model-name-or-path $huggingface_model_local_dir \
            --token2wav-path $token2wav_model_dir \
            --backend $backend \
            --batch-size $batch_size --token2wav-batch-size $token2wav_batch_size \
            --engine-dir $trt_engines_dir \
            --split-name ${dataset} || exit 1
"""
import argparse
import json
import os
import time
import asyncio

import torch
import torchaudio
import s3tokenizer
import soundfile as sf
import requests
import httpx
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

from token2wav_cosyvoice3 import CosyVoice3_Token2Wav

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


async def send_request_async(client, url, payload):
    response = await client.post(url, json=payload, timeout=None)
    response.raise_for_status()
    response_json = response.json()
    return response_json['choices'][0]['message']['content']


async def send_batch_requests_async(api_base, model_name, chats, temperature, top_p, top_k):
    async with httpx.AsyncClient() as client:
        tasks = []
        for chat in chats:
            payload = {
                "model": model_name,
                "messages": chat,
                "max_tokens": 2048,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": 1.1,
                "stop": ["<|eos1|>", "<|eos|>"],
                "stream": False,
            }
            tasks.append(send_request_async(client, api_base, payload))
        return await asyncio.gather(*tasks)


def extract_speech_ids(speech_tokens_str):
    """Extract speech IDs from token strings like <|s_23456|>"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def convert_cosy3_tokens_to_speech_id_str(cosy3_tokens):
    """Convert CosyVoice3 tokens to speech IDs string like <|s_23456|>"""
    if hasattr(cosy3_tokens, 'cpu'):
        cosy3_tokens = cosy3_tokens.cpu().numpy().tolist()
    speech_id_str = ""
    for token in cosy3_tokens:
        speech_id_str += f"<|s_{token}|>"
    return speech_id_str


def get_args():
    parser = argparse.ArgumentParser(description="Speech generation using LLM + CosyVoice3")
    parser.add_argument(
        "--split-name", type=str, default="wenetspeech4tts",
        help="huggingface dataset split name",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="dir to save result",
    )
    parser.add_argument(
        "--batch-size", default=1, type=int,
        help="batch size (per-device) for LLM inference",
    )
    parser.add_argument(
        "--token2wav-batch-size", default=1, type=int,
        help="batch size (per-device) for token2wav inference",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="workers for dataloader",
    )
    parser.add_argument(
        "--prefetch", type=int, default=None, help="prefetch for dataloader",
    )
    parser.add_argument(
        "--llm-model-name-or-path", required=True, type=str,
        help="CosyVoice3 HF LLM path (e.g. ./hf_cosyvoice3_llm)",
    )
    parser.add_argument(
        "--token2wav-path", required=True, type=str,
        help="CosyVoice3 model path (e.g. /workspace_yuekai/HF/Fun-CosyVoice3-0.5B-2512)",
    )
    parser.add_argument(
        "--enable-trt", action="store_true",
        help="Enable TensorRT for flow decoder estimator",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Enable streaming for flow decoder estimator",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="top p for sampling",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="temperature for sampling",
    )
    parser.add_argument(
        "--top-k", type=int, default=15, help="top k for sampling",
    )
    parser.add_argument(
        "--backend", type=str, default="hf",
        choices=["hf", "trtllm", "vllm", "trtllm-serve"],
        help="Backend to use for LLM inference",
    )
    parser.add_argument(
        "--engine-dir", type=str, default=None,
        help="TensorRT-LLM engine directory (required when backend is 'trtllm')",
    )
    parser.add_argument(
        "--kv-cache-free-gpu-memory-fraction", type=float, default=0.6,
        help="Fraction of GPU memory to free for KV cache (TensorRT-LLM only)",
    )
    parser.add_argument(
        "--openai-api-base", type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI API base URL (for trtllm-serve backend)",
    )
    parser.add_argument(
        "--openai-model-name", type=str, default="trt_engines_bfloat16",
        help="Model name to use with OpenAI API (for trtllm-serve backend)",
    )
    parser.add_argument(
        "--epoch", type=int, default=1, help="Epoch to run",
    )
    return parser.parse_args()


def data_collator(batch, tokenizer, s3_tokenizer):
    """Data collator: extracts cosy3 tokens from prompt_audio using v3 s3 tokenizer."""
    device = s3_tokenizer.device if s3_tokenizer is not None else torch.device("cpu")
    target_sample_rate = 16000

    input_ids_list, prompt_audio_list, prompt_text_list = [], [], []
    mels, prompt_audio_cosy3tokens_list, full_text_list = [], [], []
    chat_list = []

    for item in batch:
        prompt_text, target_text = item["prompt_text"], item["target_text"]
        prompt_text_list.append(prompt_text)
        full_text = 'You are a helpful assistant.<|endofprompt|>' + prompt_text + target_text
        full_text_list.append(full_text)

        # Get prompt audio (convert to 16kHz for s3 tokenizer)
        ref_audio = torch.from_numpy(item["prompt_audio"]["array"]).float().unsqueeze(0)
        ref_sr = item["prompt_audio"]["sampling_rate"]
        if ref_sr != target_sample_rate:
            ref_audio = torchaudio.transforms.Resample(ref_sr, target_sample_rate)(ref_audio)
        prompt_audio_list.append(ref_audio)

        # Extract cosy3 tokens from prompt_audio using v3 s3 tokenizer
        mels.append(s3tokenizer.log_mel_spectrogram(ref_audio.squeeze(0)))

    # Batch tokenization with v3 tokenizer
    if len(mels) > 0:
        mels_padded, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = s3_tokenizer.quantize(mels_padded.to(device), mels_lens.to(device))
        for i in range(len(codes)):
            prompt_audio_cosy3tokens_list.append(codes[i, :codes_lens[i].item()])

    # Build LLM inputs
    for i, prompt_audio_cosy3tokens in enumerate(prompt_audio_cosy3tokens_list):
        prompt_audio_cosy3_id_str = convert_cosy3_tokens_to_speech_id_str(
            prompt_audio_cosy3tokens)
        chat = [
            {"role": "user", "content": full_text_list[i]},
            {"role": "assistant", "content": prompt_audio_cosy3_id_str}
        ]
        chat_list.append(chat)
        input_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors='pt', continue_final_message=True)
        input_ids_list.append(input_ids.squeeze(0))

    ids = [item["id"] for item in batch]

    return {
        "input_ids": input_ids_list,
        "ids": ids,
        "prompt_text": prompt_text_list,
        "prompt_audio_list": prompt_audio_list,
        "chat_list": chat_list,
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available()
    local_rank = 0
    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name_or_path)

    if args.backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(args.llm_model_name_or_path)
        model.eval()
        model.to(device)
        runner = None
    elif args.backend == "trtllm":
        if args.engine_dir is None:
            raise ValueError("--engine-dir is required when backend is 'trtllm'")
        runtime_rank = tensorrt_llm.mpi_rank()
        model = None
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            rank=runtime_rank,
            max_output_len=2048,
            enable_context_fmha_fp32_acc=False,
            max_batch_size=args.batch_size,
            max_input_len=512,
            kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
            cuda_graph_mode=False,
            gather_generation_logits=False,
        )
        runner = ModelRunnerCpp.from_dir(**runner_kwargs)
    elif args.backend == "vllm":
        model = LLM(model=args.llm_model_name_or_path, gpu_memory_utilization=0.4)
        runner = None
    elif args.backend == "trtllm-serve":
        model = None
        runner = None
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    token2wav_model = CosyVoice3_Token2Wav(
        model_dir=args.token2wav_path, enable_trt=args.enable_trt, device_id=local_rank, streaming=args.streaming
    )

    # Load v3 s3 tokenizer for prompt audio tokenization in data_collator
    s3_tokenizer = s3tokenizer.load_model(
        f"{args.token2wav_path}/speech_tokenizer_v3.onnx"
    ).to(device).eval()

    dataset = load_dataset(
        "yuekai/seed_tts_cosy2",
        split=args.split_name,
        trust_remote_code=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=partial(data_collator, tokenizer=tokenizer, s3_tokenizer=s3_tokenizer),
    )

    for epoch in range(args.epoch):
        print(f"Running epoch {epoch}")
        total_llm_time = 0
        total_token2wav_time = 0
        total_data_load_time = 0
        total_llm_post_processing_time = 0
        total_audio_save_time = 0
        total_audio_samples = 0
        start_time = time.time()

        progress_bar = tqdm(total=len(dataset), desc="Processing", unit="wavs")

        last_batch_end_time = time.time()
        for batch in dataloader:
            data_loaded_time = time.time()
            total_data_load_time += data_loaded_time - last_batch_end_time

            with torch.no_grad():
                llm_start_time = time.time()

                if args.backend == "hf":
                    input_ids_list = batch["input_ids"]
                    if len(input_ids_list) == 1:
                        input_ids = input_ids_list[0].unsqueeze(0)
                        attention_mask = torch.ones_like(input_ids)
                    else:
                        max_len = max([len(ids) for ids in input_ids_list])
                        input_ids_list_new = [
                            torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id)])
                            for ids in input_ids_list
                        ]
                        input_ids = torch.stack(input_ids_list_new)
                        attention_mask = torch.zeros_like(input_ids)
                        for i in range(len(input_ids_list)):
                            attention_mask[i, :len(input_ids_list[i])] = 1

                    outputs = model.generate(
                        input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        max_new_tokens=2048,
                        do_sample=True,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        repetition_penalty=1.1,
                        top_k=args.top_k,
                    )
                    torch.cuda.synchronize()

                elif args.backend == "trtllm":
                    batch_input_ids = list(batch["input_ids"])
                    input_lengths = [x.size(0) for x in batch_input_ids]

                    end_id = tokenizer.convert_tokens_to_ids("<|eos1|>") if "<|eos1|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
                    outputs = runner.generate(
                        batch_input_ids=batch_input_ids,
                        max_new_tokens=2048,
                        end_id=end_id,
                        pad_id=end_id,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=1.1,
                        num_return_sequences=1,
                        streaming=False,
                        output_sequence_lengths=True,
                        output_generation_logits=False,
                        return_dict=True,
                        return_all_generated_tokens=False
                    )
                    torch.cuda.synchronize()
                    output_ids, sequence_lengths = outputs["output_ids"], outputs["sequence_lengths"]
                    num_output_sents, num_beams, _ = output_ids.size()
                    assert num_beams == 1
                    batch_size = len(batch["input_ids"])
                    num_return_sequences = num_output_sents // batch_size
                    assert num_return_sequences == 1
                    outputs = []
                    for i in range(batch_size * num_return_sequences):
                        batch_idx = i // num_return_sequences
                        output_begin = input_lengths[batch_idx]
                        output_end = sequence_lengths[i][0]
                        outputs_i = output_ids[i][0][:output_end].tolist()
                        outputs.append(outputs_i)

                elif args.backend == "vllm":
                    input_ids_list = [ids.tolist() for ids in batch["input_ids"]]
                    sampling_params = SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=1.1,
                        max_tokens=2048,
                    )
                    outputs = model.generate(prompt_token_ids=input_ids_list, sampling_params=sampling_params)
                    for j, output in enumerate(outputs):
                        outputs[j] = input_ids_list[j] + output.outputs[0].token_ids

                elif args.backend == "trtllm-serve":
                    if args.batch_size > 1:
                        outputs = asyncio.run(send_batch_requests_async(
                            args.openai_api_base,
                            args.openai_model_name,
                            batch["chat_list"],
                            args.temperature,
                            args.top_p,
                            args.top_k,
                        ))
                    else:
                        outputs = []
                        for chat in batch["chat_list"]:
                            payload = {
                                "model": args.openai_model_name,
                                "messages": chat,
                                "max_tokens": 2048,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "top_k": args.top_k,
                                "repetition_penalty": 1.1,
                                "stop": ["<|eos1|>", "<|eos|>"],
                                "stream": False,
                            }
                            response = requests.post(args.openai_api_base, json=payload)
                            response.raise_for_status()
                            response_json = response.json()
                            generated_content = response_json['choices'][0]['message']['content']
                            outputs.append(generated_content)

                llm_end_time = time.time()
                total_llm_time += (llm_end_time - llm_start_time)

                items_for_token_2wav = []
                for i in range(len(batch["ids"])):
                    llm_post_processing_start_time = time.time()
                    if args.backend == "trtllm-serve":
                        speech_tokens_str = outputs[i].strip().split('><')
                        if len(speech_tokens_str) > 1:
                            speech_tokens_str = [
                                t if t.startswith('<') else '<' + t for t in speech_tokens_str
                            ]
                            speech_tokens_str = [
                                t if t.endswith('>') else t + '>' for t in speech_tokens_str
                            ]
                        speech_ids = extract_speech_ids(speech_tokens_str)
                    else:
                        input_length = len(batch["input_ids"][i])
                        generated_ids = outputs[i][input_length:]
                        speech_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        speech_ids = extract_speech_ids(speech_tokens_str)
                    print(i, speech_ids[:10], "...", f"total={len(speech_ids)}")
                    if len(speech_ids) == 0:
                        print(f"Warning: No speech tokens generated for sample {batch['ids'][i]}, skipping")
                        llm_post_processing_end_time = time.time()
                        total_llm_post_processing_time += llm_post_processing_end_time - llm_post_processing_start_time
                        continue

                    current_prompt_audio = batch["prompt_audio_list"][i]

                    llm_post_processing_end_time = time.time()
                    total_llm_post_processing_time += llm_post_processing_end_time - llm_post_processing_start_time

                    items_for_token_2wav.append({
                        "speech_ids": speech_ids,
                        "prompt_audio": current_prompt_audio.squeeze(0),
                        "id": batch["ids"][i]
                    })

                for i in range(0, len(items_for_token_2wav), args.token2wav_batch_size):
                    t2w_batch = items_for_token_2wav[i:i + args.token2wav_batch_size]
                    if not t2w_batch:
                        continue

                    t2w_speech_tokens = [item["speech_ids"] for item in t2w_batch]
                    t2w_prompt_audios = [item["prompt_audio"] for item in t2w_batch]
                    t2w_sample_rates = [16000] * len(t2w_batch)

                    token2wav_start_time = time.time()
                    generated_wavs = token2wav_model(
                        t2w_speech_tokens, t2w_prompt_audios, t2w_sample_rates,
                        streaming=args.streaming,
                    )
                    token2wav_end_time = time.time()
                    total_token2wav_time += (token2wav_end_time - token2wav_start_time)

                    audio_save_start_time = time.time()
                    for j, audio_hat in enumerate(generated_wavs):
                        wav = audio_hat.squeeze().cpu().numpy()
                        total_audio_samples += len(wav)
                        sf.write(f"{args.output_dir}/{t2w_batch[j]['id']}.wav", wav, 24000)
                        print(f"Generated audio for sample {t2w_batch[j]['id']} with {len(t2w_speech_tokens[j])} tokens")
                    audio_save_end_time = time.time()
                    total_audio_save_time += audio_save_end_time - audio_save_start_time

            progress_bar.update(len(batch["ids"]))
            last_batch_end_time = time.time()

        progress_bar.close()
        end_time = time.time()
        total_audio_duration_seconds = total_audio_samples / 24000

        log_file_path = os.path.join(args.output_dir, "log.txt")
        with open(log_file_path, 'w') as f:
            log_data = {
                "args": vars(args),
                "data_load_time_seconds": total_data_load_time,
                "llm_time_seconds": total_llm_time,
                "llm_post_processing_time_seconds": total_llm_post_processing_time,
                "token2wav_time_seconds": total_token2wav_time,
                "audio_save_time_seconds": total_audio_save_time,
                "total_audio_duration_seconds": total_audio_duration_seconds,
                "pipeline_time_seconds": end_time - start_time,
            }
            print(log_data)
            f.write(json.dumps(log_data, indent=4))
        print(f"Metrics logged to {log_file_path}")


if __name__ == "__main__":
    args = get_args()
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
    elif args.backend == "trtllm":
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunnerCpp
    elif args.backend == "hf":
        from transformers import AutoModelForCausalLM
    elif args.backend == "trtllm-serve":
        pass
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")
    main(args)
