import torch
import os
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import torchaudio
import time
from token2wav_dit import CosyVoice2_Token2Wav
import soundfile as sf


def collate_fn(batch):
    ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = [], [], [], []
    prompt_speech_tokens_list, prompt_text_list = [], []
    for item in batch:
        generated_speech_tokens_list.append(item['target_audio_cosy2_tokens'])
        audio = torch.from_numpy(item['prompt_audio']['array']).float()
        prompt_audios_list.append(audio)
        prompt_audios_sample_rate.append(item['prompt_audio']['sampling_rate'])
        ids.append(item['id'])
        prompt_speech_tokens_list.append(item['prompt_audio_cosy2_tokens'])
        prompt_text_list.append(item['prompt_text'])

    return ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate, prompt_speech_tokens_list, prompt_text_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-trt", action="store_true")
    parser.add_argument("--model-dir", type=str, default="./Step-Audio-2-mini/token2wav")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="generated_wavs")
    parser.add_argument("--huggingface-dataset-split", type=str, default="wenetspeech4tts")
    parser.add_argument("--dataset-name", type=str, default="yuekai/seed_tts_cosy2")
    parser.add_argument("--strategy", type=str, default="equal", choices=["equal", "exponential"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, split=args.huggingface_dataset_split, trust_remote_code=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    token2wav_model = CosyVoice2_Token2Wav(model_dir=args.model_dir, enable_trt=args.enable_trt, streaming=True)

    CHUNK_SIZE = 25
    token_frame_rate = 25
    OVERLAP_SIZE = 0

    warmup_times = 3
    for _ in range(warmup_times):
        start_time = time.time()
        total_forward_count = 0
        for batch in data_loader:
            tts_speech_list = []
            ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate, prompt_speech_tokens_list, prompt_text_list = batch

            id, generated_speech_tokens, prompt_audio, prompt_audio_sample_rate = ids[0], generated_speech_tokens_list[0], prompt_audios_list[0], prompt_audios_sample_rate[0]

            assert prompt_audio_sample_rate == 16000

            prompt_text = prompt_text_list[0]
            prompt_speech_tokens = prompt_speech_tokens_list[0]

            semantic_token_ids_arr, token_offset = [], 0
            flow_prompt_speech_token_len = len(prompt_speech_tokens)

            buffer = generated_speech_tokens
            output_wavs = []
            chunk_index = 0
            while True:
                if args.strategy == "equal":
                    this_chunk_size = CHUNK_SIZE
                elif args.strategy == "exponential":
                    this_chunk_size = token_frame_rate * (2 ** chunk_index)

                if len(buffer) >= this_chunk_size + token2wav_model.flow.pre_lookahead_len:
                    wavs = token2wav_model.forward_streaming(
                        buffer[:this_chunk_size + token2wav_model.flow.pre_lookahead_len],
                        False, request_id=id, speaker_id=f"{id}", prompt_audio=prompt_audio,
                        prompt_audio_sample_rate=prompt_audio_sample_rate
                    )
                    buffer = buffer[this_chunk_size - OVERLAP_SIZE:]

                    output_wavs.append(wavs)
                    total_forward_count += 1
                    chunk_index += 1

                else:
                    wavs = token2wav_model.forward_streaming(
                        buffer, True, request_id=id, speaker_id=f"{id}",
                        prompt_audio=prompt_audio, prompt_audio_sample_rate=prompt_audio_sample_rate
                    )
                    output_wavs.append(wavs)
                    total_forward_count += 1
                    # chunk_index += 1
                    break

            for i, wav in enumerate(output_wavs):
                output_wavs[i] = wav.cpu().numpy().squeeze()

            audios = output_wavs
            reconstructed_audio = np.concatenate(audios)
            sf.write(os.path.join(args.output_dir, f"{id}.wav"), reconstructed_audio, 24000, "PCM_16")

        end_time = time.time()

        if _ == 0:
            token2wav_model.speaker_cache = {}
            print(f"Warmup time: {end_time - start_time} seconds")
            print("clear speaker cache")
        elif _ == 1:
            print(f"Cost time without speaker cache: {end_time - start_time} seconds")
        else:
            print(f"Cost time with speaker cache: {end_time - start_time} seconds")
            print(f"Total flow matching forward calls: {total_forward_count}")
