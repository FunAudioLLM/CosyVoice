""" Example Usage
    CUDA_VISIBLE_DEVICES=0 \
        python3 token2wav_cosyvoice3.py --enable-trt || exit 1
"""
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import s3tokenizer
import os
import logging
import argparse
import queue
import time
import numpy as np
from functools import partial
from hyperpyyaml import load_hyperpyyaml
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram
from torch.utils.data import DataLoader
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CosyVoice3 mel params from cosyvoice3.yaml (fmax=None, NOT 8000)
mel_spectrogram = partial(matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16, autocast_mode=False):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    if autocast_mode:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if not autocast_mode:
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.device = device
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            assert trt_context is not None, 'failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}'.format(trt_concurrent)
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, 'no avaialbe estimator context'

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


class CosyVoice3_Token2Wav(torch.nn.Module):
    def __init__(self, model_dir, enable_trt=False, device_id=0, autocast_mode=True, streaming=False):
        super().__init__()
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self.autocast_mode = autocast_mode
        self.streaming = streaming

        # Load flow and hift from cosyvoice3.yaml
        with open(f"{model_dir}/cosyvoice3.yaml", "r") as f:
            configs = load_hyperpyyaml(f, overrides={
                'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
            })
        self.flow = configs['flow']
        self.flow.load_state_dict(
            torch.load(f"{model_dir}/flow.pt", map_location="cpu", weights_only=True),
            strict=True
        )
        self.flow.to(self.device).eval()

        self.hift = configs['hift']
        hift_state_dict = {
            k.replace('generator.', ''): v
            for k, v in torch.load(f"{model_dir}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        # Speaker embedding model (campplus)
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            f"{model_dir}/campplus.onnx", sess_options=option,
            providers=["CPUExecutionProvider"]
        )

        # Audio tokenizer v3
        self.audio_tokenizer = s3tokenizer.load_model(
            f"{model_dir}/speech_tokenizer_v3.onnx"
        ).to(self.device).eval()

        self.fp16 = enable_trt
        if enable_trt:
            self.flow.half()
            self.load_trt(model_dir)
            self.load_spk_trt(model_dir)

    def load_trt(self, model_dir, trt_concurrent=1):
        streaming_prefix = 'streaming.' if self.streaming else ''
        if self.autocast_mode:
            onnx_path = f'{model_dir}/flow.decoder.estimator.{streaming_prefix}autocast_fp16.onnx'
            trt_path = f'{model_dir}/flow.decoder.estimator.{streaming_prefix}autocast_fp16.{self.device_id}.plan'
        else:
            onnx_path = f'{model_dir}/flow.decoder.estimator.{streaming_prefix}fp32.onnx'
            trt_path = f'{model_dir}/flow.decoder.estimator.{streaming_prefix}fp32.{self.device_id}.plan'

        if not os.path.exists(trt_path) or os.path.getsize(trt_path) == 0:
            trt_kwargs = self.get_trt_kwargs()
            convert_onnx_to_trt(trt_path, trt_kwargs, onnx_path,
                               fp16=True, autocast_mode=self.autocast_mode)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(trt_path, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(trt_path)
        self.flow.decoder.estimator = TrtContextWrapper(
            estimator_engine, trt_concurrent=trt_concurrent, device=self.device
        )

    def get_trt_kwargs(self):
        # CosyVoice3 DiT estimator has 6 inputs: x, mask, mu, t, spks, cond
        # Only inputs with dynamic dims need optimization profiles.
        # t=[2(fixed)] and spks=[2(fixed),80(fixed)] are fully fixed, TRT infers from ONNX.
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape,
                'max_shape': max_shape, 'input_names': input_names}

    def load_spk_trt(self, model_dir, trt_concurrent=1, fp16=False):
        spk_trt_path = f'{model_dir}/campplus.{self.device_id}.fp32.plan'
        spk_onnx_path = f'{model_dir}/campplus.onnx'
        if not os.path.exists(spk_trt_path) or os.path.getsize(spk_trt_path) == 0:
            trt_kwargs = self.get_spk_trt_kwargs()
            convert_onnx_to_trt(spk_trt_path, trt_kwargs, spk_onnx_path, fp16)
        import tensorrt as trt
        with open(spk_trt_path, 'rb') as f:
            spk_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert spk_engine is not None, 'failed to load trt {}'.format(spk_trt_path)
        self.spk_model = TrtContextWrapper(spk_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_spk_trt_kwargs(self):
        min_shape = [(1, 4, 80)]
        opt_shape = [(1, 500, 80)]
        max_shape = [(1, 3000, 80)]
        input_names = ["input"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape,
                'max_shape': max_shape, 'input_names': input_names}

    def forward_spk_embedding(self, spk_feat):
        if isinstance(self.spk_model, onnxruntime.InferenceSession):
            return self.spk_model.run(
                None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
        else:
            [spk_model, stream], trt_engine = self.spk_model.acquire_estimator()
            with torch.cuda.device(self.device_id):
                torch.cuda.current_stream().synchronize()
                spk_feat = spk_feat.unsqueeze(dim=0).to(self.device)
                batch_size = spk_feat.size(0)

                with stream:
                    spk_model.set_input_shape('input', (batch_size, spk_feat.size(1), 80))
                    output_tensor = torch.empty((batch_size, 192), device=spk_feat.device)

                    data_ptrs = [spk_feat.contiguous().data_ptr(),
                                 output_tensor.contiguous().data_ptr()]
                    for i, j in enumerate(data_ptrs):
                        spk_model.set_tensor_address(trt_engine.get_tensor_name(i), j)
                    assert spk_model.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                    torch.cuda.current_stream().synchronize()
                self.spk_model.release_estimator(spk_model, stream)

            return output_tensor.cpu().numpy().flatten().tolist()

    def prompt_audio_tokenization(self, prompt_audios_list):
        prompt_speech_tokens_list, prompt_speech_mels_list = [], []
        for audio in prompt_audios_list:
            assert len(audio.shape) == 1
            log_mel = s3tokenizer.log_mel_spectrogram(audio)
            prompt_speech_mels_list.append(log_mel)
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(prompt_speech_mels_list)
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.to(self.device), prompt_mels_lens_for_llm.to(self.device)
        )
        for i in range(len(prompt_speech_tokens)):
            speech_tokens_i = prompt_speech_tokens[i, :prompt_speech_tokens_lens[i].item()].tolist()
            prompt_speech_tokens_list.append(speech_tokens_i)
        return prompt_speech_tokens_list

    def get_spk_emb(self, prompt_audios_list):
        spk_emb_for_flow = []
        for audio in prompt_audios_list:
            assert len(audio.shape) == 1
            spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
            spk_emb = self.forward_spk_embedding(spk_feat)
            spk_emb_for_flow.append(spk_emb)
        spk_emb_for_flow = torch.tensor(spk_emb_for_flow)
        return spk_emb_for_flow

    def get_prompt_mels(self, prompt_audios_list, prompt_audios_sample_rate):
        prompt_mels_for_flow = []
        prompt_mels_lens_for_flow = []
        for audio, sample_rate in zip(prompt_audios_list, prompt_audios_sample_rate):
            assert len(audio.shape) == 1
            audio = audio.unsqueeze(0)
            if sample_rate != 24000:
                audio = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=24000)(audio)
            # CosyVoice3: fmax=None (Nyquist), matching cosyvoice3.yaml
            mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, 80]
            prompt_mels_for_flow.append(mel)
            prompt_mels_lens_for_flow.append(mel.shape[0])
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
            prompt_mels_for_flow, batch_first=True, padding_value=0)  # [B, T', 80]
        prompt_mels_lens_for_flow = torch.tensor(prompt_mels_lens_for_flow)
        return prompt_mels_for_flow, prompt_mels_lens_for_flow

    def forward_flow(self, prompt_speech_tokens_list, generated_speech_tokens_list,
                     prompt_mels_for_flow, prompt_mels_lens_for_flow,
                     spk_emb_for_flow):
        batch_size = len(generated_speech_tokens_list)
        generated_mels_list = []

        # CausalMaskedDiffWithDiT.inference asserts batch_size==1, so iterate per-sample
        for i in range(batch_size):
            token = torch.tensor([generated_speech_tokens_list[i]]).to(self.device)
            token_len = torch.tensor([len(generated_speech_tokens_list[i])]).to(self.device)
            prompt_token = torch.tensor([prompt_speech_tokens_list[i]]).to(self.device)
            prompt_token_len = torch.tensor([len(prompt_speech_tokens_list[i])]).to(self.device)
            prompt_feat = prompt_mels_for_flow[i:i+1, :prompt_mels_lens_for_flow[i]].to(self.device)
            prompt_feat_len = prompt_mels_lens_for_flow[i:i+1].to(self.device)
            embedding = spk_emb_for_flow[i:i+1].to(self.device)

            # CausalMaskedDiffWithDiT.inference returns mel already without prompt portion
            with torch.cuda.amp.autocast(self.fp16):
                mel, _ = self.flow.inference(
                    token=token,
                    token_len=token_len,
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    streaming=False,
                    finalize=True
                )
            generated_mels_list.append(mel)

        return generated_mels_list

    def forward_hift(self, generated_mels_list):
        generated_wavs = []
        for mel in generated_mels_list:
            # CausalHiFTGenerator.inference with finalize=True
            wav, _ = self.hift.inference(speech_feat=mel, finalize=True)
            generated_wavs.append(wav)
        return generated_wavs

    def forward_stream(self, generated_speech_tokens, prompt_speech_tokens,
                        prompt_feat, embedding,
                        token_hop_len=25, stream_scale_factor=2, token_max_hop_len=100):
        """Streaming token2wav for a single sample: process tokens in chunks."""
        prompt_token = torch.tensor([prompt_speech_tokens]).to(self.device)
        prompt_token_len = torch.tensor([len(prompt_speech_tokens)]).to(self.device)
        prompt_feat = prompt_feat.to(self.device)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]]).to(self.device)
        embedding = embedding.to(self.device)

        pre_lookahead_len = self.flow.pre_lookahead_len
        token_mel_ratio = self.flow.token_mel_ratio

        # Align first chunk with hop_len boundary
        prompt_token_pad = int(
            np.ceil(prompt_token.shape[1] / token_hop_len) * token_hop_len
            - prompt_token.shape[1]
        )

        total_tokens = len(generated_speech_tokens)
        token_offset = 0
        current_hop = token_hop_len
        hift_cache_mel = None
        speech_offset = 0
        audio_chunks = []

        while token_offset < total_tokens:
            this_hop = current_hop + prompt_token_pad if token_offset == 0 else current_hop
            remaining = total_tokens - token_offset

            if remaining >= this_hop + pre_lookahead_len:
                end_idx = token_offset + this_hop + pre_lookahead_len
                this_token = torch.tensor([generated_speech_tokens[:end_idx]]).to(self.device)
                finalize = False
            else:
                this_token = torch.tensor([generated_speech_tokens]).to(self.device)
                finalize = True

            with torch.cuda.amp.autocast(self.fp16):
                mel, _ = self.flow.inference(
                    token=this_token,
                    token_len=torch.tensor([this_token.shape[1]]).to(self.device),
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    streaming=True,
                    finalize=finalize,
                )

            mel = mel[:, :, token_offset * token_mel_ratio:]

            if hift_cache_mel is not None:
                mel = torch.concat([hift_cache_mel, mel], dim=2)
            hift_cache_mel = mel

            tts_speech, _ = self.hift.inference(speech_feat=mel, finalize=finalize)
            tts_speech = tts_speech[:, speech_offset:]
            speech_offset += tts_speech.shape[1]

            logger.info(f"[stream] token_offset={token_offset}, this_hop={this_hop}, "
                        f"mel_shape={mel.shape}, speech_len={tts_speech.shape[1]}, finalize={finalize}")

            audio_chunks.append(tts_speech)

            token_offset += this_hop
            if not finalize:
                current_hop = min(token_max_hop_len, current_hop * stream_scale_factor)
            else:
                break

        return torch.cat(audio_chunks, dim=1)

    @torch.inference_mode()
    def forward(self, generated_speech_tokens_list, prompt_audios_list,
                prompt_audios_sample_rate, streaming=False):
        assert all(sr == 16000 for sr in prompt_audios_sample_rate)

        prompt_speech_tokens_list = self.prompt_audio_tokenization(prompt_audios_list)
        prompt_mels_for_flow, prompt_mels_lens_for_flow = self.get_prompt_mels(
            prompt_audios_list, prompt_audios_sample_rate)
        spk_emb_for_flow = self.get_spk_emb(prompt_audios_list)

        # Align prompt_speech_feat and prompt_speech_token to exact 2:1 ratio
        # (matches frontend.frontend_zero_shot logic)
        for i in range(len(prompt_speech_tokens_list)):
            token_len = min(int(prompt_mels_lens_for_flow[i].item() / 2),
                            len(prompt_speech_tokens_list[i]))
            prompt_speech_tokens_list[i] = prompt_speech_tokens_list[i][:token_len]
            prompt_mels_lens_for_flow[i] = 2 * token_len

        if streaming:
            generated_wavs = []
            for i in range(len(generated_speech_tokens_list)):
                prompt_feat = prompt_mels_for_flow[i:i+1, :prompt_mels_lens_for_flow[i]]
                embedding = spk_emb_for_flow[i:i+1]
                wav = self.forward_stream(
                    generated_speech_tokens_list[i],
                    prompt_speech_tokens_list[i],
                    prompt_feat, embedding,
                )
                generated_wavs.append(wav)
            return generated_wavs

        generated_mels_list = self.forward_flow(
            prompt_speech_tokens_list, generated_speech_tokens_list,
            prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow)

        generated_wavs = self.forward_hift(generated_mels_list)
        return generated_wavs
