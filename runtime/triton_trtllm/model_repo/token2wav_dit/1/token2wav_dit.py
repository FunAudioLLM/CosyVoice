# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Example Usage
    CUDA_VISIBLE_DEVICES=0 \
        python3 token2wav.py --enable-trt || exit 1
"""
import torch
# from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import s3tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchaudio
import os
import logging
import argparse
import queue
import time
import numpy as np
from hyperpyyaml import load_hyperpyyaml


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    """perform fade_in_out in tensor style
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, dtype):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if dtype == torch.float16:
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
    if dtype == torch.float16:
        tensor_dtype = trt.DataType.HALF
    elif dtype == torch.bfloat16:
        tensor_dtype = trt.DataType.BF16
    elif dtype == torch.float32:
        tensor_dtype = trt.DataType.FLOAT
    else:
        raise ValueError('invalid dtype {}'.format(dtype))
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


class CosyVoice2_Token2Wav(torch.nn.Module):
    def __init__(self, model_dir: str, enable_trt: bool = False, device_id: int = 0, streaming: bool = False, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        with open(f"{model_dir}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
            self.flow = configs['flow']

        self.dtype = dtype
        self.flow.to(self.dtype)

        self.flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.to(self.device).eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_dir}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            f"{model_dir}/campplus.onnx", sess_options=option,
            providers=["CPUExecutionProvider"])
        self.audio_tokenizer = s3tokenizer.load_model(f"{model_dir}/speech_tokenizer_v2_25hz.onnx").to(self.device).eval()

        gpu = "l20"
        if enable_trt:
            if streaming:
                self.load_trt(
                    f'{model_dir}/flow.decoder.estimator.{self.dtype}.dynamic_batch.chunk.{gpu}.plan',
                    f'{model_dir}/flow.decoder.estimator.chunk.fp32.dynamic_batch.simplify.onnx',
                    1,
                    self.dtype, streaming
                )
            else:
                self.load_trt(
                    f'{model_dir}/flow.decoder.estimator.{self.dtype}.dynamic_batch.{gpu}.plan',
                    f'{model_dir}/flow.decoder.estimator.fp32.dynamic_batch.onnx',
                    1,
                    self.dtype
                )
            self.load_spk_trt(
                f'{model_dir}/campplus.{gpu}.fp32.trt',
                f'{model_dir}/campplus.onnx',
                1,
                False
            )

        self.streaming_flow_cache = {}
        self.speaker_cache = {}

        self.mel_cache_len = 8  # hard-coded, 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)   # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).cuda()

        # hifigan cache for streaming tts
        self.hift_cache_dict = {}

    def forward_spk_embedding(self, spk_feat):
        if isinstance(self.spk_model, onnxruntime.InferenceSession):
            return self.spk_model.run(
                None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
        else:
            [spk_model, stream], trt_engine = self.spk_model.acquire_estimator()
            # NOTE need to synchronize when switching stream
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
                    # run trt engine
                    assert spk_model.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                    torch.cuda.current_stream().synchronize()
                self.spk_model.release_estimator(spk_model, stream)

            return output_tensor.cpu().numpy().flatten().tolist()

    def load_spk_trt(self, spk_model, spk_onnx_model, trt_concurrent=1, fp16=True):
        if not os.path.exists(spk_model) or os.path.getsize(spk_model) == 0:
            trt_kwargs = self.get_spk_trt_kwargs()
            convert_onnx_to_trt(spk_model, trt_kwargs, spk_onnx_model, torch.float32)
        import tensorrt as trt
        with open(spk_model, 'rb') as f:
            spk_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert spk_engine is not None, 'failed to load trt {}'.format(spk_model)
        self.spk_model = TrtContextWrapper(spk_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_spk_trt_kwargs(self):
        min_shape = [(1, 4, 80)]
        opt_shape = [(1, 500, 80)]
        max_shape = [(1, 3000, 80)]
        input_names = ["input"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent=1, dtype=torch.float16, streaming=False):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            opt_batch_size = 2
            max_batch_size = 16
            if streaming:
                opt_batch_size, max_batch_size = 1, 1  # only support batch size 1 for streaming tts
            trt_kwargs = self.get_trt_kwargs_dynamic_batch(opt_batch_size=opt_batch_size, max_batch_size=max_batch_size, streaming=streaming)
            convert_onnx_to_trt(flow_decoder_estimator_model, trt_kwargs, flow_decoder_onnx_model, dtype)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs_dynamic_batch(self, opt_batch_size=2, max_batch_size=64, streaming=False):
        if streaming:
            min_shape = [(2, 80, 4), (2, 80, 4), (2, 80, 4), (2,), (2, 80), (16, 2, 1024, 2), (16, 2, 8, 0, 128)]
            opt_shape = [
                (opt_batch_size * 2, 80, 500), (opt_batch_size * 2, 80, 500), (opt_batch_size * 2, 80, 500),
                (opt_batch_size * 2,), (opt_batch_size * 2, 80), (16, opt_batch_size * 2, 1024, 2),
                (16, opt_batch_size * 2, 8, 100, 128)
            ]
            max_shape = [
                (max_batch_size * 2, 80, 3000), (max_batch_size * 2, 80, 3000), (max_batch_size * 2, 80, 3000),
                (max_batch_size * 2,), (max_batch_size * 2, 80), (16, max_batch_size * 2, 1024, 2),
                (16, max_batch_size * 2, 8, 1000, 128)
            ]
            input_names = ["x", "mu", "cond", "t", "spks", "cnn_cache", "att_cache"]
        else:
            min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4), (2,), (2, 80)]
            opt_shape = [
                (opt_batch_size * 2, 80, 500), (opt_batch_size * 2, 1, 500), (opt_batch_size * 2, 80, 500),
                (opt_batch_size * 2, 80, 500), (opt_batch_size * 2,), (opt_batch_size * 2, 80)
            ]
            max_shape = [
                (max_batch_size * 2, 80, 3000), (max_batch_size * 2, 1, 3000), (max_batch_size * 2, 80, 3000),
                (max_batch_size * 2, 80, 3000), (max_batch_size * 2,), (max_batch_size * 2, 80)
            ]
            input_names = ["x", "mask", "mu", "cond", "t", "spks"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def prompt_audio_tokenization(self, prompt_audios_list: list[torch.Tensor]) -> list[list[int]]:
        prompt_speech_tokens_list, prompt_speech_mels_list = [], []
        for audio in prompt_audios_list:
            assert len(audio.shape) == 1
            log_mel = s3tokenizer.log_mel_spectrogram(audio)  # [num_mels, T]
            prompt_speech_mels_list.append(log_mel)
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(prompt_speech_mels_list)
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.to(self.device), prompt_mels_lens_for_llm.to(self.device)
        )
        for i in range(len(prompt_speech_tokens)):
            speech_tokens_i = prompt_speech_tokens[i, :prompt_speech_tokens_lens[i].item()].tolist()
            prompt_speech_tokens_list.append(speech_tokens_i)
        return prompt_speech_tokens_list

    def get_spk_emb(self, prompt_audios_list: list[torch.Tensor]) -> torch.Tensor:
        spk_emb_for_flow = []
        for audio in prompt_audios_list:
            assert len(audio.shape) == 1
            spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
            spk_emb = self.forward_spk_embedding(spk_feat)

            spk_emb_for_flow.append(spk_emb)
        spk_emb_for_flow = torch.tensor(spk_emb_for_flow)
        if self.dtype != torch.float32:
            spk_emb_for_flow = spk_emb_for_flow.to(self.dtype)
        return spk_emb_for_flow

    def get_prompt_mels(self, prompt_audios_list: list[torch.Tensor], prompt_audios_sample_rate: list[int]):
        prompt_mels_for_flow = []
        prompt_mels_lens_for_flow = []
        for audio, sample_rate in zip(prompt_audios_list, prompt_audios_sample_rate):
            assert len(audio.shape) == 1
            audio = audio.unsqueeze(0)
            if sample_rate != 24000:
                audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
            mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
            mel_len = mel.shape[0]
            prompt_mels_for_flow.append(mel)
            prompt_mels_lens_for_flow.append(mel_len)
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
            prompt_mels_for_flow, batch_first=True, padding_value=0
        )  # [B, T', num_mels=80]
        prompt_mels_lens_for_flow = torch.tensor(prompt_mels_lens_for_flow)
        return prompt_mels_for_flow, prompt_mels_lens_for_flow

    def forward_flow(self, prompt_speech_tokens_list: list[list[int]],
                     generated_speech_tokens_list: list[list[int]],
                     prompt_mels_for_flow: torch.Tensor,
                     prompt_mels_lens_for_flow: torch.Tensor,
                     spk_emb_for_flow: torch.Tensor):
        batch_size = prompt_mels_for_flow.shape[0]
        flow_inputs = []
        flow_inputs_lens = []
        for prompt_speech_tokens, generated_speech_tokens in zip(prompt_speech_tokens_list, generated_speech_tokens_list):
            flow_inputs.append(torch.tensor(prompt_speech_tokens + generated_speech_tokens))
            flow_inputs_lens.append(len(prompt_speech_tokens) + len(generated_speech_tokens))

        flow_inputs = torch.nn.utils.rnn.pad_sequence(flow_inputs, batch_first=True, padding_value=0)
        flow_inputs_lens = torch.tensor(flow_inputs_lens)

        with torch.amp.autocast(self.device, dtype=torch.float16):
            generated_mels, generated_mels_lens = self.flow.inference(
                flow_inputs.to(self.device), flow_inputs_lens.to(self.device),
                prompt_mels_for_flow.to(self.device), prompt_mels_lens_for_flow.to(self.device), spk_emb_for_flow.to(self.device), 10
            )

        return generated_mels, generated_mels_lens

    def forward_hift(self, generated_mels: torch.Tensor, generated_mels_lens: torch.Tensor, prompt_mels_lens_for_flow: torch.Tensor):
        batch_size = generated_mels.shape[0]
        generated_wavs = []
        for i in range(batch_size):
            mel = generated_mels[i, :, prompt_mels_lens_for_flow[i].item():generated_mels_lens[i].item()].unsqueeze(0)
            wav, _ = self.hift(speech_feat=mel)
            generated_wavs.append(wav)
        return generated_wavs

    @torch.inference_mode()
    def forward(
        self, generated_speech_tokens_list: list[list[int]], prompt_audios_list: list[torch.Tensor], prompt_audios_sample_rate: list[int]
    ):
        assert all(sample_rate == 16000 for sample_rate in prompt_audios_sample_rate)

        prompt_speech_tokens_list, prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow = self.prepare_prompt_audio(prompt_audios_list, prompt_audios_sample_rate)

        generated_mels, generated_mels_lens = self.forward_flow(
            prompt_speech_tokens_list, generated_speech_tokens_list,
            prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow
        )

        generated_wavs = self.forward_hift(generated_mels, generated_mels_lens, prompt_mels_lens_for_flow)
        return generated_wavs

    def prepare_prompt_audio(
        self, prompt_audios_list: list[torch.Tensor], prompt_audios_sample_rate: list[int]
    ):
        assert all(sample_rate == 16000 for sample_rate in prompt_audios_sample_rate)

        prompt_speech_tokens_list = self.prompt_audio_tokenization(prompt_audios_list)

        prompt_mels_for_flow, prompt_mels_lens_for_flow = self.get_prompt_mels(prompt_audios_list, prompt_audios_sample_rate)

        spk_emb_for_flow = self.get_spk_emb(prompt_audios_list)
        return prompt_speech_tokens_list, prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow

    def get_prompt_audio_cache_for_streaming_tts(
        self, prompt_speech_tokens_list, prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow
    ):
        assert len(prompt_speech_tokens_list) == 1, "only support batch size 1 for streaming tts"
        for i, prompt_speech_tokens in enumerate(prompt_speech_tokens_list):
            prompt_speech_tokens_list[i] = torch.tensor(prompt_speech_tokens + prompt_speech_tokens_list[i][:3])
        prompt_speech_tokens_tensor = torch.nn.utils.rnn.pad_sequence(prompt_speech_tokens_list, batch_first=True, padding_value=0)

        cache = self.flow.setup_cache(
            prompt_speech_tokens_tensor.to(self.device),
            prompt_mels_for_flow.to(self.device),
            spk_emb_for_flow.to(self.device),
            n_timesteps=10
        )
        new_cache = {k: v.clone() for k, v in cache.items()}
        # Hack: this is a hack to avoid in-place changes to the cache['estimator_att_cache'] and cache['estimator_cnn_cache']
        return new_cache

    @torch.inference_mode()
    def forward_streaming(
        self, generated_speech_tokens: list[int], last_chunk: bool, request_id: str, speaker_id: str, prompt_audio: torch.Tensor = None, prompt_audio_sample_rate: int = 16000
    ):
        if speaker_id not in self.speaker_cache:
            assert prompt_audio is not None, "prompt_audio is required for new speaker"
            assert prompt_audio_sample_rate == 16000

            prompt_speech_tokens_list, prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow = self.prepare_prompt_audio([prompt_audio], [prompt_audio_sample_rate])

            token_len = min(int(prompt_mels_for_flow.shape[1] / 2), len(prompt_speech_tokens_list[0]))
            prompt_mels_for_flow = prompt_mels_for_flow[:, :2 * token_len].contiguous()
            prompt_speech_tokens_list[0] = prompt_speech_tokens_list[0][:token_len]

            prompt_audio_dict = {'spk_emb_for_flow': spk_emb_for_flow, 'prompt_mels_for_flow': prompt_mels_for_flow}

            cache_dict = self.get_prompt_audio_cache_for_streaming_tts(prompt_speech_tokens_list, prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow)
            self.speaker_cache[speaker_id] = {'prompt_audio_dict': prompt_audio_dict, 'cache_dict': cache_dict}

        if request_id not in self.streaming_flow_cache:
            self.streaming_flow_cache[request_id] = {k: v.clone() for k, v in self.speaker_cache[speaker_id]['cache_dict'].items()}
            self.hift_cache_dict[request_id] = dict(
                mel=torch.zeros(1, 80, 0, device='cuda'),
                source=torch.zeros(1, 1, 0, device='cuda'),
                speech=torch.zeros(1, 0, device='cuda'),
            )

        current_request_cache = self.streaming_flow_cache[request_id]

        current_prompt_audio_dict = self.speaker_cache[speaker_id]['prompt_audio_dict']
        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')

        chunk_mel, new_streaming_flow_cache = self.flow.inference_chunk(
            token=generated_speech_tokens,
            spk=current_prompt_audio_dict['spk_emb_for_flow'].to(self.device),
            cache=current_request_cache,
            last_chunk=last_chunk,
            n_timesteps=10,
        )

        self.streaming_flow_cache[request_id] = new_streaming_flow_cache

        if self.streaming_flow_cache[request_id]['estimator_att_cache'].shape[4] > (current_prompt_audio_dict['prompt_mels_for_flow'].shape[1] + 100):
            self.streaming_flow_cache[request_id]['estimator_att_cache'] = torch.cat([
                self.streaming_flow_cache[request_id]['estimator_att_cache'][:, :, :, :, :current_prompt_audio_dict['prompt_mels_for_flow'].shape[1]],
                self.streaming_flow_cache[request_id]['estimator_att_cache'][:, :, :, :, -100:],
            ], dim=4)

        hift_cache_mel = self.hift_cache_dict[request_id]['mel'].clone()
        hift_cache_source = self.hift_cache_dict[request_id]['source'].clone()
        hift_cache_speech = self.hift_cache_dict[request_id]['speech'].clone()
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2).clone()

        speech, source = self.hift(mel, hift_cache_source)

        # overlap speech smooth
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # update vocoder cache
        self.hift_cache_dict[request_id] = dict(
            mel=mel[..., -self.mel_cache_len:].clone().detach(),
            source=source[:, :, -self.source_cache_len:].clone().detach(),
            speech=speech[:, -self.source_cache_len:].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, :-self.source_cache_len]

        if last_chunk:
            assert request_id in self.streaming_flow_cache
            self.streaming_flow_cache.pop(request_id)
            self.hift_cache_dict.pop(request_id)

        return speech


def collate_fn(batch):
    ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = [], [], [], []
    for item in batch:
        generated_speech_tokens_list.append(item['target_audio_cosy2_tokens'])
        audio = torch.from_numpy(item['prompt_audio']['array']).float()
        prompt_audios_list.append(audio)
        prompt_audios_sample_rate.append(item['prompt_audio']['sampling_rate'])
        ids.append(item['id'])

    return ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-trt", action="store_true")
    parser.add_argument("--model-dir", type=str, default="./Step-Audio-2-mini/token2wav")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="generated_wavs")
    parser.add_argument("--huggingface-dataset-split", type=str, default="wenetspeech4tts")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup epochs, performance statistics will only be collected from the last epoch")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = CosyVoice2_Token2Wav(model_dir=args.model_dir, enable_trt=args.enable_trt)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset_name = "yuekai/seed_tts_cosy2"

    dataset = load_dataset(dataset_name, split=args.huggingface_dataset_split, trust_remote_code=True)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    for _ in range(args.warmup):
        start_time = time.time()
        for batch in data_loader:
            ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = batch

            generated_wavs = model(generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate)

            for id, wav in zip(ids, generated_wavs):
                torchaudio.save(f"{args.output_dir}/{id}.wav", wav.cpu(), 24000)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Measurement epoch time taken: {epoch_time:.4f} seconds")
