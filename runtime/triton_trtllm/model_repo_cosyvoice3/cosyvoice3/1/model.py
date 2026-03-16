import json
import re
import time
import asyncio

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils

import httpx
import torchaudio
from functools import partial
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram


torch.set_num_threads(1)

# CosyVoice3 mel params: fmax=None (Nyquist), center=False
mel_spectrogram = partial(matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)


def parse_speech_token_string(response_text):
    """Parse speech tokens from string like '<|s_123|><|s_456|>' into list of int IDs."""
    speech_tokens = response_text.strip().split('><')
    if len(speech_tokens) > 1:
        speech_tokens = ['<' + t if not t.startswith('<') else t for t in speech_tokens]
        speech_tokens = [t + '>' if not t.endswith('>') else t for t in speech_tokens]
    speech_ids = []
    for token_str in speech_tokens:
        match = re.match(r'<\|s_(\d+)\|>', token_str)
        if match:
            speech_ids.append(int(match.group(1)))
    return speech_ids


class TritonPythonModel:
    """CosyVoice3 BLS orchestrator for Triton Inference Server.

    Orchestrates: audio_tokenizer, speaker_embedding, remote LLM (httpx),
    token2wav (flow-only), and vocoder (CausalHiFTGenerator).
    Supports both streaming (decoupled) and offline (non-decoupled) modes.
    """

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        # Streaming config
        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 3
        self.token_hop_len = 15
        self.token_mel_ratio = 2
        self.dynamic_chunk_strategy = model_params.get("dynamic_chunk_strategy", "exponential")
        self.logger.log_info(f"CosyVoice3 BLS initialized, decoupled={self.decoupled}, "
                             f"chunk_strategy={self.dynamic_chunk_strategy}")

        # HTTP client for remote LLM (trtllm-serve default port: 8000)
        self.http_client = httpx.AsyncClient()
        self.api_base = model_params.get("llm_api_base", "http://localhost:8000/v1/chat/completions")

        # Speaker cache to avoid redundant audio_tokenizer/speaker_embedding calls
        self.speaker_cache = {}

    def _convert_speech_tokens_to_str(self, speech_tokens):
        """Convert speech token IDs tensor/list to string like '<|s_N|>'."""
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.cpu().numpy().flatten().tolist()
        return "".join(f"<|s_{int(tid)}|>" for tid in speech_tokens)

    def _extract_speech_feat(self, speech):
        """Extract mel spectrogram from 24kHz speech for flow prompt."""
        speech_feat = mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0).to(self.device)
        return speech_feat

    async def forward_llm_streaming(self, target_text, reference_text, prompt_speech_tokens):
        """Async generator: stream LLM tokens via httpx SSE."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 750,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": True,
        }

        buffer = ""
        async with self.http_client.stream("POST", self.api_base, json=payload, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line_data = line[len("data: "):].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(line_data)
                        content = json_data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            buffer += content
                            while True:
                                match = re.search(r"<\|s_(\d+)\|>", buffer)
                                if not match:
                                    break
                                token_num = int(match.group(1))
                                # final_id = token_num + ORIGINAL_VOCAB_SIZE
                                yield token_num
                                buffer = buffer[match.end():]
                    except json.JSONDecodeError:
                        continue

        # Flush remaining tokens
        while True:
            match = re.search(r"<\|s_(\d+)\|>", buffer)
            if not match:
                break
            token_num = int(match.group(1))
            #final_id = token_num + ORIGINAL_VOCAB_SIZE
            yield token_num
            buffer = buffer[match.end():]

    async def forward_llm_offline(self, target_text, reference_text, prompt_speech_tokens):
        """Non-streaming LLM call, returns all speech token IDs at once."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 750,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": False,
        }
        response = await self.http_client.post(self.api_base, json=payload, timeout=None)
        response.raise_for_status()
        response_json = response.json()
        generated_content = response_json['choices'][0]['message']['content']
        speech_ids = parse_speech_token_string(generated_content)
        # return [sid + ORIGINAL_VOCAB_SIZE for sid in speech_ids]
        return speech_ids

    def forward_audio_tokenizer(self, wav, wav_len):
        """BLS call to audio_tokenizer."""
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_speech_tokens')
        return torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

    def forward_speaker_embedding(self, wav):
        """BLS call to speaker_embedding."""
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack("reference_wav", to_dlpack(wav))]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_spk_embedding')
        return torch.utils.dlpack.from_dlpack(prompt_spk_embedding.to_dlpack())

    async def forward_token2wav(self, target_speech_tokens, prompt_speech_tokens,
                                prompt_speech_feat, prompt_spk_embedding,
                                request_id, token_offset=None, finalize=True,
                                priority=100):
        """Async BLS call to token2wav (flow-only). Returns mel tensor."""
        target_tokens_pb = pb_utils.Tensor.from_dlpack(
            "target_speech_tokens", to_dlpack(target_speech_tokens))
        prompt_tokens_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
        prompt_feat_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_feat", to_dlpack(prompt_speech_feat))
        prompt_emb_pb = pb_utils.Tensor.from_dlpack(
            "prompt_spk_embedding", to_dlpack(prompt_spk_embedding))

        inputs = [target_tokens_pb, prompt_tokens_pb, prompt_feat_pb, prompt_emb_pb]

        if token_offset is not None:
            inputs.append(pb_utils.Tensor("token_offset",
                          np.array([[token_offset]], dtype=np.int32)))
            inputs.append(pb_utils.Tensor("finalize",
                          np.array([[finalize]], dtype=np.bool_)))

        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav',
            requested_output_names=['mel'],
            inputs=inputs,
            request_id=request_id,
            parameters={"priority": priority},
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        mel = pb_utils.get_output_tensor_by_name(inference_response, 'mel')
        return torch.utils.dlpack.from_dlpack(mel.to_dlpack())

    async def forward_vocoder(self, mel, finalize):
        """Async BLS call to vocoder. Returns speech tensor."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [80, T] -> [1, 80, T]
        mel_pb = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel.float()))
        finalize_pb = pb_utils.Tensor("finalize",
                      np.array([[finalize]], dtype=np.bool_))

        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['tts_speech'],
            inputs=[mel_pb, finalize_pb],
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        speech = pb_utils.get_output_tensor_by_name(inference_response, 'tts_speech')
        return torch.utils.dlpack.from_dlpack(speech.to_dlpack()).cpu()

    def _prepare_prompt(self, request):
        """Extract reference audio, tokenize, compute speaker embedding and mel feat."""
        wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")

        reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text")
        reference_text = reference_text.as_numpy()[0][0].decode('utf-8') if reference_text is not None else ""
        if '<|endofprompt|>' not in reference_text:
            reference_text = 'You are a helpful assistant.<|endofprompt|>' + reference_text

        # Check speaker cache
        if reference_text in self.speaker_cache:
            cached = self.speaker_cache[reference_text]
            return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                    cached['prompt_speech_feat'], cached['prompt_spk_embedding'], reference_text)

        # Audio tokenizer
        wav_np = wav.as_numpy()
        wav_len_val = wav_len.as_numpy()[0][0]
        prompt_speech_tokens = self.forward_audio_tokenizer(wav, wav_len)
        prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)  # [1, T]

        # Speaker embedding
        wav_tensor = torch.from_numpy(wav_np)
        wav_tensor = wav_tensor[:, :wav_len_val]
        prompt_spk_embedding = self.forward_speaker_embedding(wav_tensor)

        # Mel extraction at 24kHz with CosyVoice3 params
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000)(wav_tensor)
        speech_feat = self._extract_speech_feat(prompt_speech_resample)

        # Keep full tokens for LLM prefill (untruncated)
        prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()

        # Align prompt speech feat and tokens to 2:1 ratio (for flow model only)
        orig_feat_len = speech_feat.shape[1]
        orig_token_len = prompt_speech_tokens.shape[-1]
        token_len = min(int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
        prompt_speech_feat = speech_feat[:, :2 * token_len].contiguous().half()
        prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()

        # Cache
        self.speaker_cache[reference_text] = {
            'prompt_speech_tokens_for_llm': prompt_speech_tokens_for_llm,
            'prompt_speech_tokens': prompt_speech_tokens,
            'prompt_speech_feat': prompt_speech_feat,
            'prompt_spk_embedding': prompt_spk_embedding,
        }

        return prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, reference_text

    async def _process_request_streaming(self, request):
        """Process a single request in streaming (decoupled) mode."""
        request_id = request.request_id()
        response_sender = request.get_response_sender()

        try:
            prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
                prompt_spk_embedding, reference_text = self._prepare_prompt(request)

            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')

            semantic_token_ids_arr = []
            token_offset = 0
            chunk_index = 0
            this_token_hop_len = self.token_hop_len
            accumulated_mel = None
            speech_offset = 0
            start_time = time.time()

            async for generated_id in self.forward_llm_streaming(
                target_text=target_text,
                reference_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens_for_llm,
            ):
                semantic_token_ids_arr.append(generated_id)

                while True:
                    pending_num = len(semantic_token_ids_arr) - token_offset
                    if pending_num < this_token_hop_len + self.flow_pre_lookahead_len:
                        break

                    # Prepare tokens for this chunk
                    end_idx = token_offset + this_token_hop_len + self.flow_pre_lookahead_len
                    this_tokens = torch.tensor(
                        semantic_token_ids_arr[:end_idx]
                    ).unsqueeze(0).to(torch.int32).to(self.device)

                    # Call token2wav (flow-only) -> mel_chunk
                    mel_chunk = await self.forward_token2wav(
                        this_tokens, prompt_speech_tokens,
                        prompt_speech_feat, prompt_spk_embedding,
                        request_id, token_offset=token_offset, finalize=False,
                        priority=chunk_index + 1,
                    )

                    # Accumulate mel
                    if mel_chunk.dim() == 2:
                        mel_chunk = mel_chunk.unsqueeze(0)
                    if accumulated_mel is None:
                        accumulated_mel = mel_chunk
                    else:
                        accumulated_mel = torch.cat([accumulated_mel, mel_chunk], dim=2)

                    # Call vocoder
                    speech = await self.forward_vocoder(accumulated_mel, finalize=False)

                    # Extract new speech
                    new_speech = speech[:, speech_offset:]
                    speech_offset += new_speech.shape[1]

                    if new_speech.shape[1] > 0:
                        audio_tensor = pb_utils.Tensor.from_dlpack(
                            "waveform", to_dlpack(new_speech))
                        inference_response = pb_utils.InferenceResponse(
                            output_tensors=[audio_tensor])
                        response_sender.send(inference_response)

                    token_offset += this_token_hop_len

                    # Dynamic chunk strategy
                    if self.dynamic_chunk_strategy == "exponential":
                        this_token_hop_len = self.token_frame_rate * (2 ** chunk_index)
                    elif self.dynamic_chunk_strategy == "time_based":
                        cost_time = time.time() - start_time
                        duration = token_offset / self.token_frame_rate
                        if chunk_index > 0 and cost_time > 0:
                            avg_chunk_time = cost_time / (chunk_index + 1)
                            if avg_chunk_time > 0:
                                multiples = (duration - cost_time) / avg_chunk_time
                                next_pending = len(semantic_token_ids_arr) - token_offset
                                if multiples > 4:
                                    this_token_hop_len = (next_pending // self.token_hop_len + 1) * self.token_hop_len
                                elif multiples > 2:
                                    this_token_hop_len = (next_pending // self.token_hop_len) * self.token_hop_len
                                else:
                                    this_token_hop_len = self.token_hop_len
                                this_token_hop_len = max(self.token_hop_len, this_token_hop_len)

                    chunk_index += 1

            # Final chunk with remaining tokens
            if len(semantic_token_ids_arr) > 0:
                remaining_tokens = torch.tensor(
                    semantic_token_ids_arr
                ).unsqueeze(0).to(torch.int32).to(self.device)

                mel_chunk = await self.forward_token2wav(
                    remaining_tokens, prompt_speech_tokens,
                    prompt_speech_feat, prompt_spk_embedding,
                    request_id, token_offset=token_offset, finalize=True,
                    priority=chunk_index + 1,
                )

                if mel_chunk.dim() == 2:
                    mel_chunk = mel_chunk.unsqueeze(0)
                if accumulated_mel is None:
                    accumulated_mel = mel_chunk
                else:
                    accumulated_mel = torch.cat([accumulated_mel, mel_chunk], dim=2)

                speech = await self.forward_vocoder(accumulated_mel, finalize=True)

                new_speech = speech[:, speech_offset:]
                if new_speech.shape[1] > 0:
                    audio_tensor = pb_utils.Tensor.from_dlpack(
                        "waveform", to_dlpack(new_speech))
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[audio_tensor])
                    response_sender.send(inference_response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            self.logger.log_error(f"Error in streaming request: {e}")
            error_response = pb_utils.InferenceResponse(
                error=pb_utils.TritonError(str(e)))
            response_sender.send(error_response)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    async def _process_request_offline(self, request):
        """Process a single request in offline (non-decoupled) mode."""
        request_id = request.request_id()

        prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
            prompt_spk_embedding, reference_text = self._prepare_prompt(request)

        target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
        target_text = target_text[0][0].decode('utf-8')

        # Get all speech tokens at once (use full untruncated prompt tokens for LLM)
        all_token_ids = await self.forward_llm_offline(
            target_text=target_text,
            reference_text=reference_text,
            prompt_speech_tokens=prompt_speech_tokens_for_llm,
        )

        if len(all_token_ids) == 0:
            raise pb_utils.TritonModelException("LLM generated no speech tokens")

        all_tokens = torch.tensor(all_token_ids).unsqueeze(0).to(torch.int32).to(self.device)

        # token2wav (no token_offset, finalize=True) -> full mel
        mel = await self.forward_token2wav(
            all_tokens, prompt_speech_tokens,
            prompt_speech_feat, prompt_spk_embedding,
            request_id,
        )

        # vocoder -> full speech
        speech = await self.forward_vocoder(mel, finalize=True)

        audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(speech))
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

    async def execute(self, requests):
        if self.decoupled:
            tasks = [
                asyncio.create_task(self._process_request_streaming(request))
                for request in requests
            ]
            await asyncio.gather(*tasks)
            return None
        else:
            responses = []
            for request in requests:
                try:
                    response = await self._process_request_offline(request)
                    responses.append(response)
                except Exception as e:
                    self.logger.log_error(f"Error in offline request: {e}")
                    responses.append(pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))))
            return responses

    def finalize(self):
        self.logger.log_info("Finalizing CosyVoice3 BLS model")
        if hasattr(self, "http_client"):
            asyncio.run(self.http_client.aclose())
