# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import math
import os
import re
import time
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import httpx

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

import torchaudio


from matcha.utils.audio import mel_spectrogram


ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)


def parse_speech_token_string(response_text: str) -> List[int]:
    """
    Parses a string of speech tokens (e.g., "<|s_123|><|s_456|>") into a list of integer IDs.
    """
    speech_tokens = response_text.strip().split('><')
    if len(speech_tokens) > 1:
        # Add back the missing '<' and '>' for proper parsing
        speech_tokens = ['<' + t if not t.startswith('<') else t for t in speech_tokens]
        speech_tokens = [t + '>' if not t.endswith('>') else t for t in speech_tokens]

    speech_ids = []
    for token_str in speech_tokens:
        match = re.match(r'<\|s_(\d+)\|>', token_str)
        if match:
            speech_ids.append(int(match.group(1)))
    return speech_ids


class TritonPythonModel:
    """Triton Python model for Spark TTS.

    This model orchestrates the end-to-end TTS pipeline by coordinating
    between audio tokenizer, LLM, and vocoder components.
    """

    def initialize(self, args):
        """Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        self.logger = pb_utils.Logger
        # Parse model parameters
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        self.dynamic_chunk_strategy = model_params.get("dynamic_chunk_strategy", "exponential")  # "exponential" or "time_based"
        self.logger.log_info(f"Using dynamic chunk strategy: {self.dynamic_chunk_strategy}")

        # Initialize tokenizer
        llm_tokenizer_dir = model_params["llm_tokenizer_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)
        self.prompt_template = "<|sos|>{input_text}<|task_id|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eos1|>")

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 3
        self.token_hop_len = 15

        self.http_client = httpx.AsyncClient()
        self.api_base = "http://localhost:8000/v1/chat/completions"
        self.speaker_cache = {}

    def _convert_speech_tokens_to_str(self, speech_tokens: Union[torch.Tensor, List]) -> str:
        """Converts a tensor or list of speech token IDs to a string representation."""
        if isinstance(speech_tokens, torch.Tensor):
            # Ensure tensor is on CPU and flattened
            speech_tokens = speech_tokens.cpu().numpy().flatten().tolist()

        speech_id_str = ""
        for token_id in speech_tokens:
            # Convert token ID back to the speech number N
            token_num = token_id - ORIGINAL_VOCAB_SIZE
            speech_id_str += f"<|s_{token_num}|>"
        return speech_id_str

    async def forward_llm_async(self, target_text: str, reference_text: str, prompt_speech_tokens: Union[torch.Tensor, List]):
        """
        Asynchronously sends a request to the TRTLLM-serve endpoint and processes the streaming response.
        """
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
                                final_id = token_num + ORIGINAL_VOCAB_SIZE
                                yield final_id
                                buffer = buffer[match.end():]
                    except json.JSONDecodeError:
                        self.logger.log_info(f"Skipping non-JSON line: {line_data}")
                        continue

        # Process any remaining complete tokens in the buffer after the stream ends
        while True:
            match = re.search(r"<\|s_(\d+)\|>", buffer)
            if not match:
                break
            token_num = int(match.group(1))
            final_id = token_num + ORIGINAL_VOCAB_SIZE
            yield final_id
            buffer = buffer[match.end():]

    def forward_audio_tokenizer(self, wav, wav_len):
        """Forward pass through the audio tokenizer component.

        Args:
            wav: Input waveform tensor
            wav_len: Waveform length tensor

        Returns:
            Tuple of global and semantic tokens
        """
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        # Extract and convert output tensors
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(inference_response, 'prompt_speech_tokens')
        prompt_speech_tokens = torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

        return prompt_speech_tokens

    def forward_speaker_embedding(self, wav):
        """Forward pass through the speaker embedding component.

        Args:
            wav: Input waveform tensor

        Returns:
            Prompt speaker embedding tensor
        """
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack("reference_wav", to_dlpack(wav))]
        )

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        # Extract and convert output tensors
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(inference_response, 'prompt_spk_embedding')
        prompt_spk_embedding = torch.utils.dlpack.from_dlpack(prompt_spk_embedding.to_dlpack())

        return prompt_spk_embedding

    async def forward_token2wav(
            self,
            index: int,
            target_speech_tokens: torch.Tensor,
            request_id: str,
            reference_wav: object,
            reference_wav_len: object,
            finalize: bool = None) -> torch.Tensor:
        """Forward pass through the vocoder component.

        Args:
            index: Index of the request
            target_speech_tokens: Target speech tokens tensor
            request_id: Request ID
            reference_wav: Reference waveform tensor
            reference_wav_len: Reference waveform length tensor
            finalize: Whether to finalize the request

        Returns:
            Generated waveform tensor
        """
        target_speech_tokens_tensor = pb_utils.Tensor.from_dlpack("target_speech_tokens", to_dlpack(target_speech_tokens))
        finalize_tensor = pb_utils.Tensor("finalize", np.array([[finalize]], dtype=np.bool_))
        inputs_tensor = [target_speech_tokens_tensor, reference_wav, reference_wav_len, finalize_tensor]

        # Create and execute inference request
        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav_dit',
            requested_output_names=[
                "waveform",
            ],
            inputs=inputs_tensor,
            request_id=request_id,
            parameters={"priority": index + 1},
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        # Extract and convert output waveform
        waveform = pb_utils.get_output_tensor_by_name(inference_response, 'waveform')
        waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()

        return waveform

    def _extract_speech_feat(self, speech):
        speech_feat = mel_spectrogram(
            speech,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000).squeeze(
            dim=0).transpose(
            0,
            1).to(
                self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        return speech_feat

    async def _process_request(self, request):
        request_id = request.request_id()

        reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
        reference_text = reference_text[0][0].decode('utf-8')

        wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")

        if reference_text not in self.speaker_cache:
            self.speaker_cache[reference_text] = self.forward_audio_tokenizer(wav, wav_len).unsqueeze(0)
        prompt_speech_tokens = self.speaker_cache[reference_text]

        target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
        target_text = target_text[0][0].decode('utf-8')

        if self.decoupled:
            response_sender = request.get_response_sender()

            semantic_token_ids_arr = []
            token_offset, chunk_index = 0, 0
            start_time = time.time()
            this_token_hop_len = self.token_hop_len
            async for generated_ids in self.forward_llm_async(
                target_text=target_text,
                reference_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens,
            ):
                if not generated_ids:
                    break
                semantic_token_ids_arr.append(generated_ids)
                while True:
                    pending_num = len(semantic_token_ids_arr) - token_offset
                    if pending_num >= this_token_hop_len + self.flow_pre_lookahead_len:
                        this_tts_speech_token = semantic_token_ids_arr[token_offset:token_offset + this_token_hop_len + self.flow_pre_lookahead_len]
                        this_tts_speech_token = torch.tensor(this_tts_speech_token).unsqueeze(dim=0).to(torch.int32).to(self.device)
                        sub_tts_speech = await self.forward_token2wav(
                            chunk_index,
                            this_tts_speech_token, request_id, wav, wav_len, False
                        )
                        audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(sub_tts_speech))
                        inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
                        response_sender.send(inference_response)

                        token_offset += this_token_hop_len

                        if self.dynamic_chunk_strategy == "exponential":
                            this_token_hop_len = self.token_frame_rate * (2 ** chunk_index)
                        elif self.dynamic_chunk_strategy == "equal":
                            this_token_hop_len = self.token_hop_len
                        elif self.dynamic_chunk_strategy == "time_based":
                            # see https://github.com/qi-hua/async_cosyvoice/blob/main/model.py#L306
                            cost_time = time.time() - start_time
                            duration = token_offset / self.token_frame_rate
                            if chunk_index > 0 and cost_time > 0:
                                avg_chunk_processing_time = cost_time / (chunk_index + 1)
                                if avg_chunk_processing_time > 0:
                                    multiples = (duration - cost_time) / avg_chunk_processing_time
                                    next_pending_num = len(semantic_token_ids_arr) - token_offset
                                    if multiples > 4:
                                        this_token_hop_len = (next_pending_num // self.token_hop_len + 1) * self.token_hop_len
                                    elif multiples > 2:
                                        this_token_hop_len = (next_pending_num // self.token_hop_len) * self.token_hop_len
                                    else:
                                        this_token_hop_len = self.token_hop_len
                                    this_token_hop_len = max(self.token_hop_len, this_token_hop_len)
                        chunk_index += 1
                    else:
                        break

            this_tts_speech_token = torch.tensor(semantic_token_ids_arr[token_offset:]).unsqueeze(dim=0).to(torch.int32).to(self.device)
            sub_tts_speech = await self.forward_token2wav(chunk_index, this_tts_speech_token, request_id, wav, wav_len, True)
            audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(sub_tts_speech))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
            response_sender.send(inference_response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        else:
            raise NotImplementedError("Offline TTS mode is not supported")

    async def execute(self, requests):
        """Execute inference on the batched requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses containing generated audio
        """
        tasks = [
            asyncio.create_task(self._process_request(request))
            for request in requests
        ]
        await asyncio.gather(*tasks)
        return None

    def finalize(self):
        self.logger.log_info("Finalizing CosyVoice DIT model")
        if hasattr(self, "http_client"):
            asyncio.run(self.http_client.aclose())
