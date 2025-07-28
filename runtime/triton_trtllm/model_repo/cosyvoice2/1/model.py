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
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import onnxruntime


from matcha.utils.audio import mel_spectrogram

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
        self.logger.log_info(f"model_params:{model_params}")

        # Initialize tokenizer
        llm_tokenizer_dir = model_params["llm_tokenizer_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)
        self.prompt_template = "<|sos|>{input_text}<|task_id|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eos1|>")

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        campplus_model = f'{model_params["model_dir"]}/campplus.onnx'
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])

    def forward_llm(self, input_ids):
        """
        Prepares the response from the language model based on the provided
        inputs. Creates a `pb_utils.InferenceRequest` object with passed
        `llm_request_inputs` to send to a decoupled TensorRTLLM model.
        For each response from the language model:
            - Checks for errors and raise an exception if any are found.
            - Extracts the "output_ids" tensor from the response.
            - Determines the finish reason based on the presence of the
              end-of-sequence token or reaching the maximum length.
            - Appends the generated token IDs to `output_ids`.
            - If the finish reason is determined, decodes the output IDs to text
              and prepares the final response.

        The final response includes the generated text, finish reason,
        completion tokens, prompt tokens, and total tokens.

        Parameters
        ----------
        - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

        Returns
        -------
        - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
        """
        # convert input_ids to numpy, with shape [1, sequence_length]
        input_ids = input_ids.cpu().numpy()
        max_tokens = 1024
        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.eos_token_id]], dtype=np.int32),
            "pad_id": np.array([[self.eos_token_id]], dtype=np.int32),
            "streaming": np.array([[self.decoupled]], dtype=np.bool_),
            "runtime_top_p": np.array([[0.95]], dtype=np.float32),
            "runtime_top_k": np.array([[50]], dtype=np.int32),
            "temperature": np.array([[0.8]], dtype=np.float32),
            "input_ids": input_ids,
            "input_lengths": np.array([[input_ids.shape[1]]], dtype=np.int32),
        }
        
        # Convert inputs to Triton tensors
        input_tensor_list = [
            pb_utils.Tensor(k, v) for k, v in input_dict.items()
        ]
        
        # Create and execute inference request
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=input_tensor_list,
        )
        
        llm_responses = llm_request.exec(decoupled=self.decoupled)
        if self.decoupled:
            for llm_response in llm_responses:
                if llm_response.has_error():
                    raise pb_utils.TritonModelException(llm_response.error().message())
                
                # Extract and process output
                output_ids = pb_utils.get_output_tensor_by_name(
                    llm_response, "output_ids").as_numpy()
                seq_lens = pb_utils.get_output_tensor_by_name(
                    llm_response, "sequence_length").as_numpy()
                
                # Get actual output IDs up to the sequence length
                actual_output_ids = output_ids[0][0][:seq_lens[0][0]]
                
                yield actual_output_ids
        else:
            llm_response = llm_responses
            if llm_response.has_error():
                raise pb_utils.TritonModelException(llm_response.error().message())
            
            # Extract and process output
            output_ids = pb_utils.get_output_tensor_by_name(
                llm_response, "output_ids").as_numpy()
            seq_lens = pb_utils.get_output_tensor_by_name(
                llm_response, "sequence_length").as_numpy()
            
            # Get actual output IDs up to the sequence length
            actual_output_ids = output_ids[0][0][:seq_lens[0][0]]
            
            yield actual_output_ids    
                
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

    def forward_token2wav(self, prompt_speech_tokens: torch.Tensor, prompt_speech_feat: torch.Tensor, prompt_spk_embedding: torch.Tensor, target_speech_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vocoder component.
        
        Args:
            prompt_speech_tokens: Prompt speech tokens tensor
            prompt_speech_feat: Prompt speech feat tensor
            prompt_spk_embedding: Prompt spk embedding tensor
            target_speech_tokens: Target speech tokens tensor
            
        Returns:
            Generated waveform tensor
        """
        prompt_speech_tokens_tensor = pb_utils.Tensor.from_dlpack("prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
        prompt_speech_feat_tensor = pb_utils.Tensor.from_dlpack("prompt_speech_feat", to_dlpack(prompt_speech_feat))
        prompt_spk_embedding_tensor = pb_utils.Tensor.from_dlpack("prompt_spk_embedding", to_dlpack(prompt_spk_embedding))
        target_speech_tokens_tensor = pb_utils.Tensor.from_dlpack("target_speech_tokens", to_dlpack(target_speech_tokens))
        
        # Create and execute inference request
        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav',
            requested_output_names=['waveform'],
            inputs=[prompt_speech_tokens_tensor, prompt_speech_feat_tensor, prompt_spk_embedding_tensor, target_speech_tokens_tensor]
        )
        
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        
        # Extract and convert output waveform
        waveform = pb_utils.get_output_tensor_by_name(inference_response, 'waveform')
        waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()
        
        return waveform

    def parse_input(self, text, prompt_text, prompt_speech_tokens):
        total_text = f"{prompt_text}{text}"
        prompt = self.prompt_template.format(input_text=total_text)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.int32)
        input_ids = torch.cat([input_ids, prompt_speech_tokens], dim=1)
        return input_ids

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                                {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device).half()
        return embedding


    def _extract_speech_feat(self, speech):
        speech_feat = mel_spectrogram(speech, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        return speech_feat

    def execute(self, requests):
        """Execute inference on the batched requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses containing generated audio
        """
        responses = []
        
        for request in requests:
            # Extract input tensors
            wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
            
            # Process reference audio through audio tokenizer

            prompt_speech_tokens = self.forward_audio_tokenizer(wav, wav_len)
            prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)


            wav_tensor = wav.as_numpy()
            wav_tensor = torch.from_numpy(wav_tensor)[:, :wav_len.as_numpy()[0][0]]
            prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(wav_tensor)
            speech_feat = self._extract_speech_feat(prompt_speech_resample)
            token_len = min(int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
            prompt_speech_feat = speech_feat[:, :2 * token_len].contiguous().half()
            prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()
            
            reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode('utf-8')
            
            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')
            
            # Prepare prompt for LLM
            input_ids = self.parse_input(
                text=target_text,
                prompt_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens,
            )
            
            # Generate semantic tokens with LLM
            generated_ids_iter = self.forward_llm(input_ids)

            if self.decoupled:
                response_sender = request.get_response_sender()
                request_id = request.request_id()
                for generated_ids in generated_ids_iter:
                    raise NotImplementedError("Decoupled mode is not implemented")
            else:
                generated_ids = next(generated_ids_iter)
                generated_ids = torch.tensor([generated_ids]).to(self.device)
                if generated_ids is None or len(generated_ids) == 0:
                    raise pb_utils.TritonModelException("Generated IDs is None or empty")

                prompt_spk_embedding = self._extract_spk_embedding(wav_tensor)
                audio = self.forward_token2wav(prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, generated_ids)
                
                # Prepare response
                audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
                inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
                responses.append(inference_response)
            
            if self.decoupled:
                response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                self.logger.log_info(f"send tritonserver_response_complete_final to end")
        
        if not self.decoupled:
            return responses