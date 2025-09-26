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
import os

import logging
from typing import List, Dict

import torch
from torch.utils.dlpack import to_dlpack
from torch.nn import functional as F

import triton_python_backend_utils as pb_utils

from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper
from collections import defaultdict
import numpy as np
from .token2wav_dit import CosyVoice2_Token2Wav
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)

def get_spk_id_from_prompt_audio(tensor: torch.Tensor) -> str:
    """
    Generates a unique ID for a torch.Tensor.
    Tensors with the same elements and properties will have the same ID.
    """
    # Convert tensor to a byte string
    tensor_bytes = tensor.numpy().tobytes()

    # Create a SHA-256 hash of the byte string
    hasher = hashlib.sha256()
    hasher.update(tensor_bytes)
    
    return hasher.hexdigest()

class TritonPythonModel:
    """Triton Python model for vocoder.

    This model takes global and semantic tokens as input and generates audio waveforms
    using the BiCodec vocoder.
    """

    def initialize(self, args):
        """Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {key: value["string_value"] for key, value in parameters.items()}
        model_dir = model_params["model_dir"]

        # Initialize device and vocoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing vocoder from {model_dir} on {self.device}")

        # FIXME: device id settings
        self.token2wav_model = CosyVoice2_Token2Wav(
            model_dir, enable_trt=True, streaming=True
        )
        logger.info("Token2Wav initialized successfully")

    def execute(self, requests):
        """Execute inference on the batched requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses containing generated waveforms
        """
        responses = []
        for request in requests:
            request_id = request.request_id()

            # Get inputs
            target_speech_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "target_speech_tokens")
            target_speech_tokens = torch.utils.dlpack.from_dlpack(target_speech_tokens_tensor.to_dlpack())
            target_speech_tokens = target_speech_tokens.squeeze().tolist()

            finalize = pb_utils.get_input_tensor_by_name(request, "finalize").as_numpy().item()
            wav_array = pb_utils.get_input_tensor_by_name(request, "reference_wav").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len").as_numpy().item()
            wav = torch.from_numpy(wav_array)[:, :wav_len].squeeze(0)
            spk_id = get_spk_id_from_prompt_audio(wav)

            # Handle cache
            conformer_cnn_cache = pb_utils.get_input_tensor_by_name(request, "conformer_cnn_cache")
            if conformer_cnn_cache is not None:
                self.token2wav_model.streaming_flow_cache[request_id]['conformer_cnn_cache'] = torch.utils.dlpack.from_dlpack(conformer_cnn_cache.to_dlpack())
                
                conformer_att_cache_np = pb_utils.get_input_tensor_by_name(request, "conformer_att_cache")
                self.token2wav_model.streaming_flow_cache[request_id]['conformer_att_cache'] = torch.utils.dlpack.from_dlpack(conformer_att_cache_np.to_dlpack()).transpose(0,1)
                
                estimator_cnn_cache_np = pb_utils.get_input_tensor_by_name(request, "estimator_cnn_cache")
                self.token2wav_model.streaming_flow_cache[request_id]['estimator_cnn_cache'] = torch.utils.dlpack.from_dlpack(estimator_cnn_cache_np.to_dlpack()).squeeze(0)

                estimator_att_cache_np = pb_utils.get_input_tensor_by_name(request, "estimator_att_cache")
                self.token2wav_model.streaming_flow_cache[request_id]['estimator_att_cache'] = torch.utils.dlpack.from_dlpack(estimator_att_cache_np.to_dlpack()).squeeze(0)

                mel_np = pb_utils.get_input_tensor_by_name(request, "mel")
                self.token2wav_model.streaming_flow_cache[request_id]['mel'] = torch.utils.dlpack.from_dlpack(mel_np.to_dlpack())
                
                source_np = pb_utils.get_input_tensor_by_name(request, "source")
                self.token2wav_model.hift_cache_dict[request_id]['source'] = torch.utils.dlpack.from_dlpack(source_np.to_dlpack())
                
                speech_np = pb_utils.get_input_tensor_by_name(request, "speech")
                self.token2wav_model.hift_cache_dict[request_id]['speech'] = torch.utils.dlpack.from_dlpack(speech_np.to_dlpack())

            # Forward pass
            audio_hat = self.token2wav_model.forward_streaming(
                target_speech_tokens, 
                finalize, 
                request_id=request_id, 
                speaker_id=f"{spk_id}", 
                prompt_audio=wav, 
                prompt_audio_sample_rate=16000
            )
            
            # Prepare outputs
            outputs = []
            wav_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio_hat))
            outputs.append(wav_tensor)
            
            if request_id in self.token2wav_model.streaming_flow_cache:
                cache = self.token2wav_model.streaming_flow_cache[request_id]
                hifigan_cache = self.token2wav_model.hift_cache_dict[request_id]
                conformer_cnn_cache = cache['conformer_cnn_cache']
                conformer_att_cache = cache['conformer_att_cache'].transpose(0,1)
                estimator_cnn_cache = cache['estimator_cnn_cache'].unsqueeze(0)
                estimator_att_cache = cache['estimator_att_cache'].unsqueeze(0)
                mel = hifigan_cache['mel']
                source = hifigan_cache['source']
                speech = hifigan_cache['speech']

                outputs.extend([
                    pb_utils.Tensor.from_dlpack("conformer_cnn_cache", to_dlpack(conformer_cnn_cache)),
                    pb_utils.Tensor.from_dlpack("conformer_att_cache", to_dlpack(conformer_att_cache)),
                    pb_utils.Tensor.from_dlpack("estimator_cnn_cache", to_dlpack(estimator_cnn_cache)),
                    pb_utils.Tensor.from_dlpack("estimator_att_cache", to_dlpack(estimator_att_cache)),
                    pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel)),
                    pb_utils.Tensor.from_dlpack("source", to_dlpack(source)),
                    pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech)),
                ])
            else:
                outputs.extend([pb_utils.Tensor("conformer_cnn_cache", np.array([], dtype=np.float16)),
                pb_utils.Tensor("conformer_att_cache", np.array([], dtype=np.float16)),
                pb_utils.Tensor("estimator_cnn_cache", np.array([], dtype=np.float16)),
                pb_utils.Tensor("estimator_att_cache", np.array([], dtype=np.float16)),
                pb_utils.Tensor("mel", np.array([], dtype=np.float32)),
                pb_utils.Tensor("source", np.array([], dtype=np.float32)),
                pb_utils.Tensor("speech", np.array([], dtype=np.float32)),
                ])

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)
        return responses

    def finalize(self):
        self.logger.log_info("Finalizing Token2WavDiT model")
