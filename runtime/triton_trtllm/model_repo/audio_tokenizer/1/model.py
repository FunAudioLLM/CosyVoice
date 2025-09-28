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
import torch
from torch.utils.dlpack import to_dlpack

import triton_python_backend_utils as pb_utils

import os
import numpy as np
import s3tokenizer
torch.set_num_threads(1)
ORIGINAL_VOCAB_SIZE = 151663


class TritonPythonModel:
    """Triton Python model for audio tokenization.

    This model takes reference audio input and extracts semantic tokens
    using s3tokenizer.
    """

    def initialize(self, args):
        """Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}

        self.device = torch.device("cuda")
        model_path = os.path.join(model_params["model_dir"], "speech_tokenizer_v2.onnx")
        self.audio_tokenizer = s3tokenizer.load_model(model_path).to(self.device)

    def execute(self, requests):
        """Execute inference on the batched requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses containing tokenized outputs
        """
        mels = []

        # Process each request in batch
        for request in requests:
            # Extract input tensors
            wav_array = pb_utils.get_input_tensor_by_name(
                request, "reference_wav").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(
                request, "reference_wav_len").as_numpy().item()

            wav_array = torch.from_numpy(wav_array).to(self.device)
            # Prepare inputs
            wav = wav_array[:, :wav_len].squeeze(0)
            mels.append(s3tokenizer.log_mel_spectrogram(wav))

        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = self.audio_tokenizer.quantize(mels.to(self.device), mels_lens.to(self.device))
        codes = codes.clone() + ORIGINAL_VOCAB_SIZE

        responses = []
        for i in range(len(requests)):
            prompt_speech_tokens = codes[i, :codes_lens[i].item()]
            prompt_speech_tokens_tensor = pb_utils.Tensor.from_dlpack(
                "prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[prompt_speech_tokens_tensor])
            responses.append(inference_response)

        return responses
