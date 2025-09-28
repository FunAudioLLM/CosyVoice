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
import torchaudio.compliance.kaldi as kaldi
from cosyvoice.utils.file_utils import convert_onnx_to_trt
from cosyvoice.utils.common import TrtContextWrapper
import onnxruntime


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

        model_dir = model_params["model_dir"]
        gpu = "l20"
        enable_trt = True
        if enable_trt:
            self.load_spk_trt(f'{model_dir}/campplus.{gpu}.fp32.trt',
                              f'{model_dir}/campplus.onnx',
                              1,
                              False)
        else:
            campplus_model = f'{model_dir}/campplus.onnx'
            option = onnxruntime.SessionOptions()
            option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            option.intra_op_num_threads = 1
            self.spk_model = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])

    def load_spk_trt(self, spk_model, spk_onnx_model, trt_concurrent=1, fp16=True):
        if not os.path.exists(spk_model) or os.path.getsize(spk_model) == 0:
            trt_kwargs = self.get_spk_trt_kwargs()
            convert_onnx_to_trt(spk_model, trt_kwargs, spk_onnx_model, fp16)
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

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        spk_feat = feat - feat.mean(dim=0, keepdim=True)

        if isinstance(self.spk_model, onnxruntime.InferenceSession):
            embedding = self.spk_model.run(
                None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
            embedding = torch.tensor([embedding]).to(self.device)
        else:
            [spk_model, stream], trt_engine = self.spk_model.acquire_estimator()
            # NOTE need to synchronize when switching stream
            with torch.cuda.device(self.device):
                torch.cuda.current_stream().synchronize()
                spk_feat = spk_feat.unsqueeze(dim=0).to(self.device)
                batch_size = spk_feat.size(0)

                with stream:
                    spk_model.set_input_shape('input', (batch_size, spk_feat.size(1), 80))
                    embedding = torch.empty((batch_size, 192), device=spk_feat.device)

                    data_ptrs = [spk_feat.contiguous().data_ptr(),
                                 embedding.contiguous().data_ptr()]
                    for i, j in enumerate(data_ptrs):

                        spk_model.set_tensor_address(trt_engine.get_tensor_name(i), j)
                    # run trt engine
                    assert spk_model.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                    torch.cuda.current_stream().synchronize()
                self.spk_model.release_estimator(spk_model, stream)

        return embedding.half()

    def execute(self, requests):
        """Execute inference on the batched requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses containing tokenized outputs
        """
        responses = []
        # Process each request in batch
        for request in requests:
            # Extract input tensors
            wav_array = pb_utils.get_input_tensor_by_name(
                request, "reference_wav").as_numpy()
            wav_array = torch.from_numpy(wav_array).to(self.device)

            embedding = self._extract_spk_embedding(wav_array)

            prompt_spk_embedding_tensor = pb_utils.Tensor.from_dlpack(
                "prompt_spk_embedding", to_dlpack(embedding))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[prompt_spk_embedding_tensor])

            responses.append(inference_response)

        return responses
