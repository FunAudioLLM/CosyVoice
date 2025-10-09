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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)


class CosyVoice2:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1, device='cuda'):

        self.model_dir = model_dir
        self.fp16 = fp16

        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        self.model = CosyVoice2Model(configs['flow'], configs['hift'], fp16, device)
        self.model.load('{}/flow.pt'.format(model_dir), '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)


class CosyVoice2Model:

    def __init__(self,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False,
                 device: str = 'cuda'):
        self.device = device
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()

        # streaming tts config
        self.token_hop_len = 25
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.hift_cache_dict = defaultdict(lambda: None)

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech


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

        self.token2wav_model = CosyVoice2(
            model_dir, load_jit=False, load_trt=True, fp16=True, device=self.device
        )

        spk_info_path = os.path.join(model_dir, "spk2info.pt")
        if not os.path.exists(spk_info_path):
            raise ValueError(f"spk2info.pt not found in {model_dir}")
        spk_info = torch.load(spk_info_path, map_location="cpu", weights_only=False)
        self.default_spk_info = spk_info["001"]

        logger.info("Token2Wav initialized successfully")

    def execute(self, requests):
        """Execute inference on the batched requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses containing generated waveforms
        """
        responses = []
        # Process each request in batch
        for request in requests:
            target_speech_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "target_speech_tokens").as_numpy()
            target_speech_tokens = torch.from_numpy(target_speech_tokens_tensor).to(self.device)

            prompt_speech_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "prompt_speech_tokens")
            if prompt_speech_tokens_tensor is not None:
                prompt_speech_tokens_tensor = prompt_speech_tokens_tensor.as_numpy()
                prompt_speech_feat_tensor = pb_utils.get_input_tensor_by_name(request, "prompt_speech_feat").as_numpy()
                prompt_spk_embedding_tensor = pb_utils.get_input_tensor_by_name(request, "prompt_spk_embedding").as_numpy()
                prompt_speech_tokens = torch.from_numpy(prompt_speech_tokens_tensor).to(self.device)
                prompt_speech_feat = torch.from_numpy(prompt_speech_feat_tensor).to(self.device)
                prompt_spk_embedding = torch.from_numpy(prompt_spk_embedding_tensor).to(self.device)
                prompt_speech_tokens = prompt_speech_tokens - ORIGINAL_VOCAB_SIZE
            else:
                prompt_speech_tokens = self.default_spk_info["speech_token"].to(self.device)
                prompt_speech_feat = self.default_spk_info["speech_feat"].to(torch.float16).to(self.device)
                prompt_spk_embedding = self.default_spk_info["embedding"].to(torch.float16).to(self.device)

            # shift the speech tokens according to the original vocab size
            target_speech_tokens = target_speech_tokens - ORIGINAL_VOCAB_SIZE

            # We set token_offset as an optional input to support streaming/offline tts. It has to be None when offline tts.
            token_offset = pb_utils.get_input_tensor_by_name(request, "token_offset")
            if token_offset is not None:
                token_offset = token_offset.as_numpy().item()
                finalize = pb_utils.get_input_tensor_by_name(request, "finalize").as_numpy().item()
                if not finalize:
                    stream = True
                else:
                    stream = False
                request_id = request.request_id()
                audio_hat = self.token2wav_model.model.token2wav(token=target_speech_tokens,
                                                                 prompt_token=prompt_speech_tokens,
                                                                 prompt_feat=prompt_speech_feat,
                                                                 embedding=prompt_spk_embedding,
                                                                 token_offset=token_offset,
                                                                 uuid=request_id,
                                                                 stream=stream,
                                                                 finalize=finalize)
                if finalize:
                    self.token2wav_model.model.hift_cache_dict.pop(request_id)

            else:
                tts_mel, _ = self.token2wav_model.model.flow.inference(
                    token=target_speech_tokens,
                    token_len=torch.tensor([target_speech_tokens.shape[1]], dtype=torch.int32).to(
                        self.device
                    ),
                    prompt_token=prompt_speech_tokens,
                    prompt_token_len=torch.tensor(
                        [prompt_speech_tokens.shape[1]], dtype=torch.int32
                    ).to(self.device),
                    prompt_feat=prompt_speech_feat,
                    prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(self.device),
                    embedding=prompt_spk_embedding,
                    streaming=False,
                    finalize=True,
                )

                audio_hat, _ = self.token2wav_model.model.hift.inference(
                    speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
                )

            generated_wave = audio_hat.squeeze(0).cpu().numpy()

            wav_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio_hat))
            inference_response = pb_utils.InferenceResponse(output_tensors=[wav_tensor])
            responses.append(inference_response)

        return responses
