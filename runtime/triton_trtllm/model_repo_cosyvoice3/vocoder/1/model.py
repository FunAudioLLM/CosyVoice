import json
import os
import logging

import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
from hyperpyyaml import load_hyperpyyaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_num_threads(1)


class TritonPythonModel:
    """Triton Python model for CosyVoice3 vocoder (CausalHiFTGenerator).

    Stateless: converts mel spectrogram to waveform.
    CausalHiFTGenerator manages its own internal cache.
    """

    def initialize(self, args):
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        model_dir = model_params["model_dir"]

        self.device = torch.device("cuda")

        # Load CausalHiFTGenerator from cosyvoice3.yaml
        with open(os.path.join(model_dir, 'cosyvoice3.yaml'), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={
                'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
            })
        self.hift = configs['hift']
        hift_state_dict = {
            k.replace('generator.', ''): v
            for k, v in torch.load(
                os.path.join(model_dir, 'hift.pt'),
                map_location='cpu', weights_only=True
            ).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
        logger.info("CausalHiFTGenerator initialized successfully")

    def execute(self, requests):
        responses = []
        for req_idx, request in enumerate(requests):
            mel = pb_utils.get_input_tensor_by_name(request, "mel")
            mel = torch.utils.dlpack.from_dlpack(mel.to_dlpack()).to(self.device)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # [80, T] -> [1, 80, T]

            finalize = pb_utils.get_input_tensor_by_name(request, "finalize").as_numpy().item()

            with torch.no_grad():
                speech, _ = self.hift.inference(speech_feat=mel, finalize=finalize)

            # speech shape: [1, 1, S] or [1, S] depending on hift version
            speech = speech.squeeze()  # flatten to [S]

            speech_tensor = pb_utils.Tensor.from_dlpack(
                "tts_speech", to_dlpack(speech.unsqueeze(0)))  # [1, S] for batch dim
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[speech_tensor])
            responses.append(inference_response)

        return responses
