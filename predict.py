# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import subprocess
import time
from cog import BasePredictor, Input, Path
import torchaudio

sys.path.insert(0, os.path.abspath("third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


MODEL_CACHE = "pretrained_models"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/FunAudioLLM/CosyVoice/model_cache.tar"
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        self.cosyvoice = CosyVoice2(
            "pretrained_models/CosyVoice2-0.5B",
            load_jit=True,
            load_onnx=False,
            load_trt=False,
        )

    def predict(
        self,
        source_audio: Path = Input(description="Source audio"),
        source_transcript: str = Input(
            description="Transcript of the source audio, you can use models such as whisper to transcribe first"
        ),
        tts_text: str = Input(description="Text of the audio to generate"),
        task: str = Input(
            choices=[
                "zero-shot voice clone",
                "cross-lingual voice clone",
                "Instructed Voice Generation",
            ],
            default="zero-shot voice clone",
        ),
        instruction: str = Input(
            description="Instruction for Instructed Voice Generation task", default=""
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if task == "Instructed Voice Generation":
            assert len(instruction) > 0, "Please specify the instruction."

        prompt_speech_16k = load_wav(str(source_audio), 16000)

        if task == "zero-shot voice clone":
            output = self.cosyvoice.inference_zero_shot(
                tts_text, source_transcript, prompt_speech_16k, stream=False
            )
        elif task == "cross-lingual voice clone":
            output = self.cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=False
            )
        else:
            output = self.cosyvoice.inference_instruct2(
                tts_text, instruction, prompt_speech_16k, stream=False
            )

        out_path = "/tmp/out.wav"
        torchaudio.save(
            out_path, list(output)[0]["tts_speech"], self.cosyvoice.sample_rate
        )
        return Path(out_path)
