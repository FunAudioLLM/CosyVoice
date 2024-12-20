# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import io
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
import torchaudio
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = f'{CURR_DIR}/../../..'
sys.path.append(f'{ROOT_DIR}')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

model_dir = f"{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B"
cosyvoice = CosyVoice2(model_dir) if 'CosyVoice2' in model_dir else CosyVoice(model_dir)

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


# 非流式wav数据
def build_data(model_output):
    tts_speeches = []
    for i in model_output:
        tts_speeches.append(i['tts_speech'])
    output = torch.concat(tts_speeches, dim=1)

    buffer = io.BytesIO()
    torchaudio.save(buffer, output, 22050, format="wav")
    buffer.seek(0)
    return buffer.read(-1)


# 流式pcm数据
def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form(), format: str = Form(default="pcm")):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    if format == "pcm":
        return StreamingResponse(generate_data(model_output))
    else:
        return Response(build_data(model_output), media_type="audio/wav")


@app.get("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File(), format: str = Form(default="pcm")):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    if format == "pcm":
        return StreamingResponse(generate_data(model_output))
    else:
        return Response(build_data(model_output), media_type="audio/wav")


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File(), format: str = Form(default="pcm")):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    if format == "pcm":
        return StreamingResponse(generate_data(model_output))
    else:
        return Response(build_data(model_output), media_type="audio/wav")


@app.get("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form(), format: str = Form(default="pcm")):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    if format == "pcm":
        return StreamingResponse(generate_data(model_output))
    else:
        return Response(build_data(model_output), media_type="audio/wav")


@app.get("/inference_instruct_v2")
async def inference_instruct_v2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File(), format: str = Form(default="pcm")):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    if format == "pcm":
        return StreamingResponse(generate_data(model_output))
    else:
        return Response(build_data(model_output), media_type="audio/wav")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
