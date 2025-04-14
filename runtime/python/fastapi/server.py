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
import argparse
import re
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

asset_dir = os.path.join('{}/../../../voice'.format(ROOT_DIR))
app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


def generate_txt(tts_text):
    for str in re.split(r'[.,!?\n]', tts_text):
        yield str


def load_txt(prompt_text):
    with open(prompt_text, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    return prompt_text


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), person: str = Form()):
    prompt_speech_16k = load_wav(f'{asset_dir}/{person}/prompt.wav', 16000)
    prompt_text = load_txt(f'{asset_dir}/{person}/prompt.txt')
    # generate_txt (yield) is optional
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=True)
    # model_output = cosyvoice.inference_zero_shot(generate_txt(tts_text), prompt_text, prompt_speech_16k, stream=True)
    return StreamingResponse(generate_data(model_output))

@app.get("/inference_zero_shot_use_cache")
@app.post("/inference_zero_shot_use_cache")
async def inference_zero_shot_use_cache(tts_text: str = Form(), person: str = Form()):
    # generate_txt (yield) is optional
    model_output = cosyvoice.inference_zero_shot_use_cache(tts_text, 'prompt_wav_cache/{}.pt'.format(person), stream=True)
    # model_output = cosyvoice.inference_zero_shot(generate_txt(tts_text), prompt_text, prompt_speech_16k, stream=True)
    return StreamingResponse(generate_data(model_output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=21559)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B-trt',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice2(args.model_dir)
    except Exception:
        raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)