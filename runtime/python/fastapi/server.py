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
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import librosa
import random
import torch

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

max_val = 0.8

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech
    
def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def generate_stream(model_output):
    for i in model_output:
        tts_audio = i['tts_speech'].numpy().tobytes()
        yield tts_audio
        
def generate_header():
    headers = {
        "X-Custom-Header-sampleRate": f"{cosyvoice.sample_rate}"
    }
    return headers
    
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_sft(tts_text, spk_id ,stream = False)
    return StreamingResponse(generate_data(model_output),headers=generate_header())

@app.post("/stream/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_sft(tts_text, spk_id, stream = True)
    return StreamingResponse(generate_stream(model_output),headers=generate_header())

@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = postprocess(load_wav(prompt_wav.file, 16000))
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k,stream = False)
   
    return StreamingResponse(generate_data(model_output),headers=generate_header())

@app.post("/stream/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = postprocess(load_wav(prompt_wav.file, 16000))
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream = True)
    return StreamingResponse(generate_stream(model_output),headers=generate_header())

@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = postprocess(load_wav(prompt_wav.file, 16000))
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream = False)
    return StreamingResponse(generate_data(model_output),headers=generate_header())
    
@app.post("/stream/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = postprocess(load_wav(prompt_wav.file, 16000))
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream = True)
    return StreamingResponse(generate_stream(model_output),headers=generate_header())

@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text, stream = False)
    return StreamingResponse(generate_data(model_output),headers=generate_header())

@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    set_all_random_seed(generate_seed()["value"])
    prompt_speech_16k = postprocess(load_wav(prompt_wav.file, 16000))
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text,prompt_speech_16k, stream = False)
    return StreamingResponse(generate_data(model_output),headers=generate_header())

@app.post("/stream/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    set_all_random_seed(generate_seed()["value"])
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text, stream = True)
    return StreamingResponse(generate_stream(model_output),headers=generate_header())
    
import torch
import threading

# 检查当前 GPU 的显存使用情况
def check_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # 转换为MB
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # 转换为MB
    logging.info(f"Allocated memory: {allocated:.2f} MB")
    logging.info(f"Reserved memory: {reserved:.2f} MB")
    logging.info(f"Total memory: {total_memory:.2f} MB")
    return allocated, reserved, total_memory

# 释放未使用的显存
def release_memory():
    torch.cuda.empty_cache()
    logging.info("Memory has been released.")

# 检查显存使用情况并在需要时释放显存
def monitor_and_release_memory():
    allocated, reserved, total_memory = check_memory_usage()
    if allocated >= total_memory / 2:
        logging.info("Allocated memory exceeds half of the total memory. Releasing memory...")
        release_memory()
    else:
        logging.info("Memory usage is within acceptable limits.")

# 定时器函数，每10分钟运行一次
def run_periodically(interval, func):
    def wrapper():
        func()
        threading.Timer(interval, wrapper).start()
    
    wrapper()

@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    # 设置每个进程最多使用 50% 的 GPU 显存
    #torch.cuda.set_per_process_memory_fraction(0.8, 0)
    #logging.info('Torch set_per_process_memory_fraction 0.8')
    cosyvoice = CosyVoice2(args.model_dir) if 'CosyVoice2' in args.model_dir else CosyVoice(args.model_dir)
    # 每10分钟（600秒）运行一次 monitor_and_release_memory
    #run_periodically(600, monitor_and_release_memory)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
