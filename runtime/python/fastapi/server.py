import io
import os
import queue
import subprocess
import sys
import argparse
import threading
import time
import uuid
import wave
import uvicorn
import numpy as np
from loguru import logger
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import fade_in_out_audio
from tools.vad import get_speech


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    audio = wav_chunk_header()
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16)
        tts_audio = tts_audio.tobytes()
        audio += tts_audio
    yield audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))

class GenerateSpeechRequest(BaseModel):
    input: str
    model: str = "tts-1-hd"
    voice: str = "1"
    response_format: str = "mp3"
    speed: float = 0.9

VOICE_MAP = {
    '1': '/home/andrew/CosyVoice/samples/doremon.mp3',
    '2': '/home/andrew/CosyVoice/samples/jack-sparrow.mp3',
    '3': '/home/andrew/CosyVoice/samples/songtung-mtp.wav',
    '4': '/home/andrew/CosyVoice/samples/speechify_8.wav',
    '5': '/home/andrew/CosyVoice/samples/quynh.wav',
    '6': '/home/andrew/CosyVoice/samples/speechify_11.wav',
    '7': '/home/andrew/CosyVoice/samples/diep-chi.wav',
}

SPEED_MAP = {
    '1': 1.0,
    '2': 1.0,
    '3': 1.0,
    '4': 1.0,
    '5': 1.0,
    '6': 1.0,
    '7': 0.9,
}

SPEECH_PROMPT_DURATION_MAP = {
    '1': 3,
    '2': 5,
    '3': 5,
    '4': 5,
    '5': 10,
    '6': 5,
    '7': 5,
}

def wav_chunk_header(sample_rate=22050, bit_depth=16, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatCompletionResponse(BaseModel):
    model: str = "fake"
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[Any] = []
    usage: UsageInfo = UsageInfo()

class ChatCompletionsRequest(BaseModel):
    messages: List[dict]
    model: str
    temperature: float
    max_tokens: int
    stream: bool

@app.post("chat/completions")
@app.post("/v1/chat/completions")
async def completions(completions_request: ChatCompletionsRequest):
    return ChatCompletionResponse()

@app.post("/audio/speech", response_class=StreamingResponse)
@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def tts(tts_request: GenerateSpeechRequest):
    logger.info(f"Generate speech request: {tts_request.dict()}")
    prompt_wav = VOICE_MAP[tts_request.voice]
    prompt_speech_16k = load_wav(prompt_wav, 16000)
    if prompt_speech_16k.abs().max() > 0.9:
        prompt_speech_16k = prompt_speech_16k / prompt_speech_16k.abs().max() * 0.9
    prompt_speech_16k = get_speech(
        audio_input=prompt_speech_16k.squeeze(0),
        min_duration=3,
        max_duration=SPEECH_PROMPT_DURATION_MAP[tts_request.voice]
    ).unsqueeze(0)
    # prompt_speech_16k = fade_in_out_audio(prompt_speech_16k)

    def build_ffmpeg_args(response_format, input_format, sample_rate=24000):
        if input_format == 'WAV':
            ffmpeg_args = ["ffmpeg", "-loglevel", "error", "-f", "WAV", "-i", "-"]
        else:
            ffmpeg_args = ["ffmpeg", "-loglevel", "error", "-f", input_format, "-ar", sample_rate, "-ac", "1", "-i", "-"]
        if response_format == "mp3":
            ffmpeg_args.extend(["-f", "mp3", "-c:a", "libmp3lame", "-ab", "64k"])
        elif response_format == "opus":
            ffmpeg_args.extend(["-f", "ogg", "-c:a", "libopus"])
        elif response_format == "aac":
            ffmpeg_args.extend(["-f", "adts", "-c:a", "aac", "-ab", "64k"])
        elif response_format == "flac":
            ffmpeg_args.extend(["-f", "flac", "-c:a", "flac"])
        elif response_format == "wav":
            ffmpeg_args.extend(["-f", "wav", "-c:a", "pcm_s16le"])
        elif response_format == "pcm": # even though pcm is technically 'raw', we still use ffmpeg to adjust the speed
            ffmpeg_args.extend(["-f", "s16le", "-c:a", "pcm_s16le"])
        return ffmpeg_args

    def exception_check(exq: queue.Queue):
        try:
            e = exq.get_nowait()
        except queue.Empty:
            return
        raise e

    if tts_request.response_format == "mp3":
        media_type = "audio/mpeg"
    elif tts_request.response_format == "opus":
        media_type = "audio/ogg;codec=opus" # codecs?
    elif tts_request.response_format == "aac":
        media_type = "audio/aac"
    elif tts_request.response_format == "flac":
        media_type = "audio/x-flac"
    elif tts_request.response_format == "wav":
        media_type = "audio/wav"
    elif tts_request.response_format == "pcm":
        media_type = "audio/pcm;rate=24000"
    else:
        raise ValueError(f"Invalid response_format: '{tts_request.response_format}'", param='response_format')

    ffmpeg_args = None
    ffmpeg_args = build_ffmpeg_args(tts_request.response_format, input_format="f32le", sample_rate="24000")
    ffmpeg_args.extend(["-"])
    ffmpeg_proc = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    in_q = queue.Queue()
    ex_q = queue.Queue()

    def generator():
        # text -> in_q
        try:
            model_output = cosyvoice.inference_cross_lingual(
                tts_text=tts_request.input,
                prompt_speech_16k=prompt_speech_16k,
                speed=SPEED_MAP[tts_request.voice],
                stream=False
            )
            for chunk in model_output:
                exception_check(ex_q)
                chunk = chunk['tts_speech'].numpy().tobytes()
                in_q.put(chunk)
        except BrokenPipeError as e:
            logger.info("Client disconnected - 'Broken pipe'")
        except Exception as e:
            logger.error(f"Exception: {repr(e)}")
            raise e
        finally:
            in_q.put(None) # sentinel

    def out_writer(): 
        # in_q -> ffmpeg
        try:
            while True:
                chunk = in_q.get()
                if chunk is None: # sentinel
                    break
                ffmpeg_proc.stdin.write(chunk) # BrokenPipeError from here on client disconnect
        except Exception as e: # BrokenPipeError
            ex_q.put(e)  # we need to get this exception into the generation loop
            ffmpeg_proc.kill()
            return
        finally:
            ffmpeg_proc.stdin.close()
            
    generator_worker = threading.Thread(target=generator, daemon=True)
    generator_worker.start()
    out_writer_worker = threading.Thread(target=out_writer, daemon=True)
    out_writer_worker.start()

    async def cleanup():
        try:
            ffmpeg_proc.kill()
            # del generator_worker
            # del out_writer_worker
        except Exception as e:
            logger.error(f"Exception: {repr(e)}")

    return StreamingResponse(
        content=ffmpeg_proc.stdout,
        media_type=media_type,
        background=cleanup
    )

@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)