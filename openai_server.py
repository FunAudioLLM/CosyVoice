import os
import sys
import argparse
import logging
import asyncio
import threading
import subprocess
import numpy as np
import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, Generator

# Adjust path to include CosyVoice modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError as e:
    print(f"Error importing CosyVoice: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_server")

app = FastAPI(title="CosyVoice OpenAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
cosyvoice = None
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
DEFAULT_PROMPT_WAV = "asset/zero_shot_prompt.wav"

DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

# CUSTOM VOICE MAP
VOICE_MAP = {
    "russian": {
        "text": "Всем привет, дорогие друзья! Сейчас 6.20 и мы с вами успели. Сегодня мы с вами встречаем восход солнца.",
        "wav": "asset/russian_prompt.wav"
    },
    "english": {
        "text": "And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that's coming into the family is a reason why sometimes we don't buy the whole thing.",
        "wav": "asset/cross_lingual_prompt.wav"
    }
}


class SpeechRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: Optional[str] = "中文女"
    response_format: Optional[Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']] = "mp3"
    speed: Optional[float] = 1.0

def get_ffmpeg_cmd(format: str, sample_rate: int):
    cmd = ['ffmpeg', '-f', 's16le', '-ar', str(sample_rate), '-ac', '1', '-i', 'pipe:0']
    if format == 'mp3':
        cmd.extend(['-f', 'mp3', 'pipe:1'])
    elif format == 'opus':
        cmd.extend(['-f', 'opus', '-c:a', 'libopus', 'pipe:1'])
    elif format == 'aac':
        cmd.extend(['-f', 'adts', 'pipe:1'])
    elif format == 'flac':
        cmd.extend(['-f', 'flac', 'pipe:1'])
    elif format == 'wav':
        cmd.extend(['-f', 'wav', 'pipe:1'])
    else:
        raise ValueError(f"Unsupported format via ffmpeg: {format}")
    return cmd

@app.on_event("startup")
async def startup_event():
    global cosyvoice
    logger.info(f"Loading model from {MODEL_DIR}...")
    try:
        if not os.path.exists(MODEL_DIR):
            logger.warning(f"Model directory {MODEL_DIR} not found. Please run download_models.py first.")
        else:
            cosyvoice = AutoModel(model_dir=MODEL_DIR)
            logger.info(f"Model loaded. Sample rate: {cosyvoice.sample_rate}")
            available_spks = cosyvoice.list_available_spks()
            logger.info(f"Available speakers: {available_spks}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": "cosyvoice-tts",
                "object": "model",
                "created": 1234567890,
                "owned_by": "cosyvoice",
            }
        ]
    })

@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if not cosyvoice:
        raise HTTPException(status_code=500, detail="Model not loaded or invalid model directory.")
    
    text = req.input
    spk_id = req.voice
    speed = req.speed if req.speed else 1.0

    # Custom Voice Logic
    prompt_text = DEFAULT_PROMPT_TEXT
    prompt_wav = DEFAULT_PROMPT_WAV
    
    # Check if requested voice matches our map (case-insensitive)
    for key, val in VOICE_MAP.items():
        if key.lower() in spk_id.lower():
            prompt_text = val["text"]
            prompt_wav = val["wav"]
            logger.info(f"Using custom prompt for voice '{spk_id}': {val['wav']}")
            break
    format = req.response_format
    
    available_spks = cosyvoice.list_available_spks()
    use_zero_shot = spk_id not in available_spks

    logger.info(f"TTS Request: text='{text[:20]}...', voice='{spk_id}', format={format}, speed={speed}, zero_shot={use_zero_shot}")

    async def audio_generator():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        sentinel = object()
        ffmpeg_proc = None
        
        if format != 'pcm':
            try:
                cmd = get_ffmpeg_cmd(format, cosyvoice.sample_rate)
                ffmpeg_proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
            except Exception as e:
                logger.error(f"ffmpeg error: {e}")
                return

        def producer_thread():
            try:
                if use_zero_shot:
                    generator = cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=True, speed=speed)
                else:
                    generator = cosyvoice.inference_sft(text, spk_id, stream=True, speed=speed)

                for i in generator:
                    raw_data = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                    if ffmpeg_proc:
                        try:
                            ffmpeg_proc.stdin.write(raw_data)
                            ffmpeg_proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            break
                    else:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_data)
                
                if ffmpeg_proc:
                    ffmpeg_proc.stdin.close()
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)
            except Exception as e:
                logger.error(f"Producer error: {e}")
                if ffmpeg_proc:
                    ffmpeg_proc.terminate()
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        def reader_thread():
            if not ffmpeg_proc: return
            try:
                while True:
                    chunk = ffmpeg_proc.stdout.read(4096)
                    if not chunk: break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                ffmpeg_proc.wait()
            except Exception as e:
                logger.error(f"Reader error: {e}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=producer_thread, daemon=True).start()
        if ffmpeg_proc:
            threading.Thread(target=reader_thread, daemon=True).start()
            
        while True:
            chunk = await queue.get()
            if chunk is sentinel: break
            yield chunk

    media_type = f"audio/{format}"
    if format == 'pcm': media_type = "application/octet-stream"
    elif format == 'mp3': media_type = "audio/mpeg"
    elif format == 'wav': media_type = "audio/wav"
    
    return StreamingResponse(audio_generator(), media_type=media_type)

@app.websocket("/v1/audio/speech/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # 1. Receive config
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        spk_id = config.get("voice", "中文女")
        speed = config.get("speed", 1.0)
        language = config.get("language")
        # format = config.get("response_format", "pcm") # For simplicity, WS yields PCM for now
        
        available_spks = cosyvoice.list_available_spks()
        use_zero_shot = spk_id not in available_spks
        
        # 2. Setup text generator for bi-streaming
        text_queue = asyncio.Queue()
        
        def sync_text_generator():
            # This generator will be consumed by CosyVoice in a background thread
            first_chunk = True
            while True:
                # We need to get items from the async queue in a sync way
                future = asyncio.run_coroutine_threadsafe(text_queue.get(), loop)
                val = future.result()
                if val is None:
                    break
                
                # Prepend language tag if provided in config
                if first_chunk and language:
                    val = f"<|{language}|>" + val
                    first_chunk = False
                elif first_chunk:
                    first_chunk = False
                    
                yield val

        loop = asyncio.get_running_loop()
        
        # 3. Start inference in a separate thread
        inference_finished_event = asyncio.Event()

        def run_inference():
            try:
                # Use the sync generator which bridges to the async queue
                if use_zero_shot:
                    # Note: inference_zero_shot with generator is only supported if the model is CosyVoice2/3
                    # and it calls inference_bistream internally.
                    gen = cosyvoice.inference_zero_shot(sync_text_generator(), DEFAULT_PROMPT_TEXT, DEFAULT_PROMPT_WAV, stream=True, speed=speed)
                else:
                    gen = cosyvoice.inference_sft(sync_text_generator(), spk_id, stream=True, speed=speed)
                
                for output in gen:
                    audio_data = (output['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                    asyncio.run_coroutine_threadsafe(websocket.send_bytes(audio_data), loop)
            except Exception as e:
                logger.error(f"WS Inference error: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                loop.call_soon_threadsafe(inference_finished_event.set)

        inference_thread = threading.Thread(target=run_inference, daemon=True)
        inference_thread.start()

        # 4. Receive text tokens
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            text_chunk = msg.get("text")
            if text_chunk:
                await text_queue.put(text_chunk)
            if msg.get("event") == "flush":
                # CosyVoice doesn't have an explicit flush per chunk, 
                # but we can handle it if we want to restart the generator
                pass
            if msg.get("event") == "end":
                await text_queue.put(None)
                break
        
        # Wait for inference to finish sending all audio
        await inference_finished_event.wait()

        # Explicitly close WebSocket after all audio sent
        await websocket.close()
        logger.info("WebSocket closed after audio complete")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Ensure thread stops
        await text_queue.put(None)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    args = parser.parse_args()
    
    MODEL_DIR = args.model_dir
    uvicorn.run(app, host=args.host, port=args.port)