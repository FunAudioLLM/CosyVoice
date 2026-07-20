#!/usr/bin/env python3
"""CosyVoice Streaming TTS WebSocket Server.

Inspired by the FunASR real-time WS server pattern.
Provides streaming text-to-speech via WebSocket — clients send text,
receive int16 PCM audio chunks (24000 Hz).

Protocol (all client→server messages are JSON):
  → {"cmd": "synthesize", "text": "...", "prompt_text": "...", "prompt_wav": "...", "speed": 1.0}
  ← {"event": "ready", "sample_rate": 24000, "prompt_text": "..."}
  ← {"event": "start", "text": "...", "sample_rate": 24000}
  ← binary PCM chunks (int16, 24000 Hz, mono)
  ← {"event": "end"}
  → {"cmd": "stop"}
  ← {"event": "stopped"}
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import time

import numpy as np
import torch
import websockets

sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  global model & config (lazy-loaded at first client connect)
# ---------------------------------------------------------------------------
_cosyvoice = None
_server_config = {}
_synth_lock = threading.Lock()  # only one synthesis at a time


def _load_model(args):
    global _cosyvoice, _server_config
    if _cosyvoice is not None:
        return
    logger.info('Loading CosyVoice model: %s', args.model_dir)
    from cosyvoice.cli.cosyvoice import AutoModel
    _cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )
    _server_config = {
        'prompt_text': args.prompt_text,
        'prompt_wav': args.prompt_wav,
        'sample_rate': _cosyvoice.sample_rate,
    }
    logger.info('Model ready – sample_rate=%d', _server_config['sample_rate'])


# ---------------------------------------------------------------------------
#  per-synthesis runner (runs in a background thread, feeds an async queue)
# ---------------------------------------------------------------------------

def _run_synthesis(cosyvoice, text, prompt_text, prompt_wav, speed, aqueue, loop, cancel_event):
    """Execute TTS in a sync thread; push (type, payload) tuples into aqueue."""
    try:
        for model_output in cosyvoice.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=True, speed=speed,
        ):
            if cancel_event.is_set():
                break
            speech = model_output['tts_speech'].detach().cpu().numpy().squeeze()
            speech = np.clip(speech, -1.0, 1.0)
            pcm = (speech * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(aqueue.put_nowait, ('chunk', pcm))
        loop.call_soon_threadsafe(aqueue.put_nowait, ('done', None))
    except Exception as exc:
        loop.call_soon_threadsafe(aqueue.put_nowait, ('error', str(exc)))


# ---------------------------------------------------------------------------
#  WebSocket handler
# ---------------------------------------------------------------------------

async def handle_client(websocket, args):
    cosyvoice = _cosyvoice

    logger.info('Client connected: %s', websocket.remote_address)

    # Send server info on connect
    await websocket.send(json.dumps({
        'event': 'ready',
        'sample_rate': _server_config['sample_rate'],
        'prompt_text': _server_config['prompt_text'],
    }))

    cancel_event = threading.Event()
    synth_active = False

    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue

            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({'event': 'error', 'message': 'Invalid JSON'}))
                continue

            cmd = msg.get('cmd', '').lower()

            # ---- synthesize ----
            if cmd == 'synthesize':
                text = msg.get('text', '').strip()
                if not text:
                    await websocket.send(json.dumps({'event': 'error', 'message': 'Empty text'}))
                    continue

                if synth_active:
                    await websocket.send(json.dumps({'event': 'error',
                                                     'message': 'Synthesis already in progress'}))
                    continue

                acquired = _synth_lock.acquire(blocking=False)
                if not acquired:
                    await websocket.send(json.dumps({'event': 'error',
                                                     'message': 'Server busy'}))
                    continue

                synth_active = True
                cancel_event.clear()

                prompt_text = msg.get('prompt_text', _server_config['prompt_text'])
                prompt_wav = msg.get('prompt_wav', _server_config['prompt_wav'])
                speed = float(msg.get('speed', 1.0))

                # Handle client-uploaded prompt audio (base64-encoded)
                tmp_prompt_wav = None
                if 'prompt_wav_data' in msg and msg['prompt_wav_data']:
                    try:
                        wav_bytes = base64.b64decode(msg['prompt_wav_data'])
                        suffix = os.path.splitext(msg.get('prompt_wav_filename', ''))[1] or '.wav'
                        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        tmp.write(wav_bytes)
                        tmp.close()
                        tmp_prompt_wav = tmp.name
                        prompt_wav = tmp_prompt_wav
                        logger.info('Using client-uploaded prompt audio: %s', tmp_prompt_wav)
                    except Exception as exc:
                        logger.error('Failed to decode prompt_wav_data: %s', exc)
                        await websocket.send(json.dumps({'event': 'error',
                                                         'message': f'Bad prompt_wav_data: {exc}'}))
                        continue

                await websocket.send(json.dumps({
                    'event': 'start',
                    'text': text,
                    'sample_rate': _server_config['sample_rate'],
                }))

                logger.info('Synthesizing (%d chars): %s…', len(text), text[:60])

                aqueue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                thr = threading.Thread(
                    target=_run_synthesis,
                    args=(cosyvoice, text, prompt_text, prompt_wav, speed,
                          aqueue, loop, cancel_event),
                    daemon=True,
                )
                thr.start()

                try:
                    while True:
                        try:
                            typ, data = await asyncio.wait_for(aqueue.get(), timeout=120)
                        except asyncio.TimeoutError:
                            await websocket.send(json.dumps({'event': 'error',
                                                             'message': 'Synthesis timeout'}))
                            cancel_event.set()
                            break

                        if typ == 'chunk':
                            await websocket.send(data)
                        elif typ == 'done':
                            await websocket.send(json.dumps({'event': 'end'}))
                            break
                        elif typ == 'error':
                            await websocket.send(json.dumps({'event': 'error', 'message': data}))
                            break
                finally:
                    cancel_event.set()
                    thr.join(timeout=5)
                    synth_active = False
                    _synth_lock.release()
                    # Clean up temp prompt wav file
                    if tmp_prompt_wav and os.path.exists(tmp_prompt_wav):
                        try:
                            os.unlink(tmp_prompt_wav)
                        except OSError:
                            pass
                    logger.info('Synthesis finished')

            # ---- stop ----
            elif cmd == 'stop':
                if synth_active:
                    cancel_event.set()
                await websocket.send(json.dumps({'event': 'stopped'}))

            # ---- info ----
            elif cmd == 'info':
                await websocket.send(json.dumps({
                    'event': 'info',
                    'sample_rate': _server_config['sample_rate'],
                    'prompt_text': _server_config['prompt_text'],
                    'prompt_wav': _server_config['prompt_wav'],
                }))

            else:
                await websocket.send(json.dumps({'event': 'error',
                                                 'message': f'Unknown command: {cmd}'}))

    except websockets.exceptions.ConnectionClosed:
        logger.info('Client disconnected')
    except Exception as exc:
        logger.error('Handler error: %s', exc, exc_info=True)
    finally:
        if synth_active:
            cancel_event.set()
            _synth_lock.release() if _synth_lock.locked() else None


async def _main(args):
    _load_model(args)
    logger.info('Server on ws://0.0.0.0:%d', args.port)
    async with websockets.serve(
        lambda ws, _loop=None: handle_client(ws, args),
        '0.0.0.0', args.port,
        max_size=10 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=10,
    ):
        await asyncio.Future()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description='CosyVoice Streaming TTS WebSocket Server')
    p.add_argument('--port', type=int, default=10096,
                   help='WebSocket server port (default: 10096)')
    p.add_argument('--model-dir', type=str,
                   default='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                   help='ModelScope model name or local path')
    p.add_argument('--prompt-text', type=str,
                   default='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                   help='Default prompt (reference) text for CosyVoice3 zero-shot (must include <|endofprompt|> separator)')
    p.add_argument('--prompt-wav', type=str,
                   default='asset/zero_shot_prompt.wav',
                   help='Default prompt (reference) audio file for zero-shot voice cloning')
    p.add_argument('--load-trt', action=argparse.BooleanOptionalAction, default=True,
                   help='Load TensorRT engine for flow decoder (default: True)')
    p.add_argument('--load-vllm', action=argparse.BooleanOptionalAction, default=True,
                   help='Load vLLM engine for LLM (default: True)')
    p.add_argument('--fp16', action='store_true', default=False,
                   help='Use fp16 precision')
    return p


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    asyncio.run(_main(args))
