"""FastAPI wrapper for CosyVoice3 zero-shot TTS, no model lock.

Drops the global model lock so vLLM's continuous batching can fuse concurrent
requests at the LLM step. Flow matching + hift may not be strictly thread-safe,
but the in-process concurrent bench ran without crashes — exposing the same
behavior here lets us measure HTTP-side QPS without lock serialization.

Endpoints:
- GET  /health     → {"ok": true, "model_loaded": bool}
- POST /tts        → {"text": "...", "seed": 0} → wav bytes
- GET  /metrics    → cumulative request count + audio seconds generated
                     plus in_flight gauge
"""
import io, os, sys, time, threading
sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from fe_cache import enable_fe_cache
import torchaudio
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse
from pydantic import BaseModel

PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
PROMPT_WAV = './asset/zero_shot_prompt.wav'
MODEL_DIR = os.environ.get('MODEL_DIR', 'pretrained_models/Fun-CosyVoice3-0.5B')
LOAD_TRT = os.environ.get('LOAD_TRT', '1') == '1'
FP16 = os.environ.get('FP16', '1') == '1'

app = FastAPI(title='CosyVoice3 TTS (lockfree)')
_model = None
_metrics = {'requests': 0, 'audio_seconds': 0.0, 'total_wall_seconds': 0.0,
            'in_flight': 0, 'errors': 0}
_metrics_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str
    seed: int = 0


@app.on_event('startup')
def load_model():
    global _model
    print(f'[startup] loading {MODEL_DIR}, trt={LOAD_TRT}, fp16={FP16} ...', flush=True)
    t0 = time.time()
    _model = AutoModel(model_dir=MODEL_DIR, load_trt=LOAD_TRT, load_vllm=True, fp16=FP16)
    enable_fe_cache(_model)
    print(f'[startup] loaded in {time.time()-t0:.2f}s; FE prompt cache enabled', flush=True)


@app.get('/')
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(here, 'web', 'index.html')
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type='text/html')
    raise HTTPException(404, 'web/index.html not found')


@app.get('/health')
def health():
    return {'ok': _model is not None, 'model_loaded': _model is not None}


@app.get('/metrics')
def metrics():
    with _metrics_lock:
        m = {k: v for k, v in _metrics.items() if k != 'ttfa_samples'}
        ttfa_samples = list(_metrics.get('ttfa_samples', []))
    m['realtime_factor'] = (m['audio_seconds'] / m['total_wall_seconds']) if m['total_wall_seconds'] > 0 else None
    if ttfa_samples:
        ttfa_samples.sort()
        n = len(ttfa_samples)
        m['ttfa_p50_ms'] = round(ttfa_samples[n // 2] * 1000, 1)
        m['ttfa_p95_ms'] = round(ttfa_samples[int(n * 0.95)] * 1000, 1)
        m['ttfa_p99_ms'] = round(ttfa_samples[int(n * 0.99)] * 1000, 1) if n >= 100 else None
        m['ttfa_count']  = n
    return m


@app.post('/tts')
def tts(req: TTSRequest):
    if _model is None:
        raise HTTPException(503, 'model not loaded')
    if not req.text.strip():
        raise HTTPException(400, 'empty text')

    with _metrics_lock:
        _metrics['in_flight'] += 1
    t0 = time.time()
    try:
        # NB: no lock — relies on vllm thread-safety + tolerated CosyVoice races
        set_all_random_seed(req.seed)
        chunks = []
        for j in _model.inference_zero_shot(req.text, PROMPT_TEXT, PROMPT_WAV, stream=False):
            chunks.append(j['tts_speech'])
    except Exception as e:
        with _metrics_lock:
            _metrics['errors'] += 1
            _metrics['in_flight'] -= 1
        raise HTTPException(500, f'inference failed: {type(e).__name__}: {e}')
    wall = time.time() - t0

    if not chunks:
        with _metrics_lock:
            _metrics['errors'] += 1
            _metrics['in_flight'] -= 1
        raise HTTPException(500, 'no audio generated')

    audio = torch.cat(chunks, dim=-1)
    audio_sec = audio.shape[-1] / _model.sample_rate

    buf = io.BytesIO()
    torchaudio.save(buf, audio, _model.sample_rate, format='wav')
    buf.seek(0)

    with _metrics_lock:
        _metrics['requests'] += 1
        _metrics['audio_seconds'] += audio_sec
        _metrics['total_wall_seconds'] += wall
        _metrics['in_flight'] -= 1

    return Response(
        content=buf.read(),
        media_type='audio/wav',
        headers={
            'X-Audio-Seconds': f'{audio_sec:.3f}',
            'X-Wall-Seconds':  f'{wall:.3f}',
            'X-RTF':           f'{wall/audio_sec:.3f}',
        },
    )


@app.post('/tts/stream')
def tts_stream(req: TTSRequest):
    """Streaming TTS: returns chunked raw PCM int16 mono.

    Client should read chunks as they arrive — the time between request send
    and first byte received is TTFA (Time To First Audio).

    Sample rate is in `X-Sample-Rate` header.
    """
    if _model is None:
        raise HTTPException(503, 'model not loaded')
    if not req.text.strip():
        raise HTTPException(400, 'empty text')

    sr = _model.sample_rate
    text = req.text
    seed = req.seed

    with _metrics_lock:
        _metrics['in_flight'] += 1
    started = time.time()
    state = {'first_chunk_at': None, 'audio_sec': 0.0, 'errored': False}

    def gen():
        try:
            set_all_random_seed(seed)
            for j in _model.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=True):
                tensor = j['tts_speech'].squeeze().contiguous()
                # tts_speech is float in [-1, 1]; encode to int16 PCM
                int16 = (tensor.clamp(-1, 1) * 32767).to(torch.int16)
                pcm_bytes = int16.cpu().numpy().tobytes()
                if state['first_chunk_at'] is None:
                    state['first_chunk_at'] = time.time()
                state['audio_sec'] += tensor.shape[-1] / sr
                yield pcm_bytes
        except Exception as e:
            state['errored'] = True
            print(f'[stream] inference error: {type(e).__name__}: {e}', flush=True)
            # Can't raise HTTPException after streaming started; just log.
        finally:
            wall = time.time() - started
            ttfa = (state['first_chunk_at'] - started) if state['first_chunk_at'] else None
            with _metrics_lock:
                _metrics['in_flight'] -= 1
                if state['errored']:
                    _metrics['errors'] += 1
                else:
                    _metrics['requests'] += 1
                    _metrics['audio_seconds'] += state['audio_sec']
                    _metrics['total_wall_seconds'] += wall
                    if ttfa is not None:
                        _metrics.setdefault('ttfa_samples', []).append(ttfa)
                        # cap memory: keep last 1000
                        if len(_metrics['ttfa_samples']) > 1000:
                            _metrics['ttfa_samples'] = _metrics['ttfa_samples'][-1000:]

    return StreamingResponse(
        gen(),
        media_type='audio/L16',
        headers={
            'X-Sample-Rate': str(sr),
            'X-Channels':    '1',
            'X-Format':      'int16',
        },
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
