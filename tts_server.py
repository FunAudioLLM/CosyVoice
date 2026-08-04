"""
CosyVoice TTS HTTP 服务 — 兼容 livekit-agent 的 /v1/tts 接口。

接收 JSON {"text": "...", "format": "wav"}，返回 WAV 音频字节。
使用 CosyVoice 3.0 zero-shot 语音克隆，默认 prompt 音色可配置。

环境变量:
    COSYVOICE_MODEL_DIR    模型目录或 ModelScope ID（默认 FunAudioLLM/Fun-CosyVoice3-0.5B-2512）
    COSYVOICE_PROMPT_TEXT  zero-shot prompt 文本
    COSYVOICE_PROMPT_WAV   zero-shot prompt 音频路径
    COSYVOICE_PORT         服务端口（默认 9200）
    COSYVOICE_DEVICE       cpu / cuda:0（默认 cuda:0）
"""

from __future__ import annotations

import io
import logging
import os
import sys
import wave

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── 路径设置 ────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cosyvoice-tts")

# ── 配置 ────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
PROMPT_TEXT = os.environ.get(
    "COSYVOICE_PROMPT_TEXT",
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
)
PROMPT_WAV = os.environ.get("COSYVOICE_PROMPT_WAV", os.path.join(ROOT_DIR, "asset", "zero_shot_prompt.wav"))
DEVICE = os.environ.get("COSYVOICE_DEVICE", "cuda:0")
PORT = int(os.environ.get("COSYVOICE_PORT", "9200"))

# ── 推理路径：RTX 2080 Ti 必须用 vLLM + TensorRT ──────────────────────
# 环境变量 LOAD_VLLM/LOAD_TRT/FP16 控制推理后端。
# 原生 PyTorch (VLLM=false TRT=false) 在 SM 7.5 GPU 上会产生杂音。
LOAD_VLLM = os.environ.get("COSYVOICE_LOAD_VLLM", "true").lower() in ("true", "1", "yes")
LOAD_TRT = os.environ.get("COSYVOICE_LOAD_TRT", "true").lower() in ("true", "1", "yes")
FP16 = os.environ.get("COSYVOICE_FP16", "false").lower() in ("true", "1", "yes")

# ── FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(title="cosyvoice-tts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型在 FastAPI startup 阶段预加载，避免服务已监听但首次请求才触发冷启动。
_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("⏳ 加载 CosyVoice 模型: %s  (device=%s, vllm=%s, trt=%s, fp16=%s) …",
                    MODEL_DIR, DEVICE, LOAD_VLLM, LOAD_TRT, FP16)
        from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
        _model = AutoModel(model_dir=MODEL_DIR, load_trt=LOAD_TRT, load_vllm=LOAD_VLLM, fp16=FP16)
        logger.info("✅ CosyVoice 模型就绪，sample_rate=%d", _model.sample_rate)
    return _model


@app.on_event("startup")
def preload_model() -> None:
    """Block readiness until vLLM/TensorRT and CosyVoice are initialized."""
    get_model()


# ── 辅助函数 ────────────────────────────────────────────────────────


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """将 raw PCM int16 → WAV (RIFF header)"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(bits_per_sample // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class TTSRequest(BaseModel):
    text: str
    format: str = "wav"


# ── API ─────────────────────────────────────────────────────────────


@app.get("/healthz")
async def healthz() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="CosyVoice model is not ready")
    return {"status": "ok", "model_ready": True, "sample_rate": _model.sample_rate}


@app.post("/v1/tts")
async def synthesize(req: TTSRequest) -> Response:
    """文本转语音，返回 WAV 音频。兼容 livekit-agent 的 TTS 接口。"""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    model = get_model()

    # zero-shot 推理（prompt_wav 直接传文件路径，frontend 内部会调用 load_wav）
    try:
        pcm_chunks: list[bytes] = []
        for output in model.inference_zero_shot(req.text.strip(), PROMPT_TEXT, PROMPT_WAV, stream=False):
            # output['tts_speech'] 是 torch.Tensor (1D, float32, 范围 [-1,1])
            speech = output["tts_speech"]
            if hasattr(speech, "numpy"):
                speech = speech.numpy()
            pcm = (speech * (2**15)).astype(np.int16).tobytes()
            pcm_chunks.append(pcm)
    except Exception as e:
        logger.error("❌ TTS 推理失败: %s", e)
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {e}")

    pcm_all = b"".join(pcm_chunks)

    if req.format == "wav":
        wav_bytes = pcm_to_wav(pcm_all, model.sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")
    else:
        return Response(content=pcm_all, media_type="audio/L16;rate={}".format(model.sample_rate))


# ── 启动 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 CosyVoice TTS 服务启动: http://0.0.0.0:%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
