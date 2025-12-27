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
"""
CosyVoice FastAPI Server with GPU Optimization Support

This server provides TTS inference endpoints with:
- Automatic GPU optimization based on hardware detection
- Real-time GPU monitoring and statistics
- Performance benchmarking endpoints
- Dynamic configuration updates
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import numpy as np
import uvicorn

try:
    import uvloop

    uvloop.install()
except ImportError:
    pass

try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    from fastapi.responses import JSONResponse as ORJSONResponse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# GPU monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available - GPU monitoring will be limited")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(ROOT_DIR))
sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI(
    title="CosyVoice TTS API",
    description="GPU-accelerated text-to-speech with automatic optimization",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class GpuStats:
    """Current GPU statistics."""

    available: bool = False
    name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    utilization_gpu: int = 0
    utilization_memory: int = 0
    temperature: int = 0
    power_draw_watts: float = 0.0


@dataclass
class PerformanceStats:
    """Server performance statistics."""

    total_requests: int = 0
    total_audio_seconds: float = 0.0
    average_rtf: float = 0.0
    min_rtf: float = float("inf")
    max_rtf: float = 0.0
    average_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class OptimizationConfig:
    """Current optimization configuration."""

    fp16_enabled: bool = False
    tensorrt_enabled: bool = False
    trt_concurrent: int = 1
    vllm_enabled: bool = False
    model_type: str = "CosyVoice"
    model_dir: str = ""


# Global state
class ServerState:
    def __init__(self):
        self.cosyvoice = None
        self.config = OptimizationConfig()
        self.stats = PerformanceStats()
        self.start_time = time.time()
        self._init_gpu_monitoring()

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring with pynvml."""
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logging.info("GPU monitoring initialized with pynvml")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU monitoring: {e}")

    def get_gpu_stats(self) -> GpuStats:
        """Get current GPU statistics."""
        stats = GpuStats()

        if not PYNVML_AVAILABLE or self.gpu_handle is None:
            # Fallback to nvidia-smi
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(",")]
                    if len(parts) >= 8:
                        stats.available = True
                        stats.name = parts[0]
                        stats.memory_total_mb = int(float(parts[1]))
                        stats.memory_used_mb = int(float(parts[2]))
                        stats.memory_free_mb = int(float(parts[3]))
                        stats.utilization_gpu = int(float(parts[4]))
                        stats.utilization_memory = int(float(parts[5]))
                        stats.temperature = int(float(parts[6]))
                        try:
                            stats.power_draw_watts = float(parts[7])
                        except:
                            stats.power_draw_watts = 0.0
            except Exception as e:
                logging.debug(f"nvidia-smi fallback failed: {e}")
            return stats

        try:
            stats.available = True
            stats.name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            if isinstance(stats.name, bytes):
                stats.name = stats.name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            stats.memory_total_mb = mem_info.total // (1024 * 1024)
            stats.memory_used_mb = mem_info.used // (1024 * 1024)
            stats.memory_free_mb = mem_info.free // (1024 * 1024)

            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            stats.utilization_gpu = util.gpu
            stats.utilization_memory = util.memory

            stats.temperature = pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                stats.power_draw_watts = power / 1000.0  # mW to W
            except:
                pass

        except Exception as e:
            logging.warning(f"Error getting GPU stats: {e}")
            stats.available = False

        return stats

    def record_request(self, audio_seconds: float, latency_ms: float):
        """Record a completed request for statistics."""
        rtf = (latency_ms / 1000.0) / max(audio_seconds, 0.001)

        n = self.stats.total_requests
        self.stats.total_requests += 1
        self.stats.total_audio_seconds += audio_seconds

        # Running averages
        self.stats.average_rtf = (self.stats.average_rtf * n + rtf) / (n + 1)
        self.stats.average_latency_ms = (
            self.stats.average_latency_ms * n + latency_ms
        ) / (n + 1)

        # Min/max
        if n == 0:
            self.stats.min_rtf = rtf
            self.stats.max_rtf = rtf
        else:
            self.stats.min_rtf = min(self.stats.min_rtf, rtf)
            self.stats.max_rtf = max(self.stats.max_rtf, rtf)

        self.stats.uptime_seconds = time.time() - self.start_time


state = ServerState()


def generate_data(
    model_output, state_ref: ServerState = None, start_time: float = None
):
    """Generator for streaming audio data with metrics collection."""
    total_samples = 0
    for i in model_output:
        audio = i["tts_speech"].numpy()
        total_samples += audio.shape[1] if len(audio.shape) > 1 else len(audio)
        tts_audio = (audio * (2**15)).astype(np.int16).tobytes()
        yield tts_audio

    # Record metrics after generation completes
    if state_ref and start_time:
        sample_rate = 22050  # CosyVoice output sample rate
        audio_seconds = total_samples / sample_rate
        latency_ms = (time.time() - start_time) * 1000
        state_ref.record_request(audio_seconds, latency_ms)


# ==============================================================================
# Health and Status Endpoints
# ==============================================================================


@app.get("/")
async def root():
    """Root endpoint - returns server info."""
    return {
        "name": "CosyVoice TTS Server",
        "version": "2.0.0",
        "status": "running",
        "model_type": state.config.model_type,
        "gpu_available": state.get_gpu_stats().available,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_stats = state.get_gpu_stats()
    return {
        "status": "healthy",
        "model_loaded": state.cosyvoice is not None,
        "gpu_available": gpu_stats.available,
        "gpu_name": gpu_stats.name,
        "uptime_seconds": time.time() - state.start_time,
    }


@app.get("/gpu_stats")
async def gpu_stats():
    """Get current GPU statistics."""
    stats = state.get_gpu_stats()
    return asdict(stats)


@app.get("/performance_stats")
async def performance_stats():
    """Get server performance statistics."""
    state.stats.uptime_seconds = time.time() - state.start_time
    return asdict(state.stats)


@app.get("/config")
async def get_config():
    """Get current optimization configuration."""
    return asdict(state.config)


@app.get("/speakers")
async def list_speakers():
    """List available speakers for SFT mode."""
    if state.cosyvoice is None:
        return []
    try:
        speakers = state.cosyvoice.list_available_spks()
        return speakers
    except Exception as e:
        logging.warning(f"Failed to list speakers: {e}")
        return ["中文女", "中文男", "英文女", "英文男"]


# ==============================================================================
# Inference Endpoints
# ==============================================================================


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    """SFT (Speaker Fine-Tuned) inference."""
    start_time = time.time()
    model_output = state.cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(
        generate_data(model_output, state, start_time), media_type="audio/raw"
    )


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
):
    """Zero-shot voice cloning inference."""
    start_time = time.time()
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = state.cosyvoice.inference_zero_shot(
        tts_text, prompt_text, prompt_speech_16k
    )
    return StreamingResponse(
        generate_data(model_output, state, start_time), media_type="audio/raw"
    )


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(), prompt_wav: UploadFile = File()
):
    """Cross-lingual voice cloning inference."""
    start_time = time.time()
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = state.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(
        generate_data(model_output, state, start_time), media_type="audio/raw"
    )


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
    tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()
):
    """Instruct-based inference with style/emotion control."""
    start_time = time.time()
    model_output = state.cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(
        generate_data(model_output, state, start_time), media_type="audio/raw"
    )


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()
):
    """Instruct2 inference (CosyVoice2/3 only)."""
    start_time = time.time()
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = state.cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k
    )
    return StreamingResponse(
        generate_data(model_output, state, start_time), media_type="audio/raw"
    )


# ==============================================================================
# Benchmarking Endpoints
# ==============================================================================


@app.post("/benchmark")
async def benchmark(
    text: str = Form(
        default="This is a benchmark test for measuring inference performance."
    ),
    iterations: int = Form(default=3),
):
    """
    Run a quick benchmark to measure RTF (Real-Time Factor).

    Returns average RTF across iterations - lower is better.
    RTF < 1.0 means faster than realtime.
    """
    if state.cosyvoice is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get a speaker for testing
    speakers = state.cosyvoice.list_available_spks()
    if not speakers:
        raise HTTPException(status_code=503, detail="No speakers available")

    speaker = speakers[0]
    iterations = min(max(iterations, 1), 10)  # Clamp to 1-10

    rtfs = []
    latencies = []

    for i in range(iterations):
        start_time = time.time()
        total_samples = 0

        for output in state.cosyvoice.inference_sft(text, speaker):
            audio = output["tts_speech"].numpy()
            total_samples += audio.shape[1] if len(audio.shape) > 1 else len(audio)

        latency = time.time() - start_time
        audio_seconds = total_samples / 22050
        rtf = latency / max(audio_seconds, 0.001)

        rtfs.append(rtf)
        latencies.append(latency * 1000)

    return {
        "text_length": len(text),
        "iterations": iterations,
        "average_rtf": sum(rtfs) / len(rtfs),
        "min_rtf": min(rtfs),
        "max_rtf": max(rtfs),
        "average_latency_ms": sum(latencies) / len(latencies),
        "gpu_utilization": state.get_gpu_stats().utilization_gpu,
        "inference_faster_than_realtime": sum(rtfs) / len(rtfs) < 1.0,
    }


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CosyVoice TTS Server with GPU Optimization"
    )
    parser.add_argument("--port", type=int, default=50000, help="Port to listen on")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="iic/CosyVoice-300M",
        help="Model directory or ModelScope repo ID",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference")
    parser.add_argument(
        "--load-trt", action="store_true", help="Enable TensorRT acceleration"
    )
    parser.add_argument(
        "--trt-concurrent",
        type=int,
        default=1,
        help="TensorRT concurrent execution contexts",
    )
    parser.add_argument(
        "--load-vllm", action="store_true", help="Enable vLLM for LLM acceleration"
    )
    parser.add_argument(
        "--load-jit", action="store_true", help="Enable JIT compilation"
    )
    args = parser.parse_args()

    # Update config
    state.config.model_dir = args.model_dir
    state.config.fp16_enabled = args.fp16
    state.config.tensorrt_enabled = args.load_trt
    state.config.trt_concurrent = args.trt_concurrent
    state.config.vllm_enabled = args.load_vllm

    logging.info(f"Loading model from: {args.model_dir}")
    logging.info(
        f"Configuration: FP16={args.fp16}, TRT={args.load_trt}, "
        f"TRT_concurrent={args.trt_concurrent}, vLLM={args.load_vllm}"
    )

    # Try loading model with various configurations
    try:
        cosyvoice = CosyVoice(
            args.model_dir,
            load_jit=args.load_jit,
            load_trt=args.load_trt,
            fp16=args.fp16,
            trt_concurrent=args.trt_concurrent,
        )
        state.config.model_type = "CosyVoice"
    except Exception as e1:
        logging.info(f"CosyVoice init failed ({e1}), trying CosyVoice2...")
        try:
            cosyvoice = CosyVoice2(
                args.model_dir,
                load_jit=args.load_jit,
                load_trt=args.load_trt,
                load_vllm=args.load_vllm,
                fp16=args.fp16,
                trt_concurrent=args.trt_concurrent,
            )
            state.config.model_type = "CosyVoice2"
        except Exception as e2:
            logging.error(f"Failed to load model: {e1}, {e2}")
            raise TypeError("No valid model_type!")

    state.cosyvoice = cosyvoice

    # Log GPU info at startup
    gpu_stats = state.get_gpu_stats()
    if gpu_stats.available:
        logging.info(f"GPU: {gpu_stats.name}")
        logging.info(
            f"VRAM: {gpu_stats.memory_total_mb} MiB total, {gpu_stats.memory_free_mb} MiB free"
        )
        logging.info(f"GPU Utilization: {gpu_stats.utilization_gpu}%")
    else:
        logging.warning("No GPU detected - inference will be slow")

    logging.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
