#!/bin/bash
set -e

CONTAINER_NAME="cosyvoice-vllm-server"
TTS_PORT=${TTS_PORT:-10096}

# ---- stop & remove existing container ----
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "==> Stopping & removing old container: ${CONTAINER_NAME}"
  docker stop ${CONTAINER_NAME} 2>/dev/null || true
  docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

echo "==> Starting CosyVoice TTS WebSocket server on port ${TTS_PORT}"

# ---- vLLM memory tuning (override via env) ----
VLLM_MAX_MODEL_LEN=${COSYVOICE_VLLM_MAX_MODEL_LEN:-4096}
VLLM_GPU_MEM_UTIL=${COSYVOICE_VLLM_GPU_MEM_UTIL:-0.05}

echo "   vLLM max_model_len=${VLLM_MAX_MODEL_LEN}  gpu_mem_util=${VLLM_GPU_MEM_UTIL}"

docker run --gpus all -d -it \
  --name "${CONTAINER_NAME}" \
  -p ${TTS_PORT}:10096 \
  -e PYTHONIOENCODING=utf-8 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_USE_V1=1 \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  -e FLASHINFER_ENABLED=0 \
  -e COSYVOICE_VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} \
  -e COSYVOICE_VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL} \
  --shm-size=4g \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosyvoice_vllm_server:latest \
  python3 serve_realtime_ws.py \
    --port 10096 \
    --model-dir FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --prompt-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
    --prompt-wav asset/zero_shot_prompt.wav

