#!/bin/bash
# Launch CosyVoice TTS server with optimizations:
#   - LD_LIBRARY_PATH set so onnxruntime-gpu finds cuDNN 8 + cuBLAS / cudart
#   - FE prompt-cache enabled (in server_cosyvoice3.py via enable_fe_cache)
#   - TRT engine + vLLM continuous batching
#   - No model lock (vLLM thread-safe, CosyVoice tolerated)
#
# Usage: bash run_server.sh

set -euo pipefail

VENV=/home/zhiqiang/.venvs/cosyvoice
NV=$VENV/lib/python3.10/site-packages/nvidia
REPO=/home/zhiqiang/repos/CosyVoice
LOG=/home/zhiqiang/server.log

paths=()
for sub in cudnn cublas cuda_runtime curand cufft cusolver cusparse nccl nvjitlink cuda_nvrtc cuda_cupti; do
  d="$NV/$sub/lib"
  [ -d "$d" ] && paths+=("$d")
done
joined=$(IFS=:; echo "${paths[*]}")
export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export LOAD_TRT=${LOAD_TRT:-1}
export MODEL_DIR=${MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}

cd "$REPO"
echo "[launcher] LD_LIBRARY_PATH set, starting server ..."
exec "$VENV/bin/python" -u server_cosyvoice3.py
