#!/bin/bash
# Expose all NVIDIA CUDA shared libraries from the venv to LD_LIBRARY_PATH
# so onnxruntime-gpu can find libcublasLt, libcudnn, etc.

VENV_NV="/home/zhiqiang/.venvs/cosyvoice/lib/python3.10/site-packages/nvidia"

paths=()
for sub in cudnn cublas cuda_runtime curand cufft cusolver cusparse nccl nvjitlink cuda_nvrtc cuda_cupti; do
  d="$VENV_NV/$sub/lib"
  [ -d "$d" ] && paths+=("$d")
done

joined=$(IFS=:; echo "${paths[*]}")

if [ -n "$LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$joined:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$joined"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
