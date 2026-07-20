CONTAINER_NAME="cosyvoice-vllm-server"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  docker stop ${CONTAINER_NAME}
  docker rm ${CONTAINER_NAME}
fi

docker run --gpus all --rm -it \
  --name "${CONTAINER_NAME}" \
  -p 18000:18000 \
  -p 18001:18001 \
  -p 18002:18002 \
  -e PYTHONIOENCODING=utf-8 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_USE_V1=1 \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  -e FLASHINFER_ENABLED=0 \
  --shm-size=4g \
  -v $HOME/.cache/modelscope:/root/.cache/modelscope \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosyvoice_vllm_server:latest \
  /bin/bash -c "python3 vllm_example.py"

