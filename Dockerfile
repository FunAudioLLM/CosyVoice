FROM vllm/vllm-openai:v0.11.0
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 依赖安装
RUN pip install \
    onnx==1.16.0 \
    onnxruntime-gpu==1.22.0 \
    HyperPyYAML==1.2.3 \
    transformers==4.57.1 \
    numpy==1.26.4 \
    openai-whisper==20250625 \
    inflect==7.3.1 \
    omegaconf==2.3.0 \
    wetext==0.0.4 \
    conformer==0.3.2 \
    diffusers==0.29.0 \
    hydra-core==1.3.2 \
    lightning==2.2.4 \
    gdown==5.1.0 \
    matplotlib==3.7.5 \
    wget==3.2 \
    x-transformers==2.11.24 \
    librosa==0.10.2 \
    pyarrow==18.1.0 \
    pyworld==0.3.4 \
    tensorboard==2.14.0 \
    tensorrt-cu12==10.13.3.9 \
    tensorrt-cu12-bindings==10.13.3.9 \
    tensorrt-cu12-libs==10.13.3.9
# 复制源码
COPY . .

ENTRYPOINT []
CMD ["/bin/bash"]