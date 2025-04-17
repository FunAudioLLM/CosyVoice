# syntax=docker/dockerfile:1

# FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++
RUN git lfs install

ENV http_proxy=socks5://183.240.180.158:10080
ENV https_proxy=socks5://183.240.180.158:10080
ENV all_proxy=socks5://183.240.180.158:10080
RUN git config --global http.proxy socks5://183.240.180.158:10080
RUN git config --global https.proxy socks5://183.240.180.158:10080

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed

# 修改requirements.txt文件，将TensorRT的CUDA 12版本改为普通版本
RUN sed -i 's/tensorrt-cu12==10.0.1/# tensorrt==9.0.0.post11.dev1/g' /opt/CosyVoice/CosyVoice/requirements.txt && \
    sed -i 's/tensorrt-cu12-bindings==10.0.1/# tensorrt-bindings==9.0.0.post11.dev1/g' /opt/CosyVoice/CosyVoice/requirements.txt && \
    sed -i 's/tensorrt-cu12-libs==10.0.1/# tensorrt-libs==9.0.0.post11.dev1/g' /opt/CosyVoice/CosyVoice/requirements.txt && \
    sed -i 's/torch==2.3.1/# torch==2.3.1/g' /opt/CosyVoice/CosyVoice/requirements.txt && \
    sed -i 's/torchaudio==2.3.1/# torchaudio==2.3.1/g' /opt/CosyVoice/CosyVoice/requirements.txt

# 先升级pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# 先安装onnxruntime-gpu，这是最耗时的部分
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --verbose onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/


# 再安装其他依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    cd CosyVoice && pip3 install --verbose -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host=pypi.tuna.tsinghua.edu.cn

RUN cd CosyVoice/runtime/python/grpc && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto