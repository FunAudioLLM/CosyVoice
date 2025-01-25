FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs
RUN git lfs install
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
RUN cd CosyVoice && pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# To solve the CUFFT_INTERNAL_ERROR bug in cu11.7 pytorch, see https://forums.developer.nvidia.com/t/bug-ubuntu-on-wsl2-rtx4090-related-cufft-runtime-error/230883
RUN pip uninstall torch torchaudio
RUN pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN cd CosyVoice/runtime/python/grpc && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto