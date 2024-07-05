FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install python3-dev cmake python3-pip git
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
RUN cd CosyVoice && pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN cd CosyVoice/runtime/python && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
CMD ["/bin/bash", "-c", "cd /opt/CosyVoice/CosyVoice/runtime/python && . ./path/sh && python3 server.py --port 50000 --max_conc 4 --model_dir speech_tts/CosyVoice-300M && sleep infinity"]