FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 设置阿里云镜像源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update -y --fix-missing
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

RUN git config --global http.proxy socks5://183.240.180.158:10080
RUN git config --global https.proxy socks5://183.240.180.158:10080
# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh 
RUN  /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

# 先升级pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip


WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
RUN conda activate ${VENV}  
RUN cd CosyVoice2-Ex 

ENV http_proxy=socks5://183.240.180.158:10080
ENV https_proxy=socks5://183.240.180.158:10080
ENV all_proxy=socks5://183.240.180.158:10080

WORKDIR /workspace/CosyVoice2-Ex
# 下一句要实现打印当前目录
RUN echo "当前目录: $(pwd)"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


ENV LOG_LEVEL=INFO

EXPOSE 8080

CMD ["sh", "-c", "python webui.py --port 8080 --log_level ${LOG_LEVEL}"]
