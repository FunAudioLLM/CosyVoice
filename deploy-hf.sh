#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate

#conda create -n cosyvoice python=3.8
#conda activate cosyvoice
#conda install -y -c conda-forge pynini==2.1.5
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
pip install -r requirements_.txt

# If you encounter sox compatibility issues
# ubuntu
apt-get -y update && apt-get -y install sox libsox-dev

mkdir -p pretrained_models
#git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
#git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
#git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
#git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
#git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
#huggingface-cli download model-scope/CosyVoice-300M --local-dir pretrained_models/CosyVoice-300M --token=$hf_token
#huggingface-cli download model-scope/CosyVoice-300M-SFT --local-dir pretrained_models/CosyVoice-300M-SFT --token=$hf_token
#huggingface-cli download FunAudioLLM/CosyVoice-ttsfrd --local-dir pretrained_models/CosyVoice-ttsfrd --token=$hf_token

ls pretrained_models

cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl

export PYTHONPATH=third_party/Matcha-TTS

python3 webui.py