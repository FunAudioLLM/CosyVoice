#!/bin/bash
git checkout main
git pull
sudo apt-get install vim unzip -y
tar -xvf test.tar.gz
cp ./pretrained_models/CosyVoice2-0.5B ./pretrained_models/CosyVoice2-0.5B-trt -r
./cosyvoice/bin/export_trt.sh
cd pretrained_models/CosyVoice-ttsfrd
unzip resource.zip -d . && \
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl