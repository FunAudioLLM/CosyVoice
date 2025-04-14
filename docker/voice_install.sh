#!/bin/bash
# 添加错误处理函数
# 注意这里没有包含安装 ros2 humble， 需要另行安装

# 设置命令回显
set -x

# 定义函数用于显示安装步骤
show_step() {
    echo "===================================================="
    echo "🔶 步骤: $@"
    echo "===================================================="
}


show_step "###################正式安装从这里开始#####################################"

# 1. 安装通用包，基础的环境
show_step "安装通用包，基础的环境"
apt-get update -y --fix-missing
apt-get install -y git curl wget ffmpeg unzip git-lfs sox libsox-dev && \
    apt-get clean

# 安装通用包
apt-get install -y apt ssh make gcc curl cmake g++ unzip lsof net-tools

# 安装ROS2 依赖的包， 并不安装 ros2 humble
apt-get install -y python3-pip ament-cmake

# 2. 安装语音需要的系统级的包或库
show_step "安装语音系统级依赖包"

# 分批安装包，以减少错误风险
# 基础开发工具
apt-get install -y pkg-config libfftw3-dev nlohmann-json3-dev libeigen3-dev

# 音频相关库
apt-get install -y libsndfile1-dev pulseaudio pulseaudio-utils

# Mesa相关
apt-get install -y mesa-utils libglu1-mesa-dev

# Pybind相关
apt-get install -y python3-pybind11 pybind11-dev

# 音频编解码相关
apt-get install -y libmpg123-dev libmad0-dev libsndio-dev libwebrtc-audio-processing-dev libwavpack-dev

# 视频编解码相关
apt-get install -y libavcodec-dev libavc1394-dev

# 3. 安装语音需要的 python 包
show_step "安装语音需要的 python 包, 放入一个轻量级的虚拟环境中，没有使用过重的conda"

echo "当前目录: $(pwd)"
# 安装轻量级的 venv
pip install virtualenv

virtualenv venv_voice
source venv_voice/bin/activate
# source /disk1/venv_torch/bin/activate

# 创建requirements.txt文件
cat > requirements.txt << EOF
torch==2.6.0+cpu
torchaudio==2.6.0
torchvision==0.21.0
yeaudio==0.0.7
tqdm==4.67.1
SoundCard==0.4.3
scikit-learn==1.6.1
scipy==1.15.1
pybind11==2.13.6
pip==25.0.1
llvmlite==0.44.0
kaldi-native-fbank==1.20.2
catkin-pkg==1.0.0
librosa==0.10.2
EOF

# 在上面venv_voice中pip安装一大堆包
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

echo "所有依赖安装完成"