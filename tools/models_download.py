# SDK模型下载
from modelscope import snapshot_download
import os

# 获取路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# 模型名称常量
MODEL_NAMES = {
    "base": "CosyVoice2-0.5B",
    "hz25": "CosyVoice-300M-25Hz",
    "sft": "CosyVoice-300M-SFT",
    "instruct": "CosyVoice-300M-Instruct",
    "ttsfrd": "CosyVoice-ttsfrd"
}

# 模型存储目录名
MODELS_DIR = "pretrained_models"

# 下载所有模型
for model_key, model_name in MODEL_NAMES.items():
    target_dir = os.path.join(PARENT_DIR, MODELS_DIR, model_name)
    snapshot_download(f"iic/{model_name}", local_dir=target_dir)