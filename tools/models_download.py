# Model download using SDK
from modelscope import snapshot_download
import os

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Model name constants
MODEL_NAMES = {
    "base": "CosyVoice2-0.5B",
    "hz25": "CosyVoice-300M-25Hz",
    "sft": "CosyVoice-300M-SFT",
    "instruct": "CosyVoice-300M-Instruct",
    "ttsfrd": "CosyVoice-ttsfrd"
}

# Model storage directory name
MODELS_DIR = "pretrained_models"

# Download all models
for model_key, model_name in MODEL_NAMES.items():
    target_dir = os.path.join(PARENT_DIR, MODELS_DIR, model_name)
    snapshot_download(f"iic/{model_name}", local_dir=target_dir)