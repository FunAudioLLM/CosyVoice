import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)


# 使用 SFT 模型，遍历所有集成的 speaker 并合成语音
spk_ids = cosyvoice.list_available_spks()
print('available spk_id:', spk_ids)
