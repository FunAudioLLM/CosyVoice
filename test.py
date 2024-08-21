from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import logging

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
print("finish model initialize")

test_sentences = ['欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务', '欢迎使用蚂蚁集团语音合成服务']

for idx, sent in enumerate(test_sentences):
    output = cosyvoice.inference_sft(sent, '中文女')
    for _, itr in enumerate(output):
        torchaudio.save(f"zero_shot_{idx}.wav", itr['tts_speech'], 22050)
    torch.cuda.synchronize()
    print("-------\n")

