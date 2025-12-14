import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('third_party/Matcha-TTS')
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)

tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'

spk_ids = ['中文女', '中文男', '粤语女']
# print('available spk_id:', spk_ids)

for spk_id in spk_ids:
    chunks = []
    for out in cosyvoice.inference_sft(tts_text, spk_id, stream=False):
        chunks.append(out['tts_speech'])
    if len(chunks) == 0:
        continue
    speech = torch.cat(chunks, dim=1)
    filename = f'sft_{spk_id}.wav'
    torchaudio.save(filename, speech, cosyvoice.sample_rate)
    print('saved', filename)
