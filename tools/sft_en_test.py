import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('third_party/Matcha-TTS')
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)

en_spk_ids = ['英文女', '英文男']

tts_text = 'Maintaining your ability to learn translates into increased marketability, improved career optionsand higher salaries.'

for spk_id in en_spk_ids:
    chunks = []
    for out in cosyvoice.inference_sft(tts_text, spk_id, stream=False):
        chunks.append(out['tts_speech'])
    if len(chunks) == 0:
        continue
    speech = torch.cat(chunks, dim=1)
    filename = f'sft_en_{spk_id}.wav'
    torchaudio.save(filename, speech, cosyvoice.sample_rate)
    print('saved', filename)
