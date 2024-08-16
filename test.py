from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import logging

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
torch.cuda.nvtx.range_push("cosyvoice inference")
output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
for idx, itr in enumerate(output):
    torchaudio.save(f"zero_shot_{idx}.wav", itr['tts_speech'], 22050)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
logging.info("######################")


torch.cuda.nvtx.range_push("cosyvoice inference")
output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
for idx, itr in enumerate(output):
    torchaudio.save(f"zero_shot_{idx}.wav", itr['tts_speech'], 22050)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
logging.info("######################")


torch.cuda.nvtx.range_push("cosyvoice inference")
output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
for idx, itr in enumerate(output):
    torchaudio.save(f"zero_shot_{idx}.wav", itr['tts_speech'], 22050)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()