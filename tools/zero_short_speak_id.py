import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('third_party/Matcha-TTS')
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
voice_file = 'asset/zero_shot_prompt.wav'
prompt_speech_16k = load_wav(voice_file, 16000)
prompt_text_file = Path(voice_file).with_suffix(".txt")
if prompt_text_file.exists():
    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

# save zero_shot spk as a voice_id for future usage
assert cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, 'my_zero_shot_spk') is True
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    torchaudio.save('speaker_id_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# contiue to use "my_zero_shot_spk" instead of "prompt_speech_16k"
...
# this step is optional unless you want to persist "my_zero_shot_spk" into "model_dir/spk2info.pt" for a new session
cosyvoice.save_spkinfo()
