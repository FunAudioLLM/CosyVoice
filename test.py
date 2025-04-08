import sys
import os
sys.path.append('third_party/Matcha-TTS')

# set test configuration
DEBUG = True
use_stream = True
model_dir='pretrained_models/CosyVoice2-0.5B-trt'
prompt_wav = './test/assets/voice_kowoon.wav'
prompt_text = '안녕하세요 저는 고운입니다. 제가 오늘 발표드릴 주제는 인공지능입니다. 잘 부탁 드립니다.'
output_file ='./test/results/stream_go_{}.wav'

if DEBUG:
    from viztracer import VizTracer
    tracer = VizTracer()
    tracer.start()

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=True, fp16=True)

prompt_speech_16k = load_wav(prompt_wav, 16000)

def text_generator():
    yield "세상은 매일 진화하고 있습니다."
    yield "AI는 이제 단순한 도구가 아니라,"
    yield "우리 삶의 일부가 되었죠."
    yield "오늘, 그 놀라운 변화를 함께 만나봅니다."

idx = 0
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), prompt_text, prompt_speech_16k, stream=use_stream)):
    torchaudio.save(output_file.format(i), j['tts_speech'], cosyvoice.sample_rate)
    idx = i

if DEBUG:
    tracer.stop()
    tracer.save()

    #combine all wav files into one
    if use_stream:
        from pydub import AudioSegment
        tmp = AudioSegment.empty()
        for i in range(idx + 1):
            tmp += AudioSegment.from_wav(output_file.format(i))
            os.remove(output_file.format(i))
        tmp.export("test/results/stream_go.wav", format="wav")
