import sys, time
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

print('Loading CosyVoice3...', flush=True)
t0 = time.time()
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=False, load_vllm=False, fp16=False)
print(f'Loaded in {time.time()-t0:.2f}s', flush=True)

text = '你好，欢迎来到 CosyVoice 三号的世界，今天我们一起来测试一下它的中文合成效果。'
prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
prompt_wav = './asset/zero_shot_prompt.wav'

print('Running inference...', flush=True)
t1 = time.time()
total_audio_seconds = 0.0
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)):
    out = f'/home/zhiqiang/zero_shot_test_{i}.wav'
    torchaudio.save(out, j['tts_speech'], cosyvoice.sample_rate)
    dur = j['tts_speech'].shape[-1] / cosyvoice.sample_rate
    total_audio_seconds += dur
    print(f'chunk {i}: saved {out}, audio_dur={dur:.2f}s, sr={cosyvoice.sample_rate}', flush=True)
elapsed = time.time() - t1
print(f'Inference done in {elapsed:.2f}s, total_audio={total_audio_seconds:.2f}s, RTF={elapsed/total_audio_seconds:.3f}', flush=True)
