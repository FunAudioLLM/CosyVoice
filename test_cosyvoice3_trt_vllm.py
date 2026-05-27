"""CosyVoice3 with TRT+vLLM. First run compiles TRT engine (5-15 min)."""
import sys, time
sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
import torchaudio


def main():
    print('Loading CosyVoice3 with TRT + vLLM (first run compiles TRT engine, may take 5-15min)...', flush=True)
    t0 = time.time()
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    print(f'Loaded in {time.time()-t0:.2f}s', flush=True)

    text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
    prompt_wav = './asset/zero_shot_prompt.wav'

    for run in range(5):
        set_all_random_seed(run)
        t1 = time.time()
        total_audio_seconds = 0.0
        for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)):
            out = f'/home/zhiqiang/trt_test_{run}_{i}.wav'
            torchaudio.save(out, j['tts_speech'], cosyvoice.sample_rate)
            total_audio_seconds += j['tts_speech'].shape[-1] / cosyvoice.sample_rate
        elapsed = time.time() - t1
        print(f'[run {run}] wall={elapsed:.2f}s audio={total_audio_seconds:.2f}s RTF={elapsed/total_audio_seconds:.3f}', flush=True)


if __name__ == '__main__':
    main()
