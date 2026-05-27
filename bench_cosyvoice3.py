"""QPS benchmark for CosyVoice3.

Usage:
    python bench_cosyvoice3.py            # vllm only
    python bench_cosyvoice3.py --trt      # vllm + trt
    python bench_cosyvoice3.py --no-vllm  # baseline (no acceleration)
"""
import sys, time, statistics, threading, queue, argparse
sys.path.append('third_party/Matcha-TTS')

PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
PROMPT_WAV = './asset/zero_shot_prompt.wav'

TEXTS = {
    'short':  '你好，今天天气真不错。',
    'medium': '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    'long':   '在人工智能技术飞速发展的今天，语音合成已经从早期生硬的拼接方式，进化到如今能够表达丰富情感、自然流畅的神经网络模型。CosyVoice 作为阿里达摩院推出的多语言语音生成模型，在零样本音色克隆、跨语种合成、多方言支持等方面都展现出了令人惊艳的能力，为众多应用场景带来了新的可能性。',
}


def run_once(model, text, seed=0):
    from cosyvoice.utils.common import set_all_random_seed
    set_all_random_seed(seed)
    t0 = time.time()
    audio_sec = 0.0
    for _, j in enumerate(model.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=False)):
        audio_sec += j['tts_speech'].shape[-1] / model.sample_rate
    return time.time() - t0, audio_sec


def bench_sequential(model, iters=5):
    print('\n=== Sequential ===', flush=True)
    for name, text in TEXTS.items():
        run_once(model, text, seed=99)  # warmup
        walls, audios = [], []
        for i in range(iters):
            w, a = run_once(model, text, seed=i)
            walls.append(w); audios.append(a)
        avg_w = statistics.mean(walls)
        avg_a = statistics.mean(audios)
        print(f'{name:>7} | chars={len(text):>3} | wall={avg_w:.2f}s audio={avg_a:.2f}s RTF={avg_w/avg_a:.3f}', flush=True)


def bench_concurrent(model, text_name='medium', concurrencies=(1, 2, 4, 8), per_round=4):
    print(f'\n=== Concurrent (text={text_name}, per_round={per_round}) ===', flush=True)
    text = TEXTS[text_name]
    for conc in concurrencies:
        total = conc * per_round
        work_q = queue.Queue()
        for i in range(total):
            work_q.put(i)
        latencies, audios = [], []
        lock = threading.Lock()

        def worker():
            while True:
                try:
                    seed = work_q.get_nowait()
                except queue.Empty:
                    return
                w, a = run_once(model, text, seed=seed)
                with lock:
                    latencies.append(w); audios.append(a)

        t0 = time.time()
        threads = [threading.Thread(target=worker) for _ in range(conc)]
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.time() - t0

        if not latencies: continue
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        qps = total / wall
        rt = sum(audios) / wall
        print(f'conc={conc} n={total} | QPS={qps:.2f} audio_thru={rt:.2f}x | lat avg={statistics.mean(latencies):.2f}s p50={p50:.2f}s p95={p95:.2f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trt', action='store_true')
    ap.add_argument('--no-vllm', action='store_true')
    ap.add_argument('--concurrent-only', action='store_true')
    args = ap.parse_args()

    use_vllm = not args.no_vllm
    use_trt = args.trt

    if use_vllm:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    from cosyvoice.cli.cosyvoice import AutoModel

    print(f'Config: vllm={use_vllm} trt={use_trt}', flush=True)
    print('Loading...', flush=True)
    t0 = time.time()
    model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
                      load_trt=use_trt, load_vllm=use_vllm, fp16=False)
    print(f'Loaded in {time.time()-t0:.2f}s', flush=True)

    if not args.concurrent_only:
        bench_sequential(model, iters=5)
    bench_concurrent(model, text_name='medium', concurrencies=(1, 2, 4, 8), per_round=4)


if __name__ == '__main__':
    main()
