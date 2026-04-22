"""Stage-by-stage profiling of CosyVoice3 inference.

Patches key methods to record per-call timings, runs N inferences, prints a
breakdown table.

Stages measured:
  TN     - text_normalize (frontend, CPU + small ONNX)
  FE     - frontend_zero_shot (audio prompt → mel + token, GPU/ONNX, runs once per req)
  LLM    - llm_job (vLLM generation thread, background)
  T2W*   - token2wav per chunk (flow matching + hift vocoder, on each yield)
  TTFA   - wall-clock from inference start to first yield
  TOTAL  - wall-clock from inference start to last yield
"""
import sys, time, statistics, threading
sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
PROMPT_WAV = './asset/zero_shot_prompt.wav'

TEXTS = {
    'short':  '你好，今天天气真不错。',
    'medium': '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
}

# global timing stash, threadlocal-ish via thread name
_per_request = {}  # thread_name -> dict of stage -> [durations_ms]
_per_request_lock = threading.Lock()


def _record(stage, dur_ms):
    name = threading.current_thread().name
    with _per_request_lock:
        d = _per_request.setdefault(name, {})
        d.setdefault(stage, []).append(dur_ms)


def patch(model):
    fe = model.frontend
    m = model.model

    orig_tn = fe.text_normalize
    orig_fzs = fe.frontend_zero_shot
    orig_llm = m.llm_job
    orig_t2w = m.token2wav

    def w_tn(text, *a, **kw):
        t0 = time.perf_counter(); r = orig_tn(text, *a, **kw)
        # text_normalize returns generator if split=True. Only time list materialization.
        if hasattr(r, '__iter__') and not isinstance(r, (str, list)):
            r = list(r)
        _record('TN', (time.perf_counter() - t0) * 1000)
        return r

    def w_fzs(*a, **kw):
        t0 = time.perf_counter(); r = orig_fzs(*a, **kw)
        _record('FE', (time.perf_counter() - t0) * 1000)
        return r

    def w_llm(*a, **kw):
        t0 = time.perf_counter(); r = orig_llm(*a, **kw)
        _record('LLM', (time.perf_counter() - t0) * 1000)
        return r

    def w_t2w(*a, **kw):
        t0 = time.perf_counter(); r = orig_t2w(*a, **kw)
        _record('T2W', (time.perf_counter() - t0) * 1000)
        return r

    fe.text_normalize = w_tn
    fe.frontend_zero_shot = w_fzs
    m.llm_job = w_llm
    m.token2wav = w_t2w


def run_one(model, text, seed, stream=False):
    threading.current_thread().name = f'req-{seed}-{int(time.time()*1000)%10000}'
    set_all_random_seed(seed)
    t_start = time.perf_counter()
    t_first = None
    audio_sec = 0.0
    chunks = 0
    for j in model.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=stream):
        if t_first is None:
            t_first = time.perf_counter()
        chunks += 1
        audio_sec += j['tts_speech'].shape[-1] / model.sample_rate
    t_end = time.perf_counter()
    name = threading.current_thread().name
    with _per_request_lock:
        d = _per_request.setdefault(name, {})
        d['TTFA'] = [(t_first - t_start) * 1000] if t_first else [0]
        d['TOTAL'] = [(t_end - t_start) * 1000]
        d['CHUNKS'] = chunks
        d['AUDIO_S'] = audio_sec
    return name


def summarize(req_names):
    """Aggregate per-stage stats across the given requests."""
    stage_totals = {}  # stage -> list of total_ms_per_request
    chunks_list = []
    audio_list = []
    for n in req_names:
        d = _per_request.get(n, {})
        chunks_list.append(d.get('CHUNKS', 0))
        audio_list.append(d.get('AUDIO_S', 0.0))
        for stage, durs in d.items():
            if stage in ('CHUNKS', 'AUDIO_S'): continue
            tot = sum(durs) if isinstance(durs, list) else durs
            stage_totals.setdefault(stage, []).append(tot)
    return stage_totals, chunks_list, audio_list


def fmt_row(name, vals):
    if not vals:
        return f'{name:>6} | n=0'
    avg = statistics.mean(vals)
    p50 = sorted(vals)[len(vals) // 2]
    p95 = sorted(vals)[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0]
    return f'{name:>6} | avg={avg:7.1f}ms p50={p50:7.1f}ms p95={p95:7.1f}ms n={len(vals)}'


def print_breakdown(label, req_names, expected_audio_per_req=None):
    stage_totals, chunks, audios = summarize(req_names)
    print(f'\n=== {label} ({len(req_names)} reqs) ===')
    if audios:
        print(f'  avg_audio_per_req={statistics.mean(audios):.2f}s avg_chunks={statistics.mean(chunks):.1f}')
    # known stages in order
    for s in ['TN', 'FE', 'LLM', 'T2W', 'TTFA', 'TOTAL']:
        if s in stage_totals:
            print('  ' + fmt_row(s, stage_totals[s]))


def main():
    print('Loading CosyVoice3 (TRT + vLLM, fp32) ...', flush=True)
    t0 = time.time()
    model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    print(f'Loaded in {time.time()-t0:.2f}s', flush=True)
    patch(model)

    # warmup
    print('Warming up...', flush=True)
    for s in (1000, 1001):
        run_one(model, TEXTS['medium'], seed=s, stream=False)
    _per_request.clear()

    # 1) Cold first request (sync, medium)
    n = run_one(model, TEXTS['medium'], seed=1, stream=False)
    print_breakdown('SYNC, medium, single request (post-warmup)', [n])

    # 2) Sequential runs (sync) for steady state
    seqs = []
    for s in range(10, 16):
        n = run_one(model, TEXTS['medium'], seed=s, stream=False)
        seqs.append(n)
    print_breakdown('SYNC, medium, sequential x6', seqs)

    # 3) Streaming single request
    n = run_one(model, TEXTS['medium'], seed=20, stream=True)
    print_breakdown('STREAM, medium, single request', [n])

    # 4) Streaming x6 sequential to see TTFA stability
    seqs = []
    for s in range(30, 36):
        n = run_one(model, TEXTS['medium'], seed=s, stream=True)
        seqs.append(n)
    print_breakdown('STREAM, medium, sequential x6', seqs)

    # 5) Concurrent stream conc=4 to see how stages overlap
    print('\n=== CONCURRENT stream, conc=4, n=8 ===')
    _per_request.clear()
    import queue, threading as th
    q = queue.Queue()
    for i in range(40, 48): q.put(i)
    names = []
    names_lock = th.Lock()

    def worker():
        while True:
            try: s = q.get_nowait()
            except queue.Empty: return
            n = run_one(model, TEXTS['medium'], seed=s, stream=True)
            with names_lock: names.append(n)

    t0 = time.time()
    threads = [th.Thread(target=worker) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    print_breakdown(f'STREAM, conc=4, total wall={wall:.2f}s', names)


if __name__ == '__main__':
    main()
