"""Deep stage breakdown of CosyVoice3 — every internal method instrumented.

Records, per request:
  Frontend stages:
    FE.tokenize_tts        - encode tts text
    FE.tokenize_prompt     - encode prompt text
    FE.speech_feat         - mel for flow (load wav 24k + mel)
    FE.speech_token        - speech_tokenizer_v3.onnx (load wav 16k + log_mel + ONNX)
    FE.spk_embedding       - campplus.onnx (load wav 16k + kaldi.fbank + ONNX)
  LLM stages:
    LLM.first_token_ms     - time from llm_job start to first token yielded
    LLM.per_token_ms       - mean time between subsequent tokens
    LLM.total_ms           - whole llm_job duration
    LLM.tokens_emitted     - count
  T2W stages (per chunk, summed if multi):
    T2W.flow_ms            - flow matching (TRT)
    T2W.hift_ms            - HiFi-GAN vocoder
    T2W.cuda_sync_ms       - explicit synchronize after
"""
import sys, time, statistics, threading
sys.path.append('third_party/Matcha-TTS')

import torch
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

# request-id (uuid) keyed records, since llm runs in its own thread but we have uuid
_recs = {}            # uuid -> dict
_recs_lock = threading.Lock()
_current_req = threading.local()


def _ensure(uuid_):
    with _recs_lock:
        return _recs.setdefault(uuid_, {})


def _rec(uuid_, key, val):
    d = _ensure(uuid_)
    d[key] = val


def _rec_add(uuid_, key, val):
    d = _ensure(uuid_)
    d[key] = d.get(key, 0) + val


def patch(model):
    fe = model.frontend
    m = model.model

    # ---- Frontend substages: time the *first* call per request via thread-local ----
    orig_etx = fe._extract_text_token
    orig_esp_feat = fe._extract_speech_feat
    orig_esp_tok = fe._extract_speech_token
    orig_spk_emb = fe._extract_spk_embedding

    def w_etx(text):
        t0 = time.perf_counter(); r = orig_etx(text)
        # first call = tts text, second = prompt text
        u = getattr(_current_req, 'uuid', None)
        if u:
            d = _ensure(u)
            key = 'FE.tokenize_tts' if 'FE.tokenize_tts' not in d else 'FE.tokenize_prompt'
            d[key] = (time.perf_counter() - t0) * 1000
        return r

    def w_esp_feat(wav):
        t0 = time.perf_counter(); r = orig_esp_feat(wav)
        u = getattr(_current_req, 'uuid', None)
        if u: _rec(u, 'FE.speech_feat', (time.perf_counter() - t0) * 1000)
        return r

    def w_esp_tok(wav):
        t0 = time.perf_counter(); r = orig_esp_tok(wav)
        u = getattr(_current_req, 'uuid', None)
        if u: _rec(u, 'FE.speech_token', (time.perf_counter() - t0) * 1000)
        return r

    def w_spk_emb(wav):
        t0 = time.perf_counter(); r = orig_spk_emb(wav)
        u = getattr(_current_req, 'uuid', None)
        if u: _rec(u, 'FE.spk_embedding', (time.perf_counter() - t0) * 1000)
        return r

    fe._extract_text_token = w_etx
    fe._extract_speech_feat = w_esp_feat
    fe._extract_speech_token = w_esp_tok
    fe._extract_spk_embedding = w_spk_emb

    # ---- LLM: wrap llm_job to time first-token vs total ----
    orig_llm_job = m.llm_job

    def w_llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        # Count what gets appended to tts_speech_token_dict[uuid] over time
        d = _ensure(uuid)
        before_len = 0
        first_token_at = None
        t0 = time.perf_counter()
        # Run original; we observe the dict as it grows via sampling thread
        stop_event = threading.Event()
        last_count = [0]
        first_t = [None]

        def watcher():
            while not stop_event.is_set():
                cur = len(m.tts_speech_token_dict.get(uuid, []))
                if first_t[0] is None and cur > 0:
                    first_t[0] = time.perf_counter()
                last_count[0] = cur
                time.sleep(0.005)  # 5ms sampling

        watch_thread = threading.Thread(target=watcher, daemon=True)
        watch_thread.start()
        try:
            r = orig_llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid)
        finally:
            stop_event.set()
            watch_thread.join(timeout=0.1)
        t_end = time.perf_counter()
        total_ms = (t_end - t0) * 1000
        n_tokens = last_count[0]
        first_ms = ((first_t[0] - t0) * 1000) if first_t[0] else None
        d['LLM.total_ms'] = total_ms
        d['LLM.tokens'] = n_tokens
        d['LLM.first_token_ms'] = first_ms
        if n_tokens > 1 and first_t[0] is not None:
            d['LLM.per_token_ms'] = ((t_end - first_t[0]) * 1000) / max(n_tokens - 1, 1)
        return r

    m.llm_job = w_llm_job

    # ---- T2W: split flow vs hift ----
    orig_flow_inf = m.flow.inference
    orig_hift_inf = m.hift.inference

    def w_flow(*a, **kw):
        t0 = time.perf_counter(); r = orig_flow_inf(*a, **kw)
        if torch.cuda.is_available(): torch.cuda.current_stream().synchronize()
        u = getattr(_current_req, 'uuid', None)
        if u: _rec_add(u, 'T2W.flow_ms', (time.perf_counter() - t0) * 1000)
        return r

    def w_hift(*a, **kw):
        t0 = time.perf_counter(); r = orig_hift_inf(*a, **kw)
        if torch.cuda.is_available(): torch.cuda.current_stream().synchronize()
        u = getattr(_current_req, 'uuid', None)
        if u: _rec_add(u, 'T2W.hift_ms', (time.perf_counter() - t0) * 1000)
        return r

    m.flow.inference = w_flow
    m.hift.inference = w_hift

    # Wrap model.tts to set the current uuid for the request thread
    orig_tts = m.tts

    def w_tts(*a, **kw):
        # Generate uuid here (matches what tts() does internally); then the inner
        # tts() will create its own. We can't easily inject. Instead, set a
        # thread-local that the patched submethods use.
        # Better: extract uuid by intercepting the call.
        import uuid as uuid_mod
        # We can't easily pre-create uuid since tts() generates its own.
        # Workaround: clear thread-local uuid, then sniff via hift_cache_dict creation.
        _current_req.uuid = None
        gen = orig_tts(*a, **kw)
        for chunk in gen:
            # by now, tts() has populated some dicts with this uuid
            # find the latest uuid known to model
            with m.lock:
                if m.tts_speech_token_dict:
                    # latest is fine for our use
                    _current_req.uuid = list(m.tts_speech_token_dict.keys())[-1]
            yield chunk

    m.tts = w_tts


def run_one(model, text, seed, stream=False):
    set_all_random_seed(seed)
    # Pre-set thread-local uuid will be set inside w_tts
    t0 = time.perf_counter()
    chunks = 0
    audio_sec = 0.0
    t_first = None
    for j in model.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=stream):
        if t_first is None:
            t_first = time.perf_counter()
        chunks += 1
        audio_sec += j['tts_speech'].shape[-1] / model.sample_rate
    t_end = time.perf_counter()
    u = getattr(_current_req, 'uuid', None)
    if u:
        d = _ensure(u)
        d['_TTFA_ms'] = ((t_first - t0) * 1000) if t_first else None
        d['_TOTAL_ms'] = (t_end - t0) * 1000
        d['_CHUNKS'] = chunks
        d['_AUDIO_S'] = audio_sec
    return u


def aggregate(uuids):
    by_key = {}
    for u in uuids:
        d = _recs.get(u, {})
        for k, v in d.items():
            if v is None: continue
            by_key.setdefault(k, []).append(v)
    return by_key


def print_table(label, uuids):
    bk = aggregate(uuids)
    print(f'\n=== {label} (n={len(uuids)}) ===')
    keys_order = [
        'FE.tokenize_tts', 'FE.tokenize_prompt',
        'FE.speech_feat', 'FE.speech_token', 'FE.spk_embedding',
        'LLM.first_token_ms', 'LLM.per_token_ms', 'LLM.tokens', 'LLM.total_ms',
        'T2W.flow_ms', 'T2W.hift_ms',
        '_TTFA_ms', '_TOTAL_ms', '_CHUNKS', '_AUDIO_S',
    ]
    for k in keys_order:
        if k not in bk: continue
        vs = bk[k]
        if k in ('LLM.tokens', '_CHUNKS'):
            print(f'  {k:>22} | avg={statistics.mean(vs):8.1f}  min={min(vs):.0f} max={max(vs):.0f}  n={len(vs)}')
        elif k == '_AUDIO_S':
            print(f'  {k:>22} | avg={statistics.mean(vs):8.2f}s n={len(vs)}')
        else:
            srt = sorted(vs)
            p50 = srt[len(srt) // 2]
            p95 = srt[int(len(srt) * 0.95)] if len(srt) > 1 else srt[0]
            print(f'  {k:>22} | avg={statistics.mean(vs):7.1f}ms p50={p50:7.1f}ms p95={p95:7.1f}ms n={len(vs)}')


def main():
    print('Loading CosyVoice3 (TRT + vLLM, fp32) ...', flush=True)
    t0 = time.time()
    model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    print(f'Loaded in {time.time()-t0:.2f}s', flush=True)
    patch(model)

    # warmup
    for s in (1000, 1001):
        run_one(model, TEXTS['medium'], seed=s, stream=False)
    _recs.clear()

    # --- Sync, sequential ---
    uuids = []
    for s in range(10, 16):
        u = run_one(model, TEXTS['medium'], seed=s, stream=False)
        if u: uuids.append(u)
    print_table('SYNC medium x6', uuids)

    # --- Stream, sequential ---
    uuids = []
    for s in range(20, 26):
        u = run_one(model, TEXTS['medium'], seed=s, stream=True)
        if u: uuids.append(u)
    print_table('STREAM medium x6', uuids)

    # --- Stream, short, sequential ---
    uuids = []
    for s in range(30, 36):
        u = run_one(model, TEXTS['short'], seed=s, stream=True)
        if u: uuids.append(u)
    print_table('STREAM short x6', uuids)


if __name__ == '__main__':
    main()
