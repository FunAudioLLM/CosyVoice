"""Same deep profile but with FE cache enabled."""
import os, sys, time, statistics
sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from fe_cache import enable_fe_cache
import profile_deep as PD


def main():
    fp16 = os.environ.get('FP16', '1') == '1'
    print(f'Loading CosyVoice3 (TRT + vLLM, fp16={fp16}) + FE cache ...', flush=True)
    t0 = time.time()
    model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=fp16)
    print(f'Loaded in {time.time()-t0:.2f}s', flush=True)

    # IMPORTANT: enable cache BEFORE patching for profile (since cache wraps frontend_zero_shot)
    enable_fe_cache(model)
    PD.patch(model)

    # warmup also primes the cache
    for s in (1000, 1001):
        PD.run_one(model, PD.TEXTS['medium'], seed=s, stream=False)
    PD._recs.clear()

    # --- Sync, sequential ---
    uuids = []
    for s in range(10, 16):
        u = PD.run_one(model, PD.TEXTS['medium'], seed=s, stream=False)
        if u: uuids.append(u)
    PD.print_table('SYNC medium x6 (cached prompt)', uuids)

    # --- Stream, sequential ---
    uuids = []
    for s in range(20, 26):
        u = PD.run_one(model, PD.TEXTS['medium'], seed=s, stream=True)
        if u: uuids.append(u)
    PD.print_table('STREAM medium x6 (cached prompt)', uuids)

    # --- Stream, short, sequential ---
    uuids = []
    for s in range(30, 36):
        u = PD.run_one(model, PD.TEXTS['short'], seed=s, stream=True)
        if u: uuids.append(u)
    PD.print_table('STREAM short x6 (cached prompt)', uuids)


if __name__ == '__main__':
    main()
