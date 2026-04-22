# CosyVoice3 + vLLM + TRT on RTX 3090 — SLO-based capacity

**Test setup**: WSL2 Ubuntu 22.04, RTX 3090 24GB, CosyVoice3 via vLLM 0.11.0 + TRT engine, no model lock, FastAPI `/tts/stream` endpoint (raw int16 PCM). Test date: 2026-04-22.

## Raw sweep data

### SHORT text (9-10 chars, ~1.6-2.0s audio / request)

| conc | QPS | audio×rt | TTFA p50 | TTFA p95 | total p50 | total p95 | errors |
|-----:|----:|---------:|---------:|---------:|----------:|----------:|-------:|
| 1 | 0.89 | 1.60x | 1170ms | 1376ms | 1.17s | 1.38s | 0 |
| 2 | 1.33 | 2.67x | 1376ms | 2007ms | 1.38s | 2.01s | 0 |
| **4** | **2.68** | **5.66x** | 1772ms | 2032ms | 1.78s | 2.03s | 0 |
| 8 | 2.71 | 5.27x | 2942ms | 3759ms | 2.97s | 3.76s | 0 |
| 16 | 3.14 | 5.86x | 4772ms | 6928ms | 4.78s | 6.95s | 0 |
| 32 | 2.93 | 5.67x | 9841ms | 15332ms | 9.93s | 15.35s | 0 |

### MEDIUM text (38-47 chars, ~3-8s audio / request)

| conc | QPS | audio×rt | TTFA p50 | TTFA p95 | total p50 | total p95 | errors |
|-----:|----:|---------:|---------:|---------:|----------:|----------:|-------:|
| 1 | 0.37 | 3.39x | 1678ms | 1688ms | 2.67s | 2.78s | 0 |
| **2** | **0.67** | **6.04x** | 2144ms | 2456ms | 3.08s | 4.08s | 0 |
| 4 | 0.69 | 6.09x | 3183ms | 4976ms | 5.38s | 9.17s | 0 |
| 8 | 0.80 | 7.05x | 4846ms | 5747ms | 8.90s | 13.23s | 0 |
| 16 | 0.77 | 6.76x | 9344ms | 11778ms | 18.93s | 29.06s | 0 |
| 32 | 0.81 | 7.15x | 18421ms | 21545ms | 37.78s | 56.13s | 0 |

## Knee-point analysis (where latency starts dominating)

Per Little's Law, in a stable system: `concurrency = latency × throughput`. The knee is where pushing more concurrency only grows latency without adding throughput.

- **SHORT**: knee at conc=4 (QPS 2.68). Going to 8 = +1% QPS, +67% TTFA p50. Going to 16 = +17% QPS but +160% TTFA p50. Efficiency dead.
- **MEDIUM**: knee at conc=2 (QPS 0.67). QPS barely climbs past that; latency grows linearly.

## SLO-bound maximum achievable QPS

Real production services pick a TTFA target + total-latency target. Here's what's achievable on one RTX 3090 with CosyVoice3+vLLM+TRT:

| SLO (TTFA p95 / total p95) | Short text QPS | Medium text QPS | Use case |
|---|---:|---:|---|
| TTFA ≤ 300ms, total ≤ 1s | **0** | **0** | Real-time voice agent ❌ not feasible |
| TTFA ≤ 1.5s, total ≤ 3s | ~0.9 (conc=1) | 0 | Voice assistant batching ⚠️ |
| TTFA ≤ 2.5s, total ≤ 4s | ~2.7 (conc=4) | ~0.7 (conc=2) | Near-realtime narration ✅ |
| TTFA ≤ 5s, total ≤ 10s | ~2.7 (conc=8) | ~0.8 (conc=8) | Batch/podcast gen ✅ |
| No SLO, max throughput | ~3.1 (conc=16) | ~0.8 (conc=32) | Offline batch ⚠️ very long tail |

## What the data says about 200 QPS

Absolute maximum observed on single 3090: **3.14 QPS** (short text, conc=16, TTFA p95=6.9s).

To reach 200 QPS with comparable SLO, need:
- **~64× short text throughput** → 64 GPUs, or a 64x faster model
- **~74× medium text throughput** → even more

Unchanged conclusion: **single RTX 3090 cannot reach 200 QPS with CosyVoice3** under any SLO that allows < 10s latency.

## Cost lens (¥ per audio hour)

Single 3090 peak audio throughput (streaming, short text, conc=16): 5.86x realtime.
- 1 GPU·hour produces ~5.86 audio·hours of short-text content
- At ¥1.5/GPU·hour: **~¥0.26 / audio·hour**

Compare:
- Aliyun Qwen3-TTS API: ~¥1-2/万字符 ≈ ¥2-5/audio·hour (depends on text density)
- Self-hosted CosyVoice3 breaks even at ~3-5 audio·hours/day; beyond that self-host is cheaper

## Optimization rounds (2026-04-23)

Apples-to-apples short-text (~9-10 chars, ~1.6s audio/req) on the same WSL+3090
+ FE-cache + lock-free server. `n=4` per concurrency, so conc=1 p95 includes a
cold-start outlier — focus on p50.

| Round | Change | conc=1 TTFA p50 | conc=4 TTFA p50 | conc=4 TTFA p95 | conc=4 QPS | conc=4 lat p95 |
|---|---|---:|---:|---:|---:|---:|
| 0 (baseline) | TRT fp32, FE-cache, lock-free | 588 ms | 1141 ms | 2067 ms | 3.39 | 2.09 s |
| **1** | **+ Flow TRT fp16** | **559 ms** (−5%) | **997 ms** (−13%) | **1210 ms** (−41%) | **3.58** (+6%) | **1.21 s** (−42%) |
| **2** | **+ vLLM `gpu_mem=0.6` + chunked-prefill + `max_num_seqs=64`** | **525 ms** (−11%) | 1137 ms (noise) | 1605 ms (−22%) | 3.33 (noise) | 1.61 s (−23%) |
| **3** | **+ Single-thread vllm.step scheduler (lock removed)** | **520 ms** (−12%) | 1115 ms | 1825 ms (−12%) | 3.41 | 1.83 s |

Round 2 wins are at higher concurrency where the larger KV-cache budget lets
vLLM batch more aggressively (low conc was already saturated):

| conc | Round 0 QPS | Round 2 QPS | Round 0 TTFA p50 | Round 2 TTFA p50 | Round 2 audio thru |
|---:|---:|---:|---:|---:|---:|
| 8 | 2.71 | **4.44** (+64%) | 2942 ms | 1787 ms (−39%) | 8.4× |
| 16 | 3.14 | **5.33** (+70%) | 4772 ms | 2973 ms (−38%) | **10.04×** |

Round 3 collapses peak concurrency from 16 → 8 by removing per-thread
`vllm.step()` lock contention (Single-thread vllm scheduler dispatches
tokens to per-uuid queues; clients block on `queue.get()` instead of
holding a global lock + sleep-polling):

| conc | Round 2 QPS | Round 3 QPS | Round 2 TTFA p50 | Round 3 TTFA p50 |
|---:|---:|---:|---:|---:|
| 8 | 4.44 | **5.81** (+31%) | 1787 ms | **1431 ms** (−20%) |
| 16 | **5.33** | 4.54 (−15%) | 2973 ms | 3382 ms (+14%) |
| 32 | n/a | 4.93 | n/a | 6059 ms |

Round 3 peak is **5.81 QPS at TTFA 1.4 s** (conc=8) vs Round 2's
**5.33 QPS at TTFA 3.0 s** (conc=16) — same throughput, half the
latency, half the queue depth. The conc=16 regression is the new
GIL-bound bottleneck: dispatching tokens from the scheduler to many
waiting `queue.put()` calls per `vllm.step()` saturates the GIL.

Effective production capacity (TTFA ≤ 1.5 s SLO):

| Round | Best conc | QPS | TTFA p50 |
|---:|---:|---:|---:|
| 0 | 4 | 2.68 | 1772 ms (over SLO) |
| 1 | 4 | 3.58 |  997 ms |
| 2 | 4 | 3.33 | 1137 ms |
| 3 | 8 | **5.81** | **1431 ms** |

Notes:
- `enable_prefix_caching=True` was silently ignored — vLLM V1 doesn't support
  it together with `enable_prompt_embeds`, so it falls back to off. Kept the
  flag for future vLLM versions.
- `max_num_seqs=64` was important: with the 0.2→0.6 mem-util bump, vLLM would
  otherwise default to ~256 seqs and reserve KV cache for them upfront, eating
  most of the headroom. 64 is enough for our concurrent-stream pattern.

Round 1 wins where it matters most for production: TTFA p95 and tail latency
collapse (−41% / −42%) because the fp16 Flow engine finishes per-request 30%
faster, draining the per-token-decode queue before a second request can pile up.
p50 gain is more modest because it was already dominated by FE/LLM-prefill
floor (~500 ms), not Flow.

Audio samples: `samples/round0_baseline/` vs `samples/round1_fp16/` — same
prompts/seeds. Long-text (~120 chars) stability checked, no degradation.

The upstream warning (`DiT tensorRT fp16 engine have some performance issue`)
did not manifest as user-perceptible artifacts in our test set.

## Key takeaways

1. **TTFA is dominated by vLLM prefill + first flow-matching batch**, not by GPU throughput. You cannot tune your way past ~1.2s TTFA on a single 3090 for CosyVoice3.
2. **Throughput saturates early** (conc=4 for short, conc=2 for medium) because the pipeline is "thick" — one request already keeps the GPU warm.
3. **Linear TTFA growth with concurrency** is a queueing effect. vLLM batches at the LLM step, but the *decode* phase isn't fully parallelized in CosyVoice's path.
4. **Streaming vs sync**: streaming trades ~20% throughput (5.9x vs 13.6x from the non-streaming bench) for the ability to report TTFA. Worth it for interactive use cases, skip for pure batch.
