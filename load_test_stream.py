"""Streaming TTS load test that measures TTFA (Time To First Audio chunk).

Uses raw socket-style HTTP client (http.client) so we control when each byte
is consumed. TTFA = wall time from request send to first non-empty body chunk.
"""
import time, argparse, statistics, json, http.client
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

SHORT_TEXTS = [
    '你好，我能帮你吗？',
    '今天天气真不错。',
    '欢迎使用我们的服务。',
    '请问有什么需要？',
    '谢谢您的反馈。',
]
MEDIUM_TEXTS = [
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
    '阿里云 CosyVoice 三号是当前开源里最先进的多语言语音合成系统之一，效果非常自然流畅。',
    '今天我们来测试这个服务在高并发场景下的延迟和吞吐表现，看看实际生产能力如何。',
]


def one_request(host, port, path, idx, texts):
    text = texts[idx % len(texts)]
    body = json.dumps({'text': text, 'seed': idx}).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Connection': 'close'}

    t_start = time.time()
    conn = http.client.HTTPConnection(host, port, timeout=180)
    conn.request('POST', path, body=body, headers=headers)
    resp = conn.getresponse()
    if resp.status != 200:
        conn.close()
        return {'idx': idx, 'error': f'HTTP {resp.status}'}

    sr = int(resp.headers.get('X-Sample-Rate', 24000))
    bytes_per_sample = 2  # int16

    # Read first chunk to capture TTFA
    first_chunk = resp.read(4096)
    if not first_chunk:
        conn.close()
        return {'idx': idx, 'error': 'empty stream'}
    t_first = time.time()

    total_bytes = len(first_chunk)
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        total_bytes += len(chunk)
    t_end = time.time()
    conn.close()

    audio_sec = total_bytes / bytes_per_sample / sr
    return {
        'idx': idx,
        'ttfa': t_first - t_start,
        'wall': t_end - t_start,
        'audio_seconds': audio_sec,
        'bytes': total_bytes,
    }


def run(url, concurrency, total, texts, label):
    parsed = urlparse(url)
    host, port, path = parsed.hostname, parsed.port or 80, parsed.path

    print(f'[{label}] conc={concurrency:>3} n={total:>3}', end='', flush=True)
    t0 = time.time()
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, host, port, path, i, texts) for i in range(total)]
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if 'error' in r:
                    errors.append(r['error'])
                else:
                    results.append(r)
            except Exception as e:
                errors.append(str(e))
    elapsed = time.time() - t0

    if not results:
        print(f' | ALL FAILED ({len(errors)} errors)')
        return None

    def pct(xs, p):
        xs = sorted(xs)
        return xs[int(len(xs) * p)] if len(xs) > 1 else xs[0]

    walls = [r['wall'] for r in results]
    ttfas = [r['ttfa'] for r in results]
    audios = [r['audio_seconds'] for r in results]
    qps = len(results) / elapsed
    rt = sum(audios) / elapsed

    out = {
        'label': label,
        'concurrency': concurrency,
        'requests_ok': len(results),
        'errors': len(errors),
        'wall_total_s': elapsed,
        'qps': qps,
        'audio_throughput_x': rt,
        'avg_audio_per_req_s': statistics.mean(audios),
        'ttfa_p50_ms': pct(ttfas, 0.50) * 1000,
        'ttfa_p95_ms': pct(ttfas, 0.95) * 1000,
        'lat_p50_s':   pct(walls, 0.50),
        'lat_p95_s':   pct(walls, 0.95),
    }
    print(f' | QPS={out["qps"]:.2f} thru={out["audio_throughput_x"]:.1f}x'
          f' | TTFA p50={out["ttfa_p50_ms"]:.0f}ms p95={out["ttfa_p95_ms"]:.0f}ms'
          f' | lat p50={out["lat_p50_s"]:.2f}s p95={out["lat_p95_s"]:.2f}s'
          f' | errors={out["errors"]}', flush=True)
    return out


def sweep(url, label, texts, concurrencies, per_round=4):
    rows = []
    for c in concurrencies:
        out = run(url, c, c * per_round, texts, label)
        if out: rows.append(out)
    return rows


def print_table(rows, title):
    print(f'\n=== {title} ===')
    print(f'{"conc":>5} | {"QPS":>6} | {"thru":>6} | {"ttfa50":>7} | {"ttfa95":>7} | {"lat50":>7} | {"lat95":>7} | {"err":>3}')
    for r in rows:
        print(f'{r["concurrency"]:>5} | {r["qps"]:>6.2f} | {r["audio_throughput_x"]:>5.2f}x '
              f'| {r["ttfa_p50_ms"]:>6.0f}ms | {r["ttfa_p95_ms"]:>6.0f}ms '
              f'| {r["lat_p50_s"]:>6.2f}s | {r["lat_p95_s"]:>6.2f}s '
              f'| {r["errors"]:>3}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000/tts/stream')
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--per-round', type=int, default=4)
    p.add_argument('--length', choices=['short', 'medium', 'both'], default='both')
    args = p.parse_args()

    short_rows = []
    medium_rows = []
    if args.length in ('short', 'both'):
        short_rows = sweep(args.url, 'short', SHORT_TEXTS, args.concurrency, args.per_round)
    if args.length in ('medium', 'both'):
        medium_rows = sweep(args.url, 'medium', MEDIUM_TEXTS, args.concurrency, args.per_round)

    if short_rows:  print_table(short_rows, f'SHORT ({len(SHORT_TEXTS[0])}-{max(len(t) for t in SHORT_TEXTS)} chars)')
    if medium_rows: print_table(medium_rows, f'MEDIUM ({min(len(t) for t in MEDIUM_TEXTS)}-{max(len(t) for t in MEDIUM_TEXTS)} chars)')
