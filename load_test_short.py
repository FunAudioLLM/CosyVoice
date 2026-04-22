"""Same as load_test.py but only short Chinese sentences (~2-3s audio each)."""
import time, argparse, statistics, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

# Short prompts, all under 15 chars / 2-3s audio
SHORT_TEXTS = [
    '你好，我能帮你吗？',
    '今天天气真不错。',
    '欢迎使用我们的服务。',
    '请问有什么需要？',
    '谢谢您的反馈。',
    '请稍等片刻。',
]


def one_request(url, idx):
    text = SHORT_TEXTS[idx % len(SHORT_TEXTS)]
    body = json.dumps({'text': text, 'seed': idx}).encode('utf-8')
    req = urllib.request.Request(url, data=body,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        audio = resp.read()
        h = dict(resp.headers)
        return {
            'idx': idx,
            'wall': time.time() - t0,
            'audio_seconds': float(h.get('x-audio-seconds', 0)),
            'server_wall': float(h.get('x-wall-seconds', 0)),
            'bytes': len(audio),
        }


def run(url, concurrency, total):
    print(f'concurrency={concurrency} total_requests={total}', flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, url, i) for i in range(total)]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f'request failed: {e}', flush=True)
    elapsed = time.time() - t0
    if not results:
        print('  ALL FAILED'); return
    walls = sorted([r['wall'] for r in results])
    audios = [r['audio_seconds'] for r in results]
    qps = len(results) / elapsed
    rt = sum(audios) / elapsed
    p50 = walls[len(walls) // 2]
    p95 = walls[int(len(walls) * 0.95)] if len(walls) > 1 else walls[0]
    print(f'  wall={elapsed:.2f}s | QPS={qps:.2f} | audio_throughput={rt:.2f}x realtime', flush=True)
    print(f'  client latency: avg={statistics.mean(walls):.2f}s p50={p50:.2f}s p95={p95:.2f}s max={max(walls):.2f}s', flush=True)
    print(f'  avg audio per req: {statistics.mean(audios):.2f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000/tts')
    p.add_argument('--concurrency', type=int, nargs='+', default=[8, 16, 32, 64])
    p.add_argument('--per-round', type=int, default=4)
    args = p.parse_args()
    for c in args.concurrency:
        run(args.url, concurrency=c, total=c * args.per_round)
