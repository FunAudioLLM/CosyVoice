"""Concurrent load test against the FastAPI TTS server."""
import time, argparse, statistics, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

TEXTS = [
    '你好，欢迎测试这个 TTS 接口的并发处理能力。',
    '阿里云 CosyVoice 三号模型是当前最先进的开源语音合成系统之一。',
    '今天我们要测试一下这个接口在高并发场景下能够处理多少请求。',
    '语音合成技术已经发展到了非常成熟的阶段，听起来自然流畅。',
    '人工智能正在改变我们的生活，从语音到图像，应用无处不在。',
]


def one_request(url, idx):
    text = TEXTS[idx % len(TEXTS)]
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
            'server_rtf': float(h.get('x-rtf', 0)),
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

    walls = [r['wall'] for r in results]
    audios = [r['audio_seconds'] for r in results]
    qps = len(results) / elapsed
    audio_total = sum(audios)
    rt_throughput = audio_total / elapsed
    walls.sort()
    p50 = walls[len(walls) // 2]
    p95 = walls[int(len(walls) * 0.95)] if len(walls) > 1 else walls[0]
    print(f'  wall={elapsed:.2f}s | QPS={qps:.2f} | audio_throughput={rt_throughput:.2f}x realtime', flush=True)
    print(f'  client latency: avg={statistics.mean(walls):.2f}s p50={p50:.2f}s p95={p95:.2f}s max={max(walls):.2f}s', flush=True)
    print(f'  audio generated: total={audio_total:.1f}s avg_per_req={statistics.mean(audios):.2f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000/tts')
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--per-round', type=int, default=8)
    args = p.parse_args()

    for c in args.concurrency:
        run(args.url, concurrency=c, total=c * args.per_round)
