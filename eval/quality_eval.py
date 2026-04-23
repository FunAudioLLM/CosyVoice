"""Automated audio quality evaluation across CosyVoice optimization rounds.

Computes the following metrics on each WAV in samples/round*/:

  Whisper WER     -- intelligibility regression detector (catches fp16 NaN
                     pronouncing wrong, quantization artifacts collapsing
                     phonemes). Compares Whisper transcript to the reference
                     text the sample was generated from.
  SECS            -- speaker similarity to the prompt audio
                     (asset/zero_shot_prompt.wav). Uses ECAPA-TDNN. Cosine
                     similarity in [-1, 1]; >=0.90 is "kept the voice",
                     <0.85 = regression.
  RMS energy      -- gross sanity (zeroed-out / clipping detection).
  Duration        -- catches truncation regressions.

Usage
-----
  python quality_eval.py --samples-root samples --reference-prompt asset/zero_shot_prompt.wav

Optional metrics (pass --with-dnsmos): DNSMOS perceptual quality (1-5).
Slow on first run (downloads weights). Skipped by default.

Dependencies (install in venv):
  pip install openai-whisper speechbrain torchaudio
  # optional: pip install dnsmos
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
from torch.nn.functional import cosine_similarity

# Map sample filename prefix -> reference text the sample was generated from.
# Matches the prompts used in samples/round*/ generation (curl loops in commits).
REFERENCE_TEXTS = {
    '你好欢迎': '你好欢迎',
    '阿里云Cos': '阿里云CosyVoice三号是当前开源里最先进的多语言语音合成系统之一',
    'long': '昨天我去图书馆借了三本关于人工智能的书，发现现代深度学习模型的发展速度真的非常惊人。'
            '短短几年时间，从GPT-2到GPT-4，再到现在的多模态大模型，每一代都有质的飞跃。'
            '我相信未来十年内，人工智能将会彻底改变我们的工作和生活方式。',
}


def find_reference_text(filename: str) -> str | None:
    """Match a wav filename to the text it was generated from."""
    stem = Path(filename).stem
    for prefix, text in REFERENCE_TEXTS.items():
        if stem.startswith(prefix):
            return text
    return None


def load_wav(path: Path, target_sr: int):
    # Use soundfile to dodge torchaudio>=2.11's torchcodec dependency.
    import soundfile as sf
    data, sr = sf.read(str(path), always_2d=True)  # (T, C) float64
    wav = torch.from_numpy(data.T).float()  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def normalize_text(s: str) -> str:
    """Strip whitespace + punctuation + lowercase -- crude CER pre-processing
    so Whisper transcripts compare fairly against the prompts."""
    import re
    s = re.sub(r'[，。！？、；：""''""()（）【】《》\s\.,!?;:\'\"\-\[\]<>]', '', s)
    return s.lower().strip()


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    r = list(normalize_text(ref))
    h = list(normalize_text(hyp))
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def init_whisper(device: str):
    import whisper
    print(f'[whisper] loading "base" model on {device} ...', flush=True)
    return whisper.load_model('base', device=device)


def init_secs(device: str):
    from speechbrain.inference.speaker import EncoderClassifier
    print(f'[secs] loading speechbrain ECAPA-TDNN on {device} ...', flush=True)
    return EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        run_opts={'device': device},
        savedir='/tmp/spkrec-ecapa-voxceleb',
    )


def whisper_transcribe(model, wav_path: Path) -> str:
    # Whisper handles its own resampling; pass the file path
    result = model.transcribe(str(wav_path), language='zh', fp16=torch.cuda.is_available())
    return result['text']


def secs_against_ref(secs_model, sample_path: Path, ref_emb):
    wav, _ = load_wav(sample_path, 16000)
    emb = secs_model.encode_batch(wav)
    return cosine_similarity(ref_emb.flatten().unsqueeze(0), emb.flatten().unsqueeze(0)).item()


def rms_db(wav_path: Path) -> float:
    wav, _ = load_wav(wav_path, 16000)
    rms = wav.pow(2).mean().sqrt().item()
    return 20 * torch.log10(torch.tensor(max(rms, 1e-10))).item()


def duration_s(wav_path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def evaluate_round(round_dir: Path, whisper_model, secs_model, ref_emb, with_dnsmos: bool):
    """Return dict with per-sample scores + per-round aggregates."""
    samples = sorted(round_dir.glob('*.wav'))
    rows = []
    for w in samples:
        ref_text = find_reference_text(w.name)
        cer_score = None
        transcript = ''
        if ref_text and whisper_model is not None:
            try:
                transcript = whisper_transcribe(whisper_model, w)
                cer_score = cer(ref_text, transcript)
            except Exception as e:
                print(f'  [whisper-fail] {w.name}: {e}', flush=True)
        secs_score = None
        if secs_model is not None and ref_emb is not None:
            try:
                secs_score = secs_against_ref(secs_model, w, ref_emb)
            except Exception as e:
                print(f'  [secs-fail] {w.name}: {e}', flush=True)
        rows.append({
            'file': w.name,
            'duration_s': round(duration_s(w), 3),
            'rms_db': round(rms_db(w), 2),
            'whisper_cer': round(cer_score, 4) if cer_score is not None else None,
            'whisper_text': transcript,
            'secs': round(secs_score, 4) if secs_score is not None else None,
        })
    if not rows:
        return None

    def _avg(key):
        vs = [r[key] for r in rows if r[key] is not None]
        return round(sum(vs) / len(vs), 4) if vs else None

    return {
        'round': round_dir.name,
        'n_samples': len(rows),
        'avg_cer': _avg('whisper_cer'),
        'avg_secs': _avg('secs'),
        'avg_rms_db': _avg('rms_db'),
        'avg_dur_s': _avg('duration_s'),
        'samples': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-root', default='samples')
    ap.add_argument('--reference-prompt', default='asset/zero_shot_prompt.wav')
    ap.add_argument('--out-json', default='eval/quality_report.json')
    ap.add_argument('--out-md', default='eval/quality_report.md')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--skip-whisper', action='store_true')
    ap.add_argument('--skip-secs', action='store_true')
    ap.add_argument('--with-dnsmos', action='store_true', help='(not yet wired)')
    args = ap.parse_args()

    samples_root = Path(args.samples_root)
    rounds = sorted(p for p in samples_root.iterdir() if p.is_dir())
    if not rounds:
        print(f'no rounds found under {samples_root}', file=sys.stderr)
        sys.exit(1)
    print(f'found {len(rounds)} rounds: {[r.name for r in rounds]}', flush=True)

    whisper_model = None if args.skip_whisper else init_whisper(args.device)
    secs_model = None
    ref_emb = None
    if not args.skip_secs:
        secs_model = init_secs(args.device)
        ref_wav, _ = load_wav(Path(args.reference_prompt), 16000)
        ref_emb = secs_model.encode_batch(ref_wav)
        print(f'[secs] reference embedding ready (shape={tuple(ref_emb.shape)})', flush=True)

    results = []
    for r in rounds:
        print(f'\n=== {r.name} ===', flush=True)
        out = evaluate_round(r, whisper_model, secs_model, ref_emb, args.with_dnsmos)
        if out:
            results.append(out)
            print(f'  avg CER={out["avg_cer"]} SECS={out["avg_secs"]} RMS_dB={out["avg_rms_db"]}', flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nwrote {args.out_json}', flush=True)

    # Markdown summary
    lines = ['# Audio quality across optimization rounds', '',
             '| round | n | avg CER | avg SECS | avg RMS dB | avg dur s | flag |',
             '|---|---:|---:|---:|---:|---:|---|']
    base = results[0]
    for row in results:
        flag = ''
        if base['avg_cer'] is not None and row['avg_cer'] is not None:
            if row['avg_cer'] > base['avg_cer'] + 0.05:
                flag += ' INTELLIGIBILITY '
        if base['avg_secs'] is not None and row['avg_secs'] is not None:
            if row['avg_secs'] < base['avg_secs'] - 0.05:
                flag += ' VOICE '
        lines.append(f'| {row["round"]} | {row["n_samples"]} | {row["avg_cer"]} | '
                     f'{row["avg_secs"]} | {row["avg_rms_db"]} | {row["avg_dur_s"]} | {flag.strip() or "-"} |')
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out_md}', flush=True)


if __name__ == '__main__':
    main()
