import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pyloudnorm as pyln
from loguru import logger
import torchaudio

# 设置 PYTHONPATH 环境变量
os.environ["PYTHONPATH"] = "third_party/Matcha-TTS"
sys.path.append("third_party/Matcha-TTS")


from tools.auto_task_help import (
    get_texts,
    has_omission,
    get_texts_with_line,
)
from cosyvoice.cli.cosyvoice import CosyVoice

# 抑制 pyloudnorm 中的 UserWarning 警告
logger.info(f"CUDA available: {torch.cuda.is_available()}")

# 初始化 CosyVoice 实例
cosyvoice = CosyVoice("./pretrained_models/CosyVoice-300M-SFT")
sft_spk = [
    # "赛诺",
    # "卡维",
    # "旁白1",
    # "凝光",
    # "枫原万叶",
]

spk_idx = 1031
spk = ""

rate = 22050


def _norm_loudness(audio, rate):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if audio.ndim == 2:
        audio = audio.squeeze()

    block_size = 0.4 * rate
    if len(audio) < block_size:
        return torch.from_numpy(audio)

    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, loudness, -16.0)
    return torch.from_numpy(normalized_audio)


def prepare_audio(audio):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    return _norm_loudness(audio, rate)


def process_text(text, spk):
    global spk_idx
    texts = get_texts_with_line(text)
    for line in texts:
        if line == "":
            continue
        logger.info(f"正在处理文本：{line}")
        try_again = 0

        while try_again < 3:
            output = cosyvoice.inference_sft(line, spk)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            is_continu, pinyin_similarity, gen_text_clean, text_clean = has_omission(
                output["tts_speech"], line
            )
            logger.info(f"""  
=========  
原始文本：{line}  
try_again: {try_again}  
speaker: {spk}  
生成文本：{gen_text_clean}  
输入文本：{text_clean}  
相似度：{pinyin_similarity}  
spk_idx: {spk_idx}
=========  
            """)

            if is_continu:
                try_again += 1
            else:
                input_audio = prepare_audio(output["tts_speech"]).squeeze(0)

                tmp_name = f"{spk}_{spk_idx}"
                temp_wav_path = f"./tmp/gen/{spk}/{tmp_name}.wav"
                os.makedirs(os.path.dirname(temp_wav_path), exist_ok=True)
                torchaudio.save(temp_wav_path, input_audio.unsqueeze(0), rate)
                spk_idx += 1

                with open(
                    f"./tmp/gen/{spk}/{tmp_name}.normalized.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(line)

                break


def generate_silence(line):
    random_duration = (
        np.random.uniform(0.05, 0.08)
        if line[-1] == "，"
        else np.random.uniform(0.09, 0.13)
    )
    return torch.zeros(int(rate * random_duration), dtype=torch.float32).unsqueeze(0)


def process_chapter(book_name, idx, spk):
    file_path = f"./tmp/{book_name}/data/chapter_{idx}.txt"
    if not os.path.exists(file_path):
        logger.info(f"文件 {file_path} 不存在，跳过处理。")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        title = file.readline().strip()

    texts = get_texts(open(file_path, "r", encoding="utf-8").read(), True)

    with tqdm(
        total=len(texts), desc=f"正在处理：{book_name}-{idx} - {title}", leave=True
    ) as pbar:
        for line in texts:
            if line == "":
                pbar.update(1)
                continue

            process_text(line, spk)
            pbar.update(1)


def main():
    global spk_idx
    parser = argparse.ArgumentParser(
        description="Process book chapters to generate audio files."
    )
    parser.add_argument(
        "--book_name", type=str, help="Name of the book", default="dqrsztxml"
    )
    parser.add_argument(
        "--start_idx", type=int, help="Starting chapter index", default=61
    )
    parser.add_argument(
        "--end_idx", type=int, help="Ending chapter index", default=1000
    )

    args = parser.parse_args()
    os.makedirs(f"./tmp/{args.book_name}/gen", exist_ok=True)

    idx = 753
    for spk in sft_spk:
        while True:
            process_chapter(args.book_name, idx, spk)
            idx = idx + 1
            if spk_idx >= 3000:
                spk_idx = 0
                break


if __name__ == "__main__":
    main()
