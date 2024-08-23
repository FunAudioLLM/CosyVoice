import json
import os
import sys


# 设置 PYTHONPATH 环境变量
os.environ["PYTHONPATH"] = "third_party/Matcha-TTS"

# 将 PYTHONPATH 添加到 sys.path
sys.path.append("third_party/Matcha-TTS")


from tools.auto_task_help import (
    get_texts,
    has_omission,
    format_line,
    get_texts_with_line,
)
import numpy as np
import torch
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio
import os
import argparse
import pyloudnorm as pyln

import warnings
from loguru import logger


# 抑制 pyloudnorm 中的 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")
logger.info(f"CUDA available: {torch.cuda.is_available()}")


# 初始化 CosyVoice 实例
cosyvoice = CosyVoice("./pretrained_models/CosyVoice-300M-SFT")
sft_spk = cosyvoice.list_avaliable_spks()

# 用于存储相似度与音频的映射
pinyin_similarity_map = {}


def _norm_loudness(audio, rate):
    """
    标准化音频响度
    :param audio: 音频数据，可以是 PyTorch 张量或 NumPy 数组
    :param rate: 采样率
    :return: 标准化后的音频数据，PyTorch 张量
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if audio.ndim == 2:
        audio = audio.squeeze()
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, loudness, -16.0)
    return torch.from_numpy(normalized_audio)


def prepare_audio(audio):
    """
    准备音频数据
    :param audio: 音频数据
    :return: 标准化后的音频数据
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    return _norm_loudness(audio, 22050)


def process_text(text, wav_list, spk):
    """
    处理文本
    :param text: 文本
    :param wav_list: 音频列表
    :param spk: 角色
    :return: 处理后的文本
    """
    texts = get_texts_with_line(text)
    for text in texts:
        try_again = 0
        line = text

        while True:
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
=========
            """)
            pinyin_similarity_map[pinyin_similarity] = output["tts_speech"]

            if is_continu:
                try_again += 1
                if try_again >= 5:
                    best_similarity = max(pinyin_similarity_map.keys())
                    best_audio = pinyin_similarity_map[best_similarity]
                    pinyin_similarity_map.clear()
                    try_again = 0
                    best_audio = prepare_audio(best_audio)
                    wav_list.append(best_audio)
                    wav_list.append(generate_silence(line))
                    break
            else:
                pinyin_similarity_map.clear()
                try_again = 0
                tts_speech = prepare_audio(output["tts_speech"])
                wav_list.append(tts_speech)
                wav_list.append(generate_silence(line))
                break

        return wav_list


def process_text_line(index, texts, try_again, wav_list, spk):
    """
    处理单行文本
    :param index: 当前处理的文本行索引
    :param texts: 文本列表
    :param try_again: 重试次数
    :param wav_list: 音频列表
    :return: 更新后的索引和重试次数
    """
    line = format_line(texts[index])

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
生成文本：{gen_text_clean}
输入文本：{text_clean}
相似度：{pinyin_similarity}
=========
    """)

    pinyin_similarity_map[pinyin_similarity] = output["tts_speech"]

    if is_continu:
        try_again += 1
        if try_again >= 5:
            best_similarity = max(pinyin_similarity_map.keys())
            best_audio = pinyin_similarity_map[best_similarity]
            pinyin_similarity_map.clear()
            try_again = 0
            index += 1
            best_audio = prepare_audio(best_audio)
            wav_list.append(best_audio)
            wav_list.append(generate_silence(line))
    else:
        pinyin_similarity_map.clear()
        try_again = 0
        index += 1
        tts_speech = prepare_audio(output["tts_speech"])
        wav_list.append(tts_speech)
        wav_list.append(generate_silence(line))

    return index, try_again


def generate_silence(line):
    """
    生成停顿音频
    :param line: 当前处理的文本行
    :return: 停顿音频数据
    """
    random_duration = (
        np.random.uniform(0.05, 0.08)
        if line[-1] == "，"
        else np.random.uniform(0.09, 0.13)
    )
    zero_wav = np.zeros(int(22050 * random_duration), dtype=np.float32)
    return torch.from_numpy(zero_wav).unsqueeze(0)


def process_chapter(book_name, idx):
    """
    处理章节
    :param book_name: 书名
    :param idx: 章节索引
    """
    file_path = f"./tmp/{book_name}/data/chapter_{idx}.txt"
    if not os.path.exists(file_path):
        logger.info(f"文件 {file_path} 不存在，跳过处理。")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
        # 获取第一行，文章标题
        title = file_content.split("\n")[0]

    texts = get_texts(file_content)

    with open(f"./tmp/{book_name}/bookname2role.json", "r", encoding="utf-8") as f:
        bookname2role = json.load(f)

    with open(f"./tmp/{book_name}/role/chapter_{idx}.json", "r", encoding="utf-8") as f:
        line_map_list = json.load(f)
    line_map = {}
    for item in line_map_list:
        line_map.update(item)

    wav_list = []
    total_texts = len(texts)
    index = 0

    with tqdm(
        total=total_texts,
        desc=f"正在处理：{book_name}-{idx} - {title}",
        leave=True,  # 确保进度条在完成后不会被清除
    ) as pbar:
        while index < total_texts:
            line = texts[index]
            content_to_process = []

            if "“" in line and "”" in line:
                # 获取所有 ”“ 之间的内容并按照顺序处理
                segments = line.split("“")
                for segment in segments:
                    if "”" in segment:
                        quote, rest = segment.split("”", 1)
                        content_to_process.append((quote, True))
                        if rest.strip():
                            content_to_process.append((rest.strip(), False))
                    else:
                        if segment.strip():
                            content_to_process.append((segment.strip(), False))
            else:
                content_to_process.append((line.strip(), False))

            for content, is_quote in content_to_process:
                skp = "旁白1"
                if is_quote:
                    bookname = line_map.get(content, {}).get("role", "")
                    skp = bookname2role.get(bookname, "旁白1")

                wav_list = process_text(content, wav_list, skp)
            index += 1
            pbar.update(1)

    wav_list = [wav if wav.ndim == 2 else wav.unsqueeze(0) for wav in wav_list]
    wav_list = torch.concat(wav_list, dim=1)
    output_path = os.path.join(f"./tmp/{book_name}/gen", f"{book_name}_{idx}.wav")
    torchaudio.save(output_path, wav_list, 22050)
    logger.info(f"文件已保存到 {output_path}")

    process_file_path = f"./tmp/{book_name}/process.txt"
    with open(process_file_path, "a", encoding="utf-8") as process_file:
        process_file.write(f"Chapter {idx} processed and saved to {output_path}\n")


def main():
    """
    主函数
    """

    parser = argparse.ArgumentParser(
        description="Process book chapters to generate audio files."
    )
    parser.add_argument(
        "--book_name", type=str, help="Name of the book", default="诡秘之主"
    )
    parser.add_argument(
        "--start_idx", type=int, help="Starting chapter index", default=1
    )
    parser.add_argument("--end_idx", type=int, help="Ending chapter index", default=10)

    args = parser.parse_args()

    book_name = args.book_name
    start_idx = args.start_idx
    end_idx = args.end_idx

    os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        process_chapter(book_name, idx)


if __name__ == "__main__":
    main()
