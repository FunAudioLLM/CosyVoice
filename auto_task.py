import re
import numpy as np
import torch
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import argparse
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

prompt_speecher = {
    # "旁白": {
    #     "wav_path": "./tmp/model/打道回府，告诉公司有个蠢货把一切都搞砸了。.wav",
    #     "wav_txt": "打道回府，告诉公司有个蠢货把一切都搞砸了。",
    # }
    # "旁白": {
    #     "wav_path": "tmp/model/樊娜轻轻点了点头，什么都没说，她脑海中浮现出的却是昨夜那惊悚的梦境。.wav",
    #     "wav_txt": "樊娜轻轻点了点头，什么都没说，她脑海中浮现出的却是昨夜那惊悚的梦境。",
    # }
    "旁白": {
        "wav_path": "tmp/model/out.wav",
        "wav_txt": "一个邋遢的乞丐坐在地上，正用那发黑的手无聊的挖着鼻孔。和其他乞丐一样，穿着一身满是补丁的布衣。裤子膝盖上碗大的一个破洞，那还能叫裤子。",
    }
}

logger.info(f"CUDA available: {torch.cuda.is_available()}")

# 初始化 CosyVoice 实例
cosyvoice = CosyVoice("pretrained_models/CosyVoice-300M")

# 加载提示语音
# prompt_speech_16k = load_wav(prompt_speecher["旁白"]["wav_path"], 16000)

# 定义语言对应的符号
language_punctuation = {
    "zh": {
        "comma": "，",
        "period": "。",
        "question_mark": "？",
        "exclamation_mark": "！",
        "ellipsis": "…",
        "colon": "：",
        "newline": "。",
    },
    "en": {
        "comma": ",",
        "period": ".",
        "question_mark": "?",
        "exclamation_mark": "!",
        "ellipsis": "...",
        "colon": ":",
        "newline": ".",
    },
}


def split_text_by_punctuation(text, punctuations):
    """
    将输入文本按指定的标点符号进行切割，并保留句末标点符号。

    参数:
    text (str): 输入文本。
    punctuations (set): 用于切割文本的标点符号集合。

    返回:
    str: 切割后的文本，每句之间用换行符分隔。
    """
    segments = []
    start = 0
    for i, char in enumerate(text):
        if char in punctuations:
            segments.append(text[start : i + 1].strip())
            start = i + 1
    if start < len(text):
        segments.append(text[start:].strip())
    return "\n".join(segments)


def process_text(texts):
    """
    处理输入的文本列表，移除无效的文本条目。

    参数:
    texts (list): 包含文本条目的列表。

    返回:
    list: 处理后的有效文本列表。

    异常:
    ValueError: 当输入的文本列表中全是无效条目时抛出。
    """
    # 判断列表中是否全是无效的文本条目
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")

    # 过滤并返回有效的文本条目
    return [text for text in texts if text not in [None, " ", "", "\n"]]


def merge_short_text_in_array(texts, threshold):
    """
    合并短文本，直到文本长度达到指定的阈值。

    参数:
    texts (list): 包含短文本的列表。
    threshold (int): 合并文本的阈值长度。

    返回:
    list: 合并后的文本列表。
    """
    # 如果文本列表长度小于2，直接返回原列表
    if len(texts) < 2:
        return texts

    result = []
    current_text = ""

    for text in texts:
        # 将当前文本添加到合并文本中
        current_text += text
        # 如果合并文本长度达到阈值，将其添加到结果列表中，并重置合并文本
        if len(current_text) >= threshold:
            result.append(current_text)
            current_text = ""

    # 处理剩余的文本
    if current_text:
        # 如果结果列表为空，直接添加剩余文本
        if not result:
            result.append(current_text)
        else:
            # 否则，将剩余文本添加到结果列表的最后一个元素中
            result[-1] += current_text

    return result


def cut_text(texts, num=30, language="zh"):
    """
    将文本列表按指定长度切割，尽量在标点符号处进行切割，确保每段长度大致相等。若 text 长度大于 num 且只有一个标点符号则不切割。

    参数:
    texts (list): 包含文本段落的列表。
    num (int): 每段的最大字符数。
    language (str): 文本语言（用于选择标点符号）。

    返回:
    list: 切割后的文本段落列表。
    """
    result = []
    for t in texts:
        while len(t) > num:
            punctuation_positions = [
                t.rfind(p, 0, num) for p in language_punctuation[language].values()
            ]
            punctuation_positions = [pos for pos in punctuation_positions if pos != -1]

            if punctuation_positions:
                cut_index = max(punctuation_positions)
            else:
                # 找不到标点符号时，在最接近 num 的地方切割
                cut_index = num - 1
                for offset in range(0, num):
                    if num - offset >= 0 and re.match(r"\W", t[num - offset]):
                        cut_index = num - offset
                        break
                    if num + offset < len(t) and re.match(r"\W", t[num + offset]):
                        cut_index = num + offset
                        break

            result.append(t[: cut_index + 1].strip())
            t = t[cut_index + 1 :].strip()

        if t:
            result.append(t)
    return result


def format_text(text, language="zh"):
    text = text.strip("\n")
    # 根据语言获取对应的符号
    punct = (
        language_punctuation["zh"] if "zh" in language else language_punctuation["en"]
    )

    # 替换规则
    text = re.sub(r" {2,}", punct["period"], text)  # 多个空格替换为句号
    text = re.sub(r"\n|\r", punct["newline"], text)  # 回车，换行符替换为句号
    text = re.sub(r" ", punct["comma"], text)  # 一个空格替换为逗号
    text = re.sub(r"[\"\'‘’“”\[\]【】〖〗]", "", text)  # 删除特殊符号
    text = re.sub(r"[:：……—]", punct["period"], text)  # 替换为句号

    # 替换所有非当前语言的符号为对应语言的符号
    if language == "en":
        text = re.sub(
            r"[，。？！…～：]",
            lambda match: (
                punct["comma"]
                if match.group(0) == "，"
                else punct["period"]
                if match.group(0) == "。"
                else punct["question_mark"]
                if match.group(0) == "？"
                else punct["exclamation_mark"]
                if match.group(0) == "！"
                else punct["ellipsis"]
                if match.group(0) == "…"
                else punct["period"]
            ),
            text,
        )
    elif language == "zh":
        text = re.sub(
            r"[,.\?!~:]+",
            lambda match: (
                punct["comma"]
                if match.group(0) == ","
                else punct["period"]
                if match.group(0) == "."
                else punct["question_mark"]
                if match.group(0) == "?"
                else punct["exclamation_mark"]
                if match.group(0) == "!"
                else punct["ellipsis"]
                if match.group(0) == "..."
                else punct["period"]
            ),
            text,
        )

    # 确保文本开头没有标点符号
    text = re.sub(r"^[，。？！…～：]|^[,.?!~:]", "", text)

    # 保留多个连续标点符号中的第一个
    def remove_consecutive_punctuation(text):
        result = []
        i = 0
        while i < len(text):
            if text[i] in punct.values():
                result.append(text[i])
                while i + 1 < len(text) and text[i + 1] in punct.values():
                    i += 1
            else:
                result.append(text[i])
            i += 1
        return "".join(result)

    text = remove_consecutive_punctuation(text)

    return text


def get_texts(text, language="zh"):
    text = split_text_by_punctuation(text, {"。", "？", "！", "～"})
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 10)
    texts = cut_text(texts, 60)
    return texts


def process_chapter(book_name, idx, prompt_speech_16k):
    # Read file path
    file_path = f"./tmp/{book_name}/data/chapter_{idx}.txt"

    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"文件 {file_path} 不存在，跳过处理。")
        return

    # Read file content
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    file_content = format_text(file_content)
    texts = get_texts(file_content)

    # Initialize wav_list
    wav_list = []

    for i, line in enumerate(tqdm(texts, desc=f"Processing chapter {idx}")):
        # Clear torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output = cosyvoice.inference_zero_shot(
            line,
            prompt_speecher["旁白"]["wav_txt"],
            prompt_speech_16k,
        )

        # Save temporary output
        tts_speech = output["tts_speech"]
        # Ensure tts_speech is 2-dimensional
        if tts_speech.ndim == 1:
            tts_speech = tts_speech.unsqueeze(0)
        wav_list.append(tts_speech)

        if line[-1] == "，":
            random_duration = np.random.uniform(0.05, 0.08)
        else:
            random_duration = np.random.uniform(0.09, 0.13)

        zero_wav = np.zeros(
            int(22050 * random_duration),
            dtype=np.float32,
        )
        # Convert numpy array to torch tensor and ensure it is 2-dimensional
        zero_wav = torch.from_numpy(zero_wav).unsqueeze(0)
        wav_list.append(zero_wav)

    # Concatenate and save to ./tmp/{book_name}/gen/{idx}.wav
    wav_list = torch.concat(wav_list, dim=1)
    output_path = os.path.join(f"./tmp/{book_name}/gen", f"{idx}.wav")
    torchaudio.save(output_path, wav_list, 22050)
    logger.info(f"文件已保存到 {output_path}")

    # Record processing information
    process_file_path = f"./tmp/{book_name}/process.txt"
    with open(process_file_path, "a", encoding="utf-8") as process_file:
        process_file.write(f"Chapter {idx} processed and saved to {output_path}\n")


def process_chapter_v2(book_name, idx):
    # Read file path
    file_path = f"./tmp/{book_name}/data/chapter_{idx}.txt"

    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"文件 {file_path} 不存在，跳过处理。")
        return

    # Read file content
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    file_content = format_text(file_content)
    texts = get_texts(file_content)

    # Initialize wav_list
    wav_list = []

    for i, line in enumerate(tqdm(texts, desc=f"Processing chapter {idx}")):
        # Clear torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output = cosyvoice.inference_sft(line, "旁白")

        # Save temporary output
        tts_speech = output["tts_speech"]
        # Ensure tts_speech is 2-dimensional
        if tts_speech.ndim == 1:
            tts_speech = tts_speech.unsqueeze(0)
        wav_list.append(tts_speech)

        if line[-1] == "，":
            random_duration = np.random.uniform(0.05, 0.08)
        else:
            random_duration = np.random.uniform(0.09, 0.13)

        zero_wav = np.zeros(
            int(22050 * random_duration),
            dtype=np.float32,
        )
        # Convert numpy array to torch tensor and ensure it is 2-dimensional
        zero_wav = torch.from_numpy(zero_wav).unsqueeze(0)
        wav_list.append(zero_wav)

    # Concatenate and save to ./tmp/{book_name}/gen/{idx}.wav
    wav_list = torch.concat(wav_list, dim=1)
    output_path = os.path.join(f"./tmp/{book_name}/gen", f"{idx}.wav")
    torchaudio.save(output_path, wav_list, 22050)
    logger.info(f"文件已保存到 {output_path}")

    # Record processing information
    process_file_path = f"./tmp/{book_name}/process.txt"
    with open(process_file_path, "a", encoding="utf-8") as process_file:
        process_file.write(f"Chapter {idx} processed and saved to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process book chapters to generate audio files."
    )
    parser.add_argument("--book_name", type=str, help="Name of the book", default="jy")
    parser.add_argument(
        "--start_idx", type=int, help="Starting chapter index", default=1
    )
    parser.add_argument("--end_idx", type=int, help="Ending chapter index", default=1)

    args = parser.parse_args()

    book_name = args.book_name
    start_idx = args.start_idx
    end_idx = args.end_idx

    # 创建输出目录
    os.makedirs(f"./tmp/{book_name}/gen", exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        # process_chapter(book_name, idx, prompt_speech_16k)
        process_chapter_v2(book_name, idx)


if __name__ == "__main__":
    main()
