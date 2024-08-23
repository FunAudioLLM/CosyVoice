import sys
import os
import traceback
import uuid


# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from tools.zh_normalization.text_normlization import TextNormalizer

from funasr import AutoModel

"""
auto_task 辅助函数
"""
# 初始化全局变量
model = None
kwargs = {}


def transcribe_and_clean(input_audio, language="zn", use_itn=False):
    import torch
    import soundfile as sf

    global model
    if model is None:
        model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 60000},
            # punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            device="cuda:0",
        )

    try:
        # 检查输入是否是文件路径或 tensor
        if isinstance(input_audio, str):
            text = model.generate(
                input=input_audio,
                batch_size_s=60,
                language=language,  # "zn", "en", "yue", "ja", "ko", "nospeech"
                cache={},
                use_itn=use_itn,
                merge_vad=True,
                merge_length_s=60,
            )[0]["text"]
        else:
            # 确保输入是 torch.Tensor 并且是 1 维
            if isinstance(input_audio, torch.Tensor) and len(input_audio.shape) == 2:
                input_audio = input_audio.squeeze(0)

            # 将 tensor 保存为临时 wav 文件
            tmp_name = str(uuid.uuid4())
            temp_wav_path = f"./tmp/gen_temp/{tmp_name}.wav"
            sf.write(temp_wav_path, input_audio, 32768)

            # 使用临时文件进行推理
            text = model.generate(
                input=input_audio,
                batch_size_s=60,
                language=language,  # "zn", "en", "yue", "ja", "ko", "nospeech"
                cache={},
                use_itn=use_itn,
                merge_vad=True,
                merge_length_s=60,
            )[0]["text"]
        os.remove(temp_wav_path)
    except:
        text = ""
        print(traceback.format_exc())
    return text


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
        # 小于 num 则直接添加
        if len(t) <= num:
            result.append(t)
            continue

        # 按照标点符号拆分原始句子
        punctuation = language_punctuation[language]
        split_pattern = f"([{punctuation['comma']}{punctuation['question_mark']}{punctuation['exclamation_mark']}{punctuation['newline']}])"
        sentences = [
            sentence.strip()
            for sentence in re.split(split_pattern, t)
            if sentence.strip()
        ]

        # 合并标点符号和句子
        merged_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                merged_sentences.append(sentences[i] + sentences[i + 1])
            else:
                merged_sentences.append(sentences[i])

        # 从前往后加句子，若添加的句子大于剩下的句子时，则不进行停止追加
        current_part = ""
        for sentence in merged_sentences:
            if len(current_part) + len(sentence) <= num:
                current_part += sentence
            else:
                if current_part:
                    result.append(current_part.strip())
                current_part = sentence

        if current_part:
            result.append(current_part.strip())

    return result


def format_text(text, language="zh"):
    text = text.strip("\n")
    punct = language_punctuation[language]

    # 替换规则
    replacements = {
        r" {2,}": punct["period"],  # 多个空格替换为句号
        r"\n|\r": punct["newline"],  # 回车，换行符替换为句号
        r"[ 、]": punct["comma"],  # 一个空格替换为逗号
        r"[:：……—；]": punct["period"],  # 替换为句号
        # r"[\"\'‘’“”\[\]【】〖〗]": "",  # 删除特殊符号
        r"[\'‘’\[\]【】〖〗]": "",
        r"[“”]": '"',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # 替换所有非当前语言的符号为对应语言的符号
    if language == "en":
        text = re.sub(
            r"[，。？！…～：]",
            lambda match: punct.get(
                next(
                    (
                        k
                        for k, v in language_punctuation["zh"].items()
                        if v == match.group(0)
                    ),
                    "period",
                )
            ),
            text,
        )
    elif language == "zh":
        text = re.sub(
            r"[,.\?!~:]+",
            lambda match: punct.get(
                next(
                    (
                        k
                        for k, v in language_punctuation["en"].items()
                        if v == match.group(0)
                    ),
                    "period",
                )
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


def remove_invalid_quotes(text):
    # 定义常见的中文标点符号
    chinese_punctuation = "。！？；：，、……"

    def replace_func(match):
        # 获取匹配到的文本
        quoted_text = match.group(1)
        # 检查最后一个字符是否为中文标点符号
        if quoted_text and quoted_text[-1] in chinese_punctuation:
            return f"“{quoted_text}”"  # 保留双引号
        else:
            return quoted_text  # 移除双引号

    # 使用正则表达式匹配中文双引号内的内容
    pattern = r"“(.*?)”"
    result = re.sub(pattern, replace_func, text)

    return result


def has_omission(gen_data, text):
    gen_text = transcribe_and_clean(gen_data)
    tx = TextNormalizer()
    sentences = tx.normalize(gen_text)
    gen_text = "".join(sentences)
    return _has_omission(gen_text, text)


def _has_omission(gen_text, text):
    """
    检查生成的文本是否有遗漏原文的内容，通过拼音比较和相似度判断。
    :param gen_text: 生成的文本
    :param text: 原始文本
    :return: 若生成的文本拼音相似度超过98%且没有增加重复字，则返回 False（没有遗漏），否则返回 True（有遗漏）
    """

    def clean_text(text):
        """
        移除文本中的标点符号、空白符，并将英文全部转为小写。
        """
        # 移除标点符号
        text = re.sub(r"[^\w\s]", "", text)
        # 移除空白符
        text = re.sub(r"\s+", "", text)
        # 将英文全部转为小写
        text = text.lower()
        return text

    def get_pinyin(text):
        """
        获取文本的拼音表示。
        """
        return " ".join(["".join(p) for p in pinyin(text, style=Style.TONE2)])

    def get_pinyin_duo(text):
        """
        获取文本的拼音表示，支持多音字。
        """
        res = []
        for ch in text:
            res.extend(pinyin(ch, heteronym=True, style=Style.FIRST_LETTER))
        return res

    def calculate_similarity(pinyin1, pinyin2):
        """
        计算两个拼音字符串的相似度。
        """
        return SequenceMatcher(None, pinyin1, pinyin2).ratio()

    # 去除标点符号
    gen_text_clean = clean_text(gen_text)
    text_clean = clean_text(text)

    if gen_text_clean == text_clean:
        return False, 100, gen_text_clean, text_clean

    # 获取拼音
    gen_text_pinyin = get_pinyin(gen_text_clean)
    text_pinyin = get_pinyin(text_clean)

    gen_text_ping_duo = get_pinyin_duo(gen_text_clean)
    text_ping_duo = get_pinyin_duo(text_clean)

    # 计算拼音相似度
    sim_ratio = calculate_similarity(gen_text_pinyin, text_pinyin) * 100
    # 如果有“儿”字，则增加5分
    if "儿" in text:
        sim_ratio += 5

    res = True
    # 判断是否有遗漏
    if len(gen_text_clean) != len(text_clean):
        # 如果字数不等，根据拼音相似度判断，每个字的差异减少5%的相似度
        length_difference = abs(len(gen_text_clean) - len(text_clean))
        res = True
        sim_ratio = sim_ratio - length_difference * 5
    else:
        # 对比 gen_text_ping_duo 与 text_ping_duo
        # 判断每个字符是否存在多音字，若存在，则对比两个字符多音字是否有相同，若有则满足字符相等，若不存在则减 5
        is_multi_word = True
        for gen_word, text_word in zip(gen_text_ping_duo, text_ping_duo):
            if not any(g in text_word for g in gen_word):
                sim_ratio -= 5
                is_multi_word = False
        if is_multi_word:
            sim_ratio = 100
        res = sim_ratio < 98

    return res, sim_ratio, gen_text_clean, text_clean


def add_spaces_around_english(text):
    """
    在前后遇到英文字符时添加空格
    """
    # 在英文字符前添加空格
    text = re.sub(r"(?<=[^\x00-\x7F])([A-Za-z])", r" \1", text)
    # 在英文字符后添加空格
    text = re.sub(r"([A-Za-z])(?=[^\x00-\x7F])", r"\1 ", text)

    return text


def replace_punctuations(sentence):
    # 使用正则表达式替换句子中间的标点符号为逗号
    def replacer(match):
        return "，" if match.end() != len(sentence) else match.group(0)

    return re.sub(r"[。！？]", replacer, sentence)


def format_line(text):
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    text = "".join(sentences)
    return text


def get_texts_v2(text):
    text = remove_invalid_quotes(text)

    texts = text.split("\n")
    text_list = []

    for line in texts:
        # 拆分成 ["“你好，我是小明。”", "小明说到，",  "“很高兴见到您。”","小明高兴的和小红握手。"]
        parts = []
        buffer = ""

        for char in line:
            if char == "“":
                if buffer:
                    parts.append(buffer.strip())
                    buffer = ""
                buffer += char
            elif char == "”":
                buffer += char
                if not buffer.endswith(("。", "，", "！", "？", "；", "：", "”")):
                    # 如果引号内没有标点符号，则删除引号
                    buffer = buffer.replace("“", "").replace("”", "").strip()
                parts.append(buffer.strip())
                buffer = ""
            else:
                buffer += char

        if buffer:
            parts.append(buffer.strip())

        text_list.extend(parts)

    return text_list


def split_text(text, max_length=30):
    chunks = []
    curr_idx = 0
    curr_text = ""
    last_end_idx = -1

    while curr_idx < len(text):
        ch = text[curr_idx]
        if ch == "“":
            # 保存之前所有文本到chunks
            if curr_text:
                tmp_text = curr_text
                if tmp_text[0] in "。？！，":
                    tmp_text = tmp_text[1:]
                if tmp_text:
                    chunks.append(tmp_text)
            curr_text = ""
            tmp_text = ""
            while curr_idx < len(text):
                temp_ch = text[curr_idx]
                tmp_text += temp_ch
                if temp_ch == "”":
                    chunks.append(tmp_text)
                    last_end_idx = curr_idx
                    break
                curr_idx += 1
            curr_idx += 1
            continue
        if ch in "。？！～，":
            last_end_idx = len(curr_text)
        curr_text += ch

        if len(curr_text) > max_length:
            if last_end_idx != -1:
                cut_point = last_end_idx + 1
                tmp_text = curr_text[:cut_point]
                if tmp_text[0] in "。？！，":
                    tmp_text = tmp_text[1:]
                chunks.append(tmp_text)
                curr_text = curr_text[cut_point:]
            else:
                tmp_text = curr_text[:cut_point]
                if tmp_text[0] in "。？！，":
                    tmp_text = tmp_text[1:]
                chunks.append(tmp_text)
                curr_text = ""
            last_end_idx = -1

        curr_idx += 1

    if curr_text:
        chunks.append(curr_text)

    return chunks


def get_texts(text):
    text = text.replace("……", "。")
    text = text.replace("。。", "。")
    text = text.replace("、", "，")

    text = remove_invalid_quotes(text)
    # 文本格式化
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    text = "".join(sentences)
    text = add_spaces_around_english(text)

    texts = split_text(text)

    return texts


def get_texts_with_line(text):
    # 切割文本
    text = split_text_by_punctuation(text, {"。", "？", "！", "～"})
    texts = text.split("\n")
    texts = process_text(texts)

    # 切分长句子
    texts = cut_text(texts, 30)

    # 合并短句子
    texts = merge_short_text_in_array(texts, 10)

    # 替换标点符号
    texts = [replace_punctuations(sentence) for sentence in texts]
    return texts


def remove_punctuation_only_paragraphs(text):
    # 按段落分割文本
    paragraphs = text.split("\n")
    # 定义常见的中文标点符号
    chinese_punctuation = "。！？；：，、"
    # 过滤掉只包含标点符号的段落
    filtered_paragraphs = [
        p for p in paragraphs if not all(char in chinese_punctuation for char in p)
    ]
    # 将段落重新拼接成文本
    return "\n".join(filtered_paragraphs)


def remove_punctuation_only_quotes(text):
    # 定义常见的中文标点符号
    chinese_punctuation = "。！？；：，、……"

    def replace_func(match):
        # 获取匹配到的文本
        quoted_text = match.group(1)
        # 判断引号中的内容是否全部为标点符号
        if all(char in chinese_punctuation for char in quoted_text):
            return ""  # 移除双引号及其中间内容
        else:
            return f"“{quoted_text}”"  # 保留双引号及其中间内容

    # 使用正则表达式匹配中文双引号内的内容
    pattern = r"“(.*?)”"
    result = re.sub(pattern, replace_func, text)

    return result


def format_text_v2(text):
    # 1. 将中文省略号替换为中文句号
    text = text.replace("……", "。")
    text = text.replace("“。”", "")
    text = text.replace("“！”", "")
    text = text.replace("“？”", "")
    text = text.replace("“～”", "")
    text = text.replace("“”", "")

    # 2. 移除每段文本中只包含标点符号的段落
    text = remove_punctuation_only_paragraphs(text)

    # 3. 移除双引号中仅包含标点符号的内容
    text = remove_punctuation_only_quotes(text)

    # 4. 移除连续的空段落
    text = re.sub(r"\n+", "\n", text).strip()

    # 5. 移除无效的引号
    text = remove_invalid_quotes(text)

    return text


if __name__ == "__main__":
    text = "第十三章，棋子十二月十二日，池田精神失常前一小时。当九月四日这几个字从天一口中说出的刹那，池田惊慌失色，他觉得自己又一次陷入彀中，却又不知这圈套的全貌。天一冷笑道：“我每次看到你这种嘴脸都会觉得非常厌烦，简直是可悲到了极点，你是所有这些交易者当中最让我不快的一个。”“所有交易者？！”池田惊呼：“你还和别人交易过！”到了这时，他似乎才明白了一些事情，可惜太晚了。"

    texts = get_texts(text)

    for text in texts:
        print(text)

    # print(text)
