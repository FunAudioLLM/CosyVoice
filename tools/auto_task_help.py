import re
from pypinyin import pinyin, Style
from difflib import SequenceMatcher


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
        text = text.replace("。", "，").replace("！", "，").replace("？", "，")

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
    text = text.replace("、", "，")
    text = text.replace("；", "，")

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


def has_omission(gen_text, text):
    """
    检查生成的文本是否有遗漏原文的内容，通过拼音比较和相似度判断。
    :param gen_text: 生成的文本
    :param text: 原始文本
    :return: 若生成的文本拼音相似度超过98%且没有增加重复字，则返回 False（没有遗漏），否则返回 True（有遗漏）
    """

    def remove_punctuation(text):
        """
        移除文本中的标点符号。
        """
        return re.sub(r"[^\w\s]", "", text)

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
    gen_text_clean = remove_punctuation(gen_text)
    text_clean = remove_punctuation(text)

    # 获取拼音
    gen_text_pinyin = get_pinyin(gen_text_clean)
    text_pinyin = get_pinyin(text_clean)

    gen_text_ping_duo = get_pinyin_duo(gen_text_clean)
    text_ping_duo = get_pinyin_duo(text_clean)

    # 计算拼音相似度
    sim_ratio = calculate_similarity(gen_text_pinyin, text_pinyin) * 100

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


def get_texts(text, language="zh"):
    text = split_text_by_punctuation(text, {"。", "？", "！", "～"})
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 10)
    texts = cut_text(texts, 40)
    return texts
