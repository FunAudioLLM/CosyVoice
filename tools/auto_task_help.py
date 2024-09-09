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
                    last_end_idx = -1  # 清空 last_end_idx
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
                if tmp_text[0] in "。？！～，":
                    tmp_text = tmp_text[1:]
                chunks.append(tmp_text)
                curr_text = curr_text[cut_point:]
                last_end_idx = -1  # 清空 last_end_idx 每次切分后
            else:
                # 找到超过阈值后的第一个标点符号
                found = False
                for i in range(curr_idx + 1, len(text)):
                    if text[i] in "。？！～，":
                        cut_point = i + 1
                        tmp_text = curr_text + text[curr_idx + 1 : cut_point]
                        if tmp_text[0] in "。？！～，":
                            tmp_text = tmp_text[1:]
                        chunks.append(tmp_text)
                        curr_text = ""
                        curr_idx = cut_point - 1  # 更新 curr_idx
                        found = True
                        break

                # 如果未找到标点符号，强制切割
                if not found:
                    tmp_text = curr_text[:max_length]
                    if tmp_text[0] in "。？！～，":
                        tmp_text = tmp_text[1:]
                    chunks.append(tmp_text)
                    curr_text = curr_text[max_length:]

        curr_idx += 1

    if curr_text:
        chunks.append(curr_text)

    return chunks


def get_texts(text, is_ignore_double_quotes=False):
    if is_ignore_double_quotes:
        text = text.replace("“", "")
        text = text.replace("”", "")
    text = text.replace("……", "。")
    text = text.replace("。。", "。")
    text = text.replace("、", "，")

    text = remove_invalid_quotes(text)
    texts = text.split("\n")
    # 清除空白符号
    texts = [text.strip() for text in texts if text.strip()]

    lines = []
    for t in texts:
        tmp = split_text(t)
        for line in tmp:
            # 文本格式化
            tx = TextNormalizer()
            sentences = tx.normalize(line)
            line = "".join(sentences)
            lines.append(line)

    return lines


def get_texts_with_line(text):
    text = text.replace("“", "").replace("“", "")
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
    text = """
第6章 非凡者
　　同样的鲁恩语，同样凝重而紧绷的感觉。
　　这是哪里？我想做什么？我也想知道。周明瑞冷静了下来，无声重复了两人的问题。
　　而让他印象最深刻的，不是单词所构成的句子，句子所蕴含的意思，而是那一男一女表现出的慌乱、警惕、惶恐和敬畏！
　　莫名其妙将两个人拉入这片灰雾世界之上，就算身为肇事者的自己，也是异常地错愕和震惊，更何况属于被动一方的他们！
　　在他们看来，这种事情这种遭遇恐怕已超越想象了吧？
　　这个瞬间，周明瑞想到了两个选择，一是假装自己也是受害者，隐藏住真实的身份，以此换取一定程度的信任，静观其变，浑水摸鱼，二是维持那一男一女眼中神秘莫测的形象，主动引导事情的发展，从中获取有价值的信息。
　　来不及多思考多推敲，周明瑞抓住脑海内一闪而过的想法，迅速做出决断，尝试第二种办法。
　　利用对方现在的心理状态，把握自身最大的优势！
　　灰雾之上短暂沉默了几秒，周明瑞轻笑了一声，语气平淡，嗓音低而不沉，就像在回应访客礼貌性的问候：
　　“一个尝试。”
　　一个尝试。一个尝试？奥黛丽·霍尔望着那被灰白雾气笼罩的神秘男子，只觉事情荒唐、好笑、惊悚、奇诡。
　　自己刚还在卧室内，梳妆台前，转头便来到了这满是灰雾的地方！
　　这是何等的匪夷所思！
　　奥黛丽吸了口气，露出无懈可击的礼节性笑容，颇为忐忑地问道：
　　“阁下，尝试结束了吗？可以让我们回去了吗？”
　　阿尔杰·威尔逊也想做类似的试探，但经历丰富的他更为沉稳，按捺住了冲动，只是沉默着旁观。
　　周明瑞望向提问者，隐约能透过模糊看见对方的身影，那是位有着柔顺金发、个子高挑的少女，但具体容貌不太清晰。
　　他没急着回答少女的问题，转头又看向另一边的男子，对方头发深蓝，如海草般凌乱，身材中等，不算健硕。
　　此时此刻，周明瑞突地有了明悟，等到自己更为强大，或者对这灰雾世界了解更深，也许就能真正看穿朦胧，看清楚少女与男子的长相。
　　这次的事件里，他们是来客，我是主人！
　　心态一变，周明瑞立刻感受到了刚才没有注意的一些细节。
　　嗓音甜美的少女和沉稳内敛的男子都相当虚幻，染着微赤，就像那两颗深红星辰在灰雾之上的投影。
　　而这投影是基于自己与深红之间的联系，无影无形但本身能真切把握到的联系。
　　切断这个联系，投影就会消散，他们就能回归。周明瑞微不可见地点头，看向金发少女，轻声笑道：
　　“当然，如果你正式提出，我现在就能让你回去。”
　　听不出恶意的奥黛丽松了口气，相信能做出如此神奇事情的先生既然给予承诺，那就肯定会严格遵守。
　　精神稍有平复，她反倒没急着提出离开，碧绿的眼眸左右转动了一下，闪烁出异样的光彩。
　　她忐忑、期待、跃跃欲试般道：
　　“这真是一次奇妙的体验。嗯，我一直期待着类似的事情，我是说，我喜欢神秘，喜欢超越自然的奇迹，不，我的重点，我的意思是，阁下，我该怎样做才能成为非凡者？”
　　她越说越是兴奋，甚至激动得有点语无伦次，小时候听长辈们讲种种奇闻怪谈时萌芽的梦想似乎终于有了实现的曙光。
　　不过几句话的工夫，她已将之前的害怕和惶恐遗忘于了脑后。
　　问得好！我也想知道答案。周明瑞自我吐槽道。
　　他开始思考该用怎样的回答维持神秘莫测的形象。
　　与此同时，他觉得这样站着对话显得有点LOW，如此场景不是该有一座神殿，一张长桌，以及众多雕刻着古老花纹、满是神秘感觉的靠背座椅，而自己端坐最上首，静静注视着客人吗？
　　周明瑞念头刚落，灰雾突地翻滚，吓了奥黛丽和阿尔杰一跳。
　　瞬息之间，他们看见周围多了一根根高耸的石柱，看见上方被宽广的穹顶笼罩。
　　整个建筑壮观、恢弘、巍峨，就像是传说里巨人的王殿。
　　穹顶正下方，灰雾簇拥处，多了一张青铜长桌，左右各有十张高背椅，前后亦安置着同样的座位，椅子背面，璀璨闪烁，深红暗敛，勾勒出不与现实对应的奇怪星座。
　　奥黛丽和阿尔杰正好相对而坐，处于最靠近上首的位置。
　　少女往左看了看，又往右瞧一瞧，忍不住低声嘀咕道：
　　“真是神奇啊。”
　　确实神奇。周明瑞伸出右手，幅度很小地摩挲着青铜长桌的边缘，表面不动声色。
　　阿尔杰亦是四下打量了一遍，几秒的沉默后，他突地开口，代替周明瑞回答了奥黛丽的问题：
　　“你是鲁恩人吧？”
　　“想成为非凡者，就加入黑夜女神教会，风暴之主教会，或蒸汽与机械之神教会。”
　　“虽然绝大多数人一生都见不到非凡，以至于怀疑教会也是同样的情况，甚至在几大教会内部，不少神职人员也有类似的想法，但我可以明确告诉你，在仲裁庭，在裁判所，在处刑机关，非凡者依旧存在，依旧在对抗着黑暗里生长的危险，只是数量和黑铁时代早期或之前相比，少了很多很多。”
　　周明瑞专注听着，肢体动作却竭力表现出听小朋友讲故事的不在意态度。
　　依靠克莱恩残留的历史学常识，他清楚黑铁时代指的是当前纪元，也就是第五纪，开始于一千三百四十九年前。
　　奥黛丽安静听完，轻呼了一口气道：
　　“先生，你说的我都知道，甚至知道更多，比如值夜者，比如代罚者，比如机械之心，但是，我不想失去自由。”
　　阿尔杰低笑了一声，含糊道：
　　“哪有不想付出代价就成为非凡者的？如果不考虑加入教会，接受考验，那你只能去找王室，找家族历史在千年以上的那几位贵族，或者，凭运气寻觅那些躲躲藏藏的邪恶组织。”
　　奥黛丽下意识鼓了鼓腮帮子，接着慌乱地左看右看，等确定神秘先生和对面的家伙都没有注意到自己的小动作，才追问道：
　　“没有别的办法了吗？”
　　阿尔杰陷入了沉默，十几个呼吸后，他扭头望向不发一言安静旁观的神秘先生周明瑞。
　　见对方不置可否，他才看回奥黛丽，斟酌着说道：
　　“我手上其实有两份序列9的魔药配方。”
　　序列9？周明瑞暗自嘀咕。
　　“真的？是哪两份？”奥黛丽明显很清楚序列9的魔药配方代表着什么。
　　阿尔杰往后微靠，语气不快不慢地回答：
　　“你知道的，人类想要成为真正的非凡者，只能依靠魔药，而魔药的名称来自‘亵渎石板’，经过巨人语、精灵语、古赫密斯语、古弗萨克语、当代赫密斯语地不断转译，早就有了符合时代特征的变化，名称不是重点，重点是它能否代表这份魔药的‘核心象征’。”
　　“我手中的序列9配方，一份叫做‘水手’，它能让你拥有出色的平衡能力，哪怕在暴风雨笼罩的船上，也能自由行走如大地，你还能获得卓越的力量，以及隐藏于皮肤下的幻鳞，这会让你像鱼一样难以被抓住，在水中灵活得仿佛海族，哪怕不用任何装备，也能轻松地潜水至少十分钟。”
　　“听起来很棒。风暴之主的‘海眷者’？”奥黛丽半是期待半是求证地反问。
　　“在古代，它确实叫做‘海眷者’。”阿尔杰没做停顿，继续说道，“第二份序列9配方叫做‘观众’，至于古代怎么称呼，我就不知道了。这份魔药能让你得到出众的精神和敏锐的观察力，我相信你看过歌剧和戏剧，能明白‘观众’代表的意思，像旁观者一样，审视世俗社会里的‘演员’，从他们的表情，他们的举止，他们的口癖，他们不为人知的动作窥见他们真实的想法。”
　　说到这里，阿尔杰强调了一句：
　　“你必须记住，不管是奢靡的宴会，还是热闹的街头，观众永远只是观众。”
　　奥黛丽听得眼睛发亮，好半天才道：
　　“为什么？好吧，这是后续的问题，我，我想我喜欢上了这种感觉，‘观众’，我该怎样获得‘观众’的配方？用什么和你交换？”
　　阿尔杰像是早有准备，沉声回答道：
　　“鬼鲨的血，至少100毫升鬼鲨的血。”
　　奥黛丽先是兴奋点头，继而担忧问道：
　　“如果我能拿到，我是说如果，我该怎么给你？又该怎么保证你拿到鬼鲨血后，将魔药的配方给我，以及这份配方的真实？”
　　阿尔杰语气平常道：
　　“我会给你一个地址，等我收到鬼鲨血，就回寄配方给你，或者直接在这里告诉你。”
　　“至于保证，我想如果有这位神秘的阁下的见证，你和我都会足够放心。”
　　说这句话的时候，他将目光转向了端坐上首的周明瑞：
　　“阁下，您能拉我们来到这里，拥有我们无法想象的伟力，您做的见证，不管是我，还是她，都不敢违背。”
　　“对！”奥黛丽眼睛一亮，激动赞同。
　　在她看来，手段让人无法想象的神秘先生确实是足够权威的见证。
　　自己和对面的家伙哪有胆量欺骗他！
　　奥黛丽半转身体，诚恳望向了周明瑞：
　　“阁下，请您做我们交易的见证。”
　　这个时候，她才发现自己竟然一直遗忘了某个问题，太不够礼貌，忙又问道：
　　“阁下，我们该怎么称呼您？”
　　阿尔杰微微点头，跟着庄重问道：
　　“阁下，我们该怎么称呼您？”
　　周明瑞听得愣了一下，放在青铜长桌上的手指轻轻敲动起来，脑海内霍然闪过了之前占卜的内容。
　　他往后一靠，收回右手，十指交叉着抵于下巴，微笑看着两人道：
　　“你们可以称呼我。”
　　说到这里，他顿了顿，语气轻和而平淡地开口：
　　“愚者。”
"""
    texts = get_texts(text)

    for text in texts:
        print(text)

    # print(text)
