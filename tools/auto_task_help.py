import sys
import os


# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from tools.zh_normalization.text_normlization import TextNormalizer

from tools.model import SenseVoiceSmall


# 加载 SenseVoiceSmall 模型
model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir)


def transcribe_and_clean(data_in, language="auto", use_itn=False):
    # 调用模型的inference方法
    res = m.inference(
        data_in=data_in,
        language=language,
        use_itn=use_itn,
        **kwargs,
    )

    # 从结果中提取文本
    raw_text = res[0][0]["text"] if res and res[0] else ""

    cleaned_text = re.sub(r"<\|.*?\|>", "", raw_text)

    # 去除空白字符
    cleaned_text = re.sub(r"\s+", "", cleaned_text)

    # 去除多余标点符号（只保留常用标点）
    cleaned_text = re.sub(r"[^\w\s，。！？]", "", cleaned_text)

    return cleaned_text


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
        r"[\"\'‘’“”\[\]【】〖〗]": "",  # 删除特殊符号
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


def get_texts(text):
    # 文本格式化
    text = format_text(text)
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    text = "".join(sentences)
    text = add_spaces_around_english(text)

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


if __name__ == "__main__":
    file_content = """
第十三章 棋子
十二月十二日，池田精神失常前一小时。
当“九月四日”这几个字从天一口中说出的刹那，池田惊慌失色，他觉得自己又一次陷入彀中，却又不知这圈套的全貌。
天一冷笑道：“我每次看到你这种嘴脸都会觉得非常厌烦，简直是可悲到了极点，你是所有这些交易者当中最让我不快的一个。”
“所有交易者？！”池田惊呼：“你还和别人交易过！”
到了这时，他似乎才明白了一些事情，可惜太晚了。
天一道：“对，就是你这种反应，多年来我都看过多少回了，你们每一个人，都忽略了还有其他交易者的可能性。说实话，我并不吃惊，基本上人类的反应九成都如此，所以你们这些人考虑任何问题，得出来的答案除了愚蠢，还是愚蠢。”
池田问道：“你还和谁交易过？难道……三浦？！松尾！”
天一叹气：“哎，和你这蠢货交流实在太辛苦，够了，这游戏到此为止吧，我已经不想和你玩下去了，反正线索条件差不多也凑齐了……”
他喝上一口咖啡，不紧不慢地道：“早在事情还未发生的时候，我已推测了你们可能做出的举动，交易只是一种引导，看你们的表现会不会超出我的预期，可惜，没有任何在我的计算范围之外的事情发生。
嗯……先来说说你的父亲吧，多年来你一直认为他是个不负责任的酒鬼，这个判断并没有错，但池田猛这个男人是有一条底线的，那就是你。
虽然你总觉得自己的人生不怎么样，但请睁大眼睛好好看看，纵然生活拮据，他还是送你去优秀的升学高中念书，打骂让你成为一个正派的人，小时候把你丢在动物园的男人，不也正是后来拼了命地四处找你的人吗？
在你把自己当成悲剧主角，抱怨着生活没有给你足够的条件时，却从未想过要靠自己去改变什么，你这种人的眼前看到的当然只有绝望。
你从未站在别人的角度上考虑过，今年也已经十七岁了，你知道父亲的生日吗？知道他的过去吗？了解他的想法吗？你什么都不知道，你和每一个凡人一样，只去考虑自己的感受，嫉妒那些天生就比自己优越的人，比如藤田、三浦那样的家伙。
但如果把你放到他们的位置上，你就不会是池田了，你便是另一个三浦而已。
你的父亲造就了你，可你不知感恩。你的心中填满了嫉妒和哀怨，但你懦弱怕事，能力又差，最终，你父亲为你买了单，他帮你杀了三浦。”
“什么！”池田颤抖，摇头，目光呆滞，口中念道：“不可能……不可能的……老爸为什么要杀三浦？！他们根本就……”
“所以我刚才就问你，要不要改变交易的内容，听听是谁杀了三浦，但你的选择跟第一次交易完成时一样的自私和愚蠢。”天一打断了池田的话道：“前天晚上，你目睹了松尾的死亡后回家，那时你的父亲其实并没有睡着，他只是为了完成我的交易而‘不和你说话’，因此他只能假装睡着。”
…………
“半夜回到家发现儿子不在，竟还满不在乎地睡了。”
…………
“昨天上午，你在学校时，他来到我的店里完成交易，接着便问我有关你昨晚究竟去哪儿了的问题。因为是我让他在特定的时间对你保持沉默的，他理所当然会认为我知道些什么。
我就告诉他，你儿子半夜去了学校，发现了尸体，并留下了线索，但没有提到任何细节。
于是，后来我们就有了一笔新的交易，我让你父亲帮我传达一个信息给你，想看看你是否能够得到启发。你应该还记得，你父亲突然心血来潮去搜电视新闻吧？”
…………
“新年将至，今年北海道的治安状况在年底依然呈下滑趋势，和全府各地区相比再次是倒数第一，除了频发的入室盗窃以外，暴力犯罪也有增加，警方发言人拒绝对此数据作出回应，今天由本台记者和我们请来的几位专家一同来……”
…………
天一慵懒地活动了两下脖子：“从你此刻的表情来看，记性不算太差嘛。其实我本人倒也不怎么看媒体报道的，我习惯直接去翻别人的想法，所以能提前知道电视上会播什么新闻。”
“你到底和多少人做过交易？”池田惊愕地问道。
天一回道：“你认识的，你不认识的，你认识的人所认识的，哈！人与人之间的联系就像错综复杂的线，只要找对了方法，像北海道这么个小地方，用极少的交易次数，就能达到牵一发而动全身的效果。”他转过头，望着身侧角落里的一个柜子：“这几天真是烧书烧得手酸啊……”
杯中的咖啡又见底了，天一添了些，继续道：“可惜你这总是幻想自己能成英雄人物的家伙，实际上太缺乏社会使命感了。实话实说，我认为你所憧憬的梦想，并不是当个英雄，而是享受英雄的待遇，却不承担英雄的代价，仅此而已。
所以你在看了新闻后完全没有反应，没有质疑，没有情绪，可悲啊，这也就是为什么，在我告诉你有‘其他交易者’之前，你根本想不到的原因。”
天一从抽屉拿出一本书，翻到了末尾，转向池田：“这是三浦最后的一些想法。”
池田看着那些文字：“杀了他……杀了他……混蛋……那个混蛋……一定要杀了他……”
“湿蚊香那家伙住得很偏僻，就在他放学的路上……无论如何也要宰了他……”
“该死，这里怎么有个酒鬼，得把他赶走。”
“这家伙！究竟……”
文字到此终止了。
天一道：“你昨天跑来我这里，其实你父亲一直在后面跟着，不过他要赶在你之前回家继续装睡，因此就没能进来找我。
今天上午你在学校跟三浦闹腾的时候，鲸鸟去了你的家，问了你父亲很多问题，让他变得越发不安起来。于是到了下午，你父亲来我这里寻求答案，我就给他看了三浦的书，当时的文字正到三浦企图杀你的内容，你爸看完以后，就回家去拿了把刀。
本来他是想打个埋伏，吓吓三浦，可那死胖子天生好斗，和你爸缠斗起来，最终，你爸就一不做二不休，把人杀了。”
池田扑向前，抓住天一的领口：“是你！都是你操纵的！这些都怪你！”
天一随手就将他推开：“有其父必有其子啊，真是一个德行。”他整理了一下衣物：“亏我还好心好意地帮你爸清理了现场，把尸体切片后转移到别的地方去了。那个大叔啊……杀完人扔了刀就跑怎么行呢，又不是随地大小便。”
“你为什么要做这些！为什么！”池田吼出声来。
天一淡然地说道：“八号晚上，松尾送完录像带回来，也问了我这个问题，我告诉他，‘因为我想看看你的贪婪’，后来在我的提点下，他放弃了自己的心之书，而和我交易了一个别人的秘密，可以让他发财的秘密。前天早上你搞错了，他不是在对你冷笑，而是在看坐在你后排的三浦。
而三浦，在九号也来问过我这个问题，我告诉他，‘因为我想看看你的暴戾’，于是他也和我做了笔交易，我承诺他，会永远守口如瓶，而条件只是让他第二天早晨揍你一顿。虽然半信半疑，但打你是家常便饭，他没理由拒绝这买卖。
十号，也就是前天，当你怀着满腹的怨恨走进我店里时，又有没有想过，今时今日，我会对你说，我只是想看看你的妒恨罢了。”
“你是疯子……疯子！”池田后退着。
“哈哈哈哈……”天一笑得确实像个疯子：“行了，快滚吧。你这种废物，我连留下‘逆十字’的兴趣都没有。不过你好歹也在我的游戏中发挥了一些作用，我最后再告诉你两件事好了。
第一，鲸鸟从最初就不曾怀疑过你，他有一种异能，像指纹、脚印、血迹等等这些，鲸鸟用肉眼就能立即看到，那天现场发生的每一件事他都能还原出来。
松尾的脖子上有两条勒痕，虽然中间部分是重合的，但当他在梁上吊久了以后，颈两侧的痕迹深浅会和中间的不一样，随便哪个警察最终都能判断出这是伪装自杀；而你在地上留下的指纹，不会成为什么证据的，因为那天的夜班保安和你做了完全一样的事情，他也在看到尸体后坐到地上倒退着爬行了。凶手是不会留下这种痕迹的，没有人会被自己刚刚布置好的现场给吓到。
昨天我只是用三浦刺激你一下，你就顺着我的意愿行事了，其实三浦也并没有如此高明，他的杀人手法是我提供的，但他执行的时候依然有瑕疵，这就是为什么我说他‘完成了一次还算不错的谋杀’，不错和完美，还差得远呢。
所以说，你在我这里所做的每笔交易，都未得到任何实质的利益，只是你那狭隘的意识在逼迫着自己成为我的棋子罢了。
还有第二点，你老爸……”天一满不在乎地说道：“会不会畏罪自杀呢……你要不要赶回去看看？”
伴随着身后让人不寒而栗的大笑，池田横冲直撞地奔出了书店的门口，再也不曾回来。
"""
    texts = get_texts(file_content)

    for text in texts:
        print(text)
