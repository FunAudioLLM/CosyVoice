import argparse
import os
import re
import shutil
import cn2an
import logging

logging.basicConfig(level=logging.INFO)


def read_novel(novel_name):
    """
    读取小说内容并进行预处理
    :param novel_name: 小说名
    :return: 预处理后的小说内容
    """
    novel_path = f"tmp/{novel_name}/{novel_name}.txt"
    try:
        with open(novel_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        logging.error(f"File not found: {novel_path}")
        return None

    # 简单清洗文本
    content = content.replace("\r\n", "\n").strip()

    # 清洗章节名： 第一百二十三章 -> 第123章  第 一千三百五十一 章 -> 第1351章
    content = clean_chapter_titles(content)

    return content


def clean_chapter_titles(content):
    """
    清洗章节标题
    :param content: 原始内容
    :return: 清洗后的内容
    """

    def replace_chinese_number(match):
        chinese_number = match.group(1).replace(" ", "")
        arabic_number = cn2an.cn2an(chinese_number, "smart")
        return f"第{arabic_number}章 {match.group(2)}"

    pattern = r"第([零一二三四五六七八九十百千万]+)章 (.*?)+"
    cleaned_content = re.sub(pattern, replace_chinese_number, content)
    return cleaned_content


def split_novel(novel_name, content):
    """
    拆分小说为每一章并保存到文件
    :param novel_name: 小说名
    :param content: 小说内容
    """
    pattern = r"第([1234567890]+)章 (.*?)+"

    # 查找所有章节标题的位置
    matches = list(re.finditer(pattern, content))

    chapters_path = f"tmp/{novel_name}/data"
    if not os.path.exists(chapters_path):
        os.makedirs(chapters_path)

    last_idx = None
    # 保存每一章的内容
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chapter_content = content[start:end].strip()
        title = chapter_content.split("\n")[0]

        pattern_num = r"第(\d+)章"
        match = re.search(pattern_num, title)
        if match:
            chapter_number = int(match.group(1))
        else:
            logging.error(f"Failed to extract chapter number from title: {title}")
            continue

        if last_idx is not None:
            if last_idx + 1 != chapter_number and chapter_number != 1:
                logging.warning(f"Chapter {last_idx + 1} is missing")
                logging.info(
                    f"Saved chapter {i+1} - '{title}' to {chapters_path}/chapter_{i+1}.txt"
                )
            last_idx = chapter_number
        else:
            last_idx = chapter_number

        if chapter_content:
            with open(
                f"{chapters_path}/chapter_{i+1}.txt", "w", encoding="utf-8"
            ) as file:
                file.write(chapter_content)


def process_novel(novel_name):
    """
    主处理函数
    :param novel_name: 小说名
    """
    content = read_novel(novel_name)
    if content:
        split_novel(novel_name, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小说 txt 拆分")
    parser.add_argument("book_path", type=str, help="小说绝对路径")

    args = parser.parse_args()

    book_path = args.book_path

    if book_path:
        # 获取小说名
        novel_name = os.path.basename(book_path).split(".")[0]

        # 移动小说到 tmp/{novel_name}/{novel_name}.txt
        tmp_dir = os.path.join("tmp", novel_name)
        os.makedirs(tmp_dir, exist_ok=True)
        new_path = os.path.join(tmp_dir, f"{novel_name}.txt")
        shutil.move(book_path, new_path)

        process_novel(novel_name)
    else:
        logging.error("请输入小说绝对路径")
