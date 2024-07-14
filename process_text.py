import os
import re


def read_novel(novel_name):
    """
    读取小说内容并进行预处理
    :param novel_name: 小说名
    :return: 预处理后的小说内容
    """
    novel_path = f"tmp/{novel_name}/{novel_name}.txt"
    with open(novel_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 简单清洗文本
    content = content.replace("\r\n", "\n").strip()
    return content


def split_novel(novel_name, content):
    """
    拆分小说为每一章并保存到文件
    :param novel_name: 小说名
    :param content: 小说内容
    """
    # 定义匹配章节标题的正则表达式
    # pattern = r"(第\s*\d+\s*章\s*.*?\n)"
    pattern = r"第\s*\S+\s*章\s*.*?\n"

    # 查找所有章节标题的位置
    matches = list(re.finditer(pattern, content))

    chapters_path = f"tmp/{novel_name}/data"
    if not os.path.exists(chapters_path):
        os.makedirs(chapters_path)

    # 保存每一章的内容
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chapter_title = match.group(0)
        chapter_content = content[start:end].strip()

        if chapter_content:
            with open(
                f"{chapters_path}/chapter_{i + 1}.txt", "w", encoding="utf-8"
            ) as file:
                file.write(chapter_content)


def load_progress(novel_name):
    """
    加载处理进度
    :param novel_name: 小说名
    :return: 已处理到的章节号
    """
    progress_path = f"tmp/{novel_name}/process.txt"
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as file:
            return int(file.read().strip())
    return 0


def save_progress(novel_name, chapter_number):
    """
    保存处理进度
    :param novel_name: 小说名
    :param chapter_number: 已处理到的章节号
    """
    progress_path = f"tmp/{novel_name}/process.txt"
    with open(progress_path, "w", encoding="utf-8") as file:
        file.write(str(chapter_number))


def process_novel(novel_name):
    """
    主处理函数
    :param novel_name: 小说名
    """
    content = read_novel(novel_name)
    split_novel(novel_name, content)
    processed_chapter = load_progress(novel_name)

    chapters_path = f"tmp/{novel_name}/data"
    chapter_files = sorted(os.listdir(chapters_path))

    for i, chapter_file in enumerate(chapter_files):
        if i >= processed_chapter:
            # 在这里处理每一章
            print(f"Processing {chapter_file}...")
            # 模拟处理
            with open(f"{chapters_path}/{chapter_file}", "r", encoding="utf-8") as file:
                chapter_content = file.read()

            # 处理完成后保存进度
            save_progress(novel_name, i + 1)


# 示例调用
novel_name = "fz"
process_novel(novel_name)
