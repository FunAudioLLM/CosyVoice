import os
import re
import requests
import json
import logging
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义请求的URL和数据负载
url = "http://localhost:11434/api/generate"


def make_post_request(url, data):
    """
    发送POST请求并处理可能的异常
    """
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 对于错误状态码抛出异常
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"请求失败: {e}")
        return None


def parse_response(response):
    """
    解析响应并处理JSON解析错误
    """
    try:
        response_json = response.json()
        return response_json.get("response", {})
    except json.JSONDecodeError:
        logger.error("响应内容不是有效的JSON格式。")
        return None


def main(prompt):
    """
    主函数，处理请求和响应
    """
    data = {
        "model": "shareai/llama3.1-dpo-zh",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "5s",
    }

    response = make_post_request(url, data)
    if response:
        response_content = parse_response(response)
        if response_content:
            try:
                res = json.loads(response_content)
                return res
            except json.JSONDecodeError:
                logger.error("解析响应内容失败。")
                return None
        else:
            logger.info("解析响应内容失败。")
    else:
        logger.info("请求失败。")


def process_file(file_path, output_dir):
    """
    处理单个文件，提取对话并分析说话者和性别
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    results = []

    for idx in range(len(lines)):
        line = lines[idx]
        matches = re.findall(r"“([^”]*)”", line)
        if matches:
            for sentence in matches:
                sentence = sentence.strip()
                for attempt in range(5):
                    prev_start = max(0, idx - 15)
                    next_end = min(len(lines), idx + 16)
                    prev_lines = lines[prev_start:idx]
                    next_lines = lines[idx + 1 : next_end]
                    t_content = "\n".join(prev_lines + [line] + next_lines)

                    prompt = f"""
【原文】：
{t_content}

【要求】
1. 分析{sentence}，这句话原文中的说话者是谁？性别是什么？
2. 若无法分析说话者，返回未知
3. 若无法分析性别，返回未知
4. 仅返回 json 结构体，格式如下：
{{
    "role": "小杰",
    "gender": "男"
}}
"""
                    response_content = main(prompt)
                    if response_content:
                        result = {
                            sentence: {
                                "role": response_content["role"],
                                "gender": response_content["gender"],
                            }
                        }
                        results.append(result)
                        logger.info(
                            f"Processed sentence: {sentence}, role: {response_content['role']}, gender: {response_content['gender']}"
                        )
                        break
                    else:
                        logger.info(
                            f"Attempt {attempt + 1} failed to process sentence: {sentence}"
                        )
                else:
                    logger.info(
                        f"Failed to process sentence: {sentence} after 5 attempts"
                    )

    output_file = os.path.join(
        output_dir, os.path.basename(file_path).replace(".txt", ".json")
    )
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=4)


def process_directory(input_dir, output_dir, idx):
    """
    处理目录中的所有txt文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_files = [
        f for f in os.listdir(input_dir) if f.endswith(".txt") and not f.startswith(".")
    ]
    txt_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # 按 idx 排序

    for txt_file in tqdm(txt_files, desc="Processing files"):
        file_idx = int(txt_file.split("_")[1].split(".")[0])
        if file_idx >= idx:
            file_path = os.path.join(input_dir, txt_file)
            process_file(file_path, output_dir)


if __name__ == "__main__":
    input_dir = "/root/code/CosyVoice/tmp/苟在仙界成大佬/data"
    output_dir = "/root/code/CosyVoice/tmp/苟在仙界成大佬/role"
    idx = 1  # 从 chapter_1.txt 开始处理
    process_directory(input_dir, output_dir, idx)
