from http import HTTPStatus
import os
import sys

import dashscope
from openai import OpenAI


# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import requests
import json
import logging
from tqdm import tqdm
from tools.zh_normalization.text_normlization import TextNormalizer


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
        return (
            response_json.get("response", {})
            .replace("```", "")
            .replace("json", "")
            .strip()
        )
    except json.JSONDecodeError:
        logger.error("响应内容不是有效的JSON格式。")
        return None


def main(prompt):
    """
    主函数，处理请求和响应
    """
    data = {
        "model": "wangshenzhi/gemma2-9b-chinese-chat:latest",
        # "model": "llama3_1_shenzhi:latest",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "5s",
        # "options": {"seed": 42},
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


def call_with_messages(msg):
    client = OpenAI(
        api_key="sk-79ddff7089fe4363917e5c143c44bc0b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="llama3.1-405b-instruct",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": msg},
        ],
        temperature=0.8,
    )

    content = (
        completion.choices[0].message.content.replace("```", "").replace("json", "")
    )
    print(content)
    return json.decoder.JSONDecoder().decode(content)


def get_top_books(bookname, current_chapter, top):
    role_data = []

    # 1. 获取 last_chapter+3>=current_chapter 的所有角色名
    for role_name, data in bookname.items():
        if data["last_chapter"] + 3 >= current_chapter:
            role_data.append((role_name, data["count"]))

    # 去重
    seen_roles = set()
    filtered_roles = []
    for role_name, count in role_data:
        if role_name not in seen_roles:
            seen_roles.add(role_name)
            filtered_roles.append((role_name, count))

    # 若不满top，再从 count 从大到小排序，补齐剩下的缺失角色
    if len(filtered_roles) < top:
        remaining_roles = [
            (role_name, data["count"])
            for role_name, data in sorted(
                bookname.items(), key=lambda item: item[1]["count"], reverse=True
            )
            if role_name not in seen_roles
        ]
        filtered_roles.extend(remaining_roles[: top - len(filtered_roles)])

    # 最终确保 top
    top_roles = filtered_roles[:top]

    # 生成结果字符串
    topbookname = ",".join([role_name for role_name, count in top_roles])

    return topbookname


def process_file(file_path, output_dir, bookname_dir, current_chapter):
    """
    处理单个文件，提取对话并分析说话者和性别
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 读取 bookname.json, 若不存在则创建
    if not os.path.exists(os.path.join(bookname_dir, "bookname.json")):
        with open(
            os.path.join(bookname_dir, "bookname.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    with open(os.path.join(bookname_dir, "bookname.json"), "r", encoding="utf-8") as f:
        bookname = json.load(f)

    lines = content.split("\n")
    results = []

    for idx in range(len(lines)):
        line = lines[idx]
        matches = re.findall(r"“([^”]*)”", line)
        if matches:
            for sentence in matches:
                sentence = sentence.strip()
                # 文本格式化
                tx = TextNormalizer()
                sentences = tx.normalize(sentence)
                sentence = "".join(sentences)
                for attempt in range(5):
                    prev_start = max(0, idx - 7)
                    next_end = min(len(lines), idx + 8)
                    prev_lines = lines[prev_start:idx]
                    next_lines = lines[idx + 1 : next_end]
                    t_content = "".join(prev_lines + [line] + next_lines)

                    topbookname = get_top_books(bookname, current_chapter, 50)
                    prompt = f"""
【原文】：
{t_content}

【任务描述】
根据给定的句子和上下文内容，判断句子是否为某个角色所说，并确定该角色的姓名和性别。请按照以下步骤执行：

1. 角色判断：分析句子【{sentence}】及其上下文，判断该句子是否由某个角色说出。
    如果是，请继续下一步。
    如果不是，请返回 "role": "未知", "gender": "未知"。

2. 角色匹配：
    2.1 优先从提供的角色姓名列表【{topbookname}】中查找匹配的角色姓名。
    2.2 如果在列表中找不到完全匹配的姓名，请根据上下文推测可能的角色姓名。
    2.3 如果推测的角色姓名与列表中的某个姓名相似（例如拼写接近），请使用列表中的姓名。

3. 姓名验证：
    如果推测出的角色姓名包含标点符号、非字符或不符合常规姓名格式，请返回 "role": "未知", "gender": "未知"。

4. 性别判定：
    4.1 根据上下文判断角色的性别（仅限 "男"、"女" 或 "未知"）。
    4.2 如果无法确定性别，请返回 "gender": "未知"。

5. 输出格式：
    结果需以以下 JSON 格式返回：
{{
    {{
        "role": "角色姓名",
        "gender": "男/女/未知"
    }}
}}
"""
                    # print(prompt)
                    # response_content = call_with_messages(prompt)
                    response_content = main(prompt)
                    if (
                        response_content
                        and "role" in response_content
                        and "gender" in response_content
                    ):
                        role_name = response_content["role"]
                        result = {
                            sentence: {
                                "role": role_name,
                                "gender": response_content["gender"],
                            }
                        }
                        if role_name != "" and role_name != "未知":
                            if role_name in bookname:
                                bookname[role_name]["count"] += 1
                                bookname[role_name]["last_chapter"] = current_chapter
                            else:
                                bookname[role_name] = {
                                    "count": 1,
                                    "last_chapter": current_chapter,
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

    bookname_file = os.path.join(bookname_dir, "bookname.json")
    with open(bookname_file, "w", encoding="utf-8") as out_f:
        json.dump(bookname, out_f, ensure_ascii=False, indent=4)


def process_directory(input_dir, output_dir, bookname_dir, idx):
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
            process_file(file_path, output_dir, bookname_dir, file_idx)


if __name__ == "__main__":
    input_dir = "/root/code/CosyVoice/tmp/诡秘之主/data"
    output_dir = "/root/code/CosyVoice/tmp/诡秘之主/role"
    bookname_dir = "/root/code/CosyVoice/tmp/诡秘之主"
    idx = 828  # 从 chapter_1.txt 开始处理
    process_directory(input_dir, output_dir, bookname_dir, idx)
