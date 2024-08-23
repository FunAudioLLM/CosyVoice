import os
import json
import random


def gen_role(role_json_path):
    # 读取role.json文件，处理可能存在的UTF-8 BOM
    with open(role_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    role_map = {}

    # 遍历所有角色并保存数据到role_map
    for role_category, role_list in data.items():
        if isinstance(role_list, list):
            for role in role_list:
                role_name = role["name"]
                if role_name not in role_map:
                    role_map[role_name] = {}
                role_map[role_name]["role"] = role_category
        else:  # 针对女主和男主这种单独的角色
            role_name = role_list["name"]
            if role_name not in role_map:
                role_map[role_name] = {}
            role_map[role_name]["role"] = role_category

    # 或者保存到一个新的JSON文件中
    with open("role_map.json", "w", encoding="utf-8") as f:
        json.dump(role_map, f, ensure_ascii=False, indent=4)

    return role_map


def gen_role_by_gender(gender, role_map):
    if gender == "女":
        female_roles = [key for key, role in role_map.items() if role["role"] == "女配"]
        if female_roles:
            return random.choice(female_roles)
        else:
            print("No female roles available.")
    else:
        male_roles = [key for key, role in role_map.items() if role["role"] == "男配"]
        if male_roles:
            return random.choice(male_roles)
        else:
            print("No male roles available.")


def main(base_dir, book_name):
    role_json_path = os.path.join(base_dir, "model/role.json")
    role_map = gen_role(role_json_path)

    bookname2role_path = os.path.join(base_dir, f"tmp/{book_name}/bookname2role.json")

    # 检查文件是否存在
    if not os.path.exists(bookname2role_path):
        # 如果文件不存在则创建一个空的字典
        data = {}
        # 创建必要的目录结构
        os.makedirs(os.path.dirname(bookname2role_path), exist_ok=True)
        # 将空字典写入文件
        with open(bookname2role_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        # 如果文件存在则读取其内容
        with open(bookname2role_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(data)

    # 读取 book_name 目录下所有 chapter_{idx}.json 文件，按照 idx 升序排列
    role_directory_path = os.path.join(base_dir, f"tmp/{book_name}/role")

    # 如果目录不存在则创建
    os.makedirs(role_directory_path, exist_ok=True)

    # 获取目录下所有文件
    files = [
        f
        for f in os.listdir(role_directory_path)
        if f.startswith("chapter_") and f.endswith(".json")
    ]

    # 按照 idx 升序排序
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # 读取所有文件内容并存储在列表中
    for file in files:
        file_path = os.path.join(role_directory_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                for item in content:
                    for key, value in item.items():
                        if data.get(value.get("role", "")) is None:
                            role_name = gen_role_by_gender(
                                value.get("gender", "男"), role_map
                            )
                            data[value["role"]] = role_name

    # 将数据写入文件
    with open(bookname2role_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 调用主函数
main("/root/code/CosyVoice", "诡秘之主")
