import subprocess
import sys
import importlib.util
import torch

def check_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"\n[✅] 命令 `{cmd}` 执行成功:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"\n[❌] 命令 `{cmd}` 执行失败:\n{e.stderr}")

def check_module(module_name):
    return importlib.util.find_spec(module_name) is not None

def check_pytorch():
    print("\n🧪 PyTorch 检查：")
    if check_module("torch"):
        try:
            import torch
            print(f"[✅] 已安装 torch，版本：{torch.__version__}")
            print(torch.cuda.is_available())
            print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
        except Exception as e:
            print(f"[❌] 检查 torch 出错: {e}")
    else:
        print("[❌] 未安装 PyTorch（torch）模块")

def check_tensorflow():
    print("\n🧪 TensorFlow 检查：")

    if check_module("tensorflow"):
        try:
            import tensorflow as tf
            print(f"[✅] 已安装 TensorFlow，版本：{tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"[✅] 检测到 GPU：{gpus}")
            else:
                print("[❌] 未检测到 GPU，请检查 CUDA 驱动或安装是否为 GPU 版本")
        except Exception as e:
            print(f"[❌] 检查 tensorflow 出错: {e}")
    else:
        print("[❌] 未安装 TensorFlow 模块")

def main():
    print("==== CUDA / 驱动 / 深度学习环境 检查工具 ====")
    check_command("nvidia-smi")
    check_pytorch()
    check_tensorflow()

if __name__ == "__main__":
    main()
