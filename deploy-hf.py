import logging
import subprocess
logging.basicConfig(level=logging.INFO)

def run_shell_script(script_path):
    """
    运行指定路径的shell脚本，并打印输出到控制台。

    :param script_path: Shell脚本的文件路径
    """
    try:
        # 使用subprocess.Popen来运行shell脚本
        with subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            # 读取输出
            for line in proc.stdout:
                print(line, end='')  # 实时打印输出
            proc.stdout.close()
            return_code = proc.wait()
            if return_code:
                print(f"Shell脚本运行出错，返回码：{return_code}")
    except Exception as e:
        print(f"运行shell脚本时发生错误：{e}")

# 使用方法示例
# 假设有一个名为example.sh的脚本文件在当前目录下
run_shell_script('deploy.sh')
 # SDK模型下载
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')

class Args:
    def __init__(self):
        self.port = 5000
        self.model_dir = 'pretrained_models/CosyVoice-300M'

from webui import main
from cosyvoice.cli.cosyvoice import CosyVoice
import numpy as np

# 创建 args 实例
args = Args()

cosyvoice = CosyVoice(args.model_dir)
sft_spk = cosyvoice.list_avaliable_spks()
prompt_sr, target_sr = 16000, 22050
default_data = np.zeros(target_sr)

# 调用 main 时传递 args
main(args,sft_spk)