# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的库
import os
import sys
import argparse  # 用于解析命令行参数
import gradio as gr  # Gradio库用于创建Web界面
import numpy as np
import torch
import torchaudio
import random
import librosa  # 音频处理库

# 设置根目录并添加第三方库路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

# 导入CosyVoice相关模块
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# 定义推理模式列表
inference_mode_list = ["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"]

# 为每种推理模式定义操作指南
instruct_dict = {
    "预训练音色": "1. 选择预训练音色\n2. 点击生成音频按钮",
    "3s极速复刻": "1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮",
    "跨语种复刻": "1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮",
    "自然语言控制": "1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮",
}

# 流式推理模式选项
stream_mode_list = [("否", False), ("是", True)]

# 音频最大值限制，用于音频归一化
max_val = 0.8


# 生成随机种子函数
def generate_seed():
    """
    生成一个随机种子，用于模型推理
    返回格式适配Gradio的更新机制
    """
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


# 音频后处理函数
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """
    对生成的音频进行后处理:
    1. 去除静音部分
    2. 音量归一化
    3. 添加尾部静音
    
    参数:
    - speech: 音频数据
    - top_db: 静音检测的分贝阈值
    - hop_length: 帧移
    - win_length: 窗长
    
    返回:
    - 处理后的音频
    """
    # 去除静音部分
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    # 音量归一化
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    # 添加尾部0.2秒静音
    speech = torch.concat(
        [speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1
    )
    return speech


# 更新操作指南函数
def change_instruction(mode_checkbox_group):
    """
    根据选择的推理模式更新操作指南
    
    参数:
    - mode_checkbox_group: 当前选择的推理模式
    
    返回:
    - 对应的操作指南文本
    """
    return instruct_dict[mode_checkbox_group]


# 生成音频的核心函数
def generate_audio(
    tts_text,                # 要合成的文本
    mode_checkbox_group,     # 推理模式
    sft_dropdown,            # 预训练音色选择
    prompt_text,             # prompt文本
    prompt_wav_upload,       # 上传的prompt音频
    prompt_wav_record,       # 录制的prompt音频
    instruct_text,           # 指令文本
    seed,                    # 随机种子
    stream,                  # 是否流式推理
    speed,                   # 语速调节
):
    """
    根据用户输入生成音频的主函数
    
    处理流程:
    1. 确定prompt音频来源
    2. 根据不同推理模式进行参数验证
    3. 调用相应的推理方法生成音频
    
    返回:
    - 生成的音频数据(流式)
    """
    # 确定prompt音频来源，优先使用上传的音频
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
        
    # 自然语言控制模式的参数验证
    if mode_checkbox_group in ["自然语言控制"]:
        # 检查模型是否支持指令控制
        if cosyvoice.instruct is False:
            gr.Warning(
                "您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型".format(
                    args.model_dir
                )
            )
            yield (cosyvoice.sample_rate, default_data)
        # 检查是否提供了指令文本
        if instruct_text == "":
            gr.Warning("您正在使用自然语言控制模式, 请输入instruct文本")
            yield (cosyvoice.sample_rate, default_data)
        # 提示用户在此模式下prompt相关输入会被忽略
        if prompt_wav is not None or prompt_text != "":
            gr.Info("您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略")
            
    # 跨语种复刻模式的参数验证
    if mode_checkbox_group in ["跨语种复刻"]:
        # 检查模型是否支持跨语种复刻
        if cosyvoice.instruct is True:
            gr.Warning(
                "您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型".format(
                    args.model_dir
                )
            )
            yield (cosyvoice.sample_rate, default_data)
        # 提示用户在此模式下instruct文本会被忽略
        if instruct_text != "":
            gr.Info("您正在使用跨语种复刻模式, instruct文本会被忽略")
        # 检查是否提供了prompt音频
        if prompt_wav is None:
            gr.Warning("您正在使用跨语种复刻模式, 请提供prompt音频")
            yield (cosyvoice.sample_rate, default_data)
        gr.Info("您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言")
        
    # 3s极速复刻和跨语种复刻模式的共同参数验证
    if mode_checkbox_group in ["3s极速复刻", "跨语种复刻"]:
        # 检查是否提供了prompt音频
        if prompt_wav is None:
            gr.Warning("prompt音频为空，您是否忘记输入prompt音频？")
            yield (cosyvoice.sample_rate, default_data)
        # 检查prompt音频采样率
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(
                "prompt音频采样率{}低于{}".format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr
                )
            )
            yield (cosyvoice.sample_rate, default_data)
            
    # 预训练音色模式的参数验证
    if mode_checkbox_group in ["预训练音色"]:
        # 提示用户在此模式下其他输入会被忽略
        if instruct_text != "" or prompt_wav is not None or prompt_text != "":
            gr.Info(
                "您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！"
            )
        # 检查是否有可用的预训练音色
        if sft_dropdown == "":
            gr.Warning("没有可用的预训练音色！")
            yield (cosyvoice.sample_rate, default_data)
            
    # 3s极速复刻模式的参数验证
    if mode_checkbox_group in ["3s极速复刻"]:
        # 检查是否提供了prompt文本
        if prompt_text == "":
            gr.Warning("prompt文本为空，您是否忘记输入prompt文本？")
            yield (cosyvoice.sample_rate, default_data)
        # 提示用户在此模式下其他输入会被忽略
        if instruct_text != "":
            gr.Info("您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！")

    # 根据不同推理模式调用相应的推理方法
    if mode_checkbox_group == "预训练音色":
        # 预训练音色模式
        logging.info("get sft inference request")
        set_all_random_seed(seed)  # 设置随机种子
        # 调用预训练音色推理方法，并流式返回结果
        # i["tts_speech"]是一个PyTorch张量,形状为[1, T],其中T是音频长度
        # .numpy()将PyTorch张量转换为NumPy数组,保持形状为[1, T]
        # .flatten()将[1, T]展平为一维数组[T],这是音频播放所需的格式
        for i in cosyvoice.inference_sft(
            tts_text, sft_dropdown, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "3s极速复刻":
        # 3s极速复刻模式
        logging.info("get zero_shot inference request")
        # 处理prompt音频
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)  # 设置随机种子
        # 调用零样本推理方法，并流式返回结果
        for i in cosyvoice.inference_zero_shot(
            tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "跨语种复刻":
        # 跨语种复刻模式
        logging.info("get cross_lingual inference request")
        # 处理prompt音频
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)  # 设置随机种子
        # 调用跨语种推理方法，并流式返回结果
        for i in cosyvoice.inference_cross_lingual(
            tts_text, prompt_speech_16k, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())
    else:
        # 自然语言控制模式
        logging.info("get instruct inference request")
        set_all_random_seed(seed)  # 设置随机种子
        # 调用指令控制推理方法，并流式返回结果
        for i in cosyvoice.inference_instruct(
            tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed
        ):
            yield (cosyvoice.sample_rate, i["tts_speech"].numpy().flatten())


# 主函数，构建Web界面
def main():
    """
    构建Gradio Web界面并启动服务
    """
    with gr.Blocks() as demo:
        # 标题和项目链接
        gr.Markdown(
            "### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)"
        )
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        # 合成文本输入框
        tts_text = gr.Textbox(
            label="输入合成文本",
            lines=1,
            value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
        )
        
        # 推理模式和参数设置区域
        with gr.Row():
            # 推理模式选择
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label="选择推理模式",
                value=inference_mode_list[0],
            )
            # 操作步骤说明
            instruction_text = gr.Text(
                label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5
            )
            # 预训练音色选择
            sft_dropdown = gr.Dropdown(
                choices=sft_spk, label="选择预训练音色", value=sft_spk[0], scale=0.25
            )
            # 流式推理选择
            stream = gr.Radio(
                choices=stream_mode_list,
                label="是否流式推理",
                value=stream_mode_list[0][1],
            )
            # 语速调节
            speed = gr.Number(
                value=1,
                label="速度调节(仅支持非流式推理)",
                minimum=0.5,
                maximum=2.0,
                step=0.1,
            )
            # 随机种子设置
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")  # 骰子图标
                seed = gr.Number(value=0, label="随机推理种子")

        # Prompt音频输入区域
        with gr.Row():
            # 上传音频文件
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="选择prompt音频文件，注意采样率不低于16khz",
            )
            # 录制音频
            prompt_wav_record = gr.Audio(
                sources="microphone", type="filepath", label="录制prompt音频文件"
            )
            
        # Prompt文本输入框
        prompt_text = gr.Textbox(
            label="输入prompt文本",
            lines=1,
            placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...",
            value="",
        )
        
        # 指令文本输入框
        instruct_text = gr.Textbox(
            label="输入instruct文本",
            lines=1,
            placeholder="请输入instruct文本.",
            value="",
        )

        # 生成按钮
        generate_button = gr.Button("生成音频")

        # 音频输出组件
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        # 事件绑定
        # 点击随机种子按钮时生成新的随机种子
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        
        # 点击生成按钮时调用generate_audio函数
        generate_button.click(
            generate_audio,
            inputs=[
                tts_text,
                mode_checkbox_group,
                sft_dropdown,
                prompt_text,
                prompt_wav_upload,
                prompt_wav_record,
                instruct_text,
                seed,
                stream,
                speed,
            ],
            outputs=[audio_output],
        )
        
        # 当推理模式改变时更新操作指南
        mode_checkbox_group.change(
            fn=change_instruction,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text],
        )
        
    # 设置队列，限制并发请求数
    demo.queue(max_size=4, default_concurrency_limit=2)
    # 启动服务
    demo.launch(server_name="0.0.0.0", server_port=args.port)


# 程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)  # Web服务端口
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="local path or modelscope repo id",  # 模型路径或ModelScope仓库ID
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="是否使用半精度(fp16)推理"
    )
    args = parser.parse_args()
    
    # 尝试加载CosyVoice模型
    try:
        cosyvoice = CosyVoice(args.model_dir, fp16=args.fp16)
    except Exception:
        # 如果加载失败，尝试加载CosyVoice2模型
        try:
            cosyvoice = CosyVoice2(args.model_dir, fp16=args.fp16)
        except Exception:
            raise TypeError("no valid model_type!")  # 如果都失败，抛出错误

    # 获取可用的预训练音色列表
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = [""]  # 如果没有可用音色，设置为空列表
        
    # 设置prompt音频采样率
    prompt_sr = 16000
    # 默认音频数据（空音频）
    default_data = np.zeros(cosyvoice.sample_rate)
    
    # 启动主函数
    main()
