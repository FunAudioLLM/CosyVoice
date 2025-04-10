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
    # 必须在导入torch和创建任何模型之前设置CUDA_VISIBLE_DEVICES
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import platform
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from cosyvoice.utils.file_utils import load_wav, logging
import shutil
import time

# # 设置日志级别为 DEBUG
# logging.basicConfig(level=logging.DEBUG, 
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# # 确保设置影响所有模块
# for name in logging.root.manager.loggerDict:
#     logging.getLogger(name).setLevel(logging.DEBUG)



# 设置环境变量禁用tokenizers并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.common import set_all_random_seed




# from modelscope import snapshot_download
# snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
try:
    shutil.copy2('spk2info.pt', 'pretrained_models/CosyVoice2-0.5B/spk2info.pt')
except Exception as e:
    logging.warning(f'复制文件失败: {e}')

inference_mode_list = ['预训练音色', '自然语言控制', '3s极速复刻', '跨语种复刻']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮\n4. (可选)保存音色模型',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮\n3. (可选)保存音色模型',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

def refresh_sft_spk():
    """刷新音色选择列表 """
    # 获取自定义音色
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{ROOT_DIR}/voices")]
    files.sort(key=lambda x: x[1], reverse=True) # 按时间排序

    # 添加预训练音色
    choices = [f[0].replace(".pt", "") for f in files] + cosyvoice.list_available_spks()

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}


def refresh_prompt_wav():
    """刷新音频选择列表"""
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{ROOT_DIR}/audios")]
    files.sort(key=lambda x: x[1], reverse=True)  # 按时间排序
    choices = ["请选择参考音频或者自己上传"] + [f[0] for f in files]

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}


def change_prompt_wav(filename):
    """切换音频文件"""
    full_path = f"{ROOT_DIR}/audios/{filename}"
    if not os.path.exists(full_path):
        logging.warning(f"音频文件不存在: {full_path}")
        return None

    return full_path

def save_voice_model(voice_name):
    """保存音色模型"""
    if not voice_name:
        gr.Info("音色名称不能为空")
        return False
        
    try:
        shutil.copyfile(f"{ROOT_DIR}/output.pt", f"{ROOT_DIR}/voices/{voice_name}.pt")
        gr.Info("音色保存成功,存放位置为voices目录")
        return True
    except Exception as e:
        logging.error(f"保存音色失败: {e}")
        gr.Warning("保存音色失败")
        return False

def generate_random_seed():
    """生成随机种子"""
    return {
        "__type__": "update",
        "value": random.randint(1, 100000000)
    }

def postprocess(speech, top_db = 60, hop_length = 220, win_length = 440):
    """音频后处理方法"""
    # 修剪静音部分
    speech, _ = librosa.effects.trim(
        speech, 
        top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    # 音量归一化
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val

    # 添加尾部静音
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    """切换模式的处理"""
    voice_dropdown_visible = mode_checkbox_group in ['预训练音色', '自然语言控制']
    save_btn_visible = mode_checkbox_group in ['3s极速复刻']
    return (
        instruct_dict[mode_checkbox_group],
        gr.update(visible=voice_dropdown_visible),
        gr.update(visible=save_btn_visible)
    )

def prompt_wav_recognition(prompt_wav):
    """音频识别文本"""
    if prompt_wav is None:
        return ''
        
    try:
        res = asr_model.generate(
            input=prompt_wav,
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
        )
        return res[0]["text"].split('|>')[-1]
    except Exception as e:
        logging.error(f"音频识别文本失败: {e}")
        gr.Warning("识别文本失败，请检查音频是否包含人声内容")
        return ''

def load_voice_data(voice_path):
    """加载音色数据"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        voice_data = torch.load(voice_path, map_location=device) if os.path.exists(voice_path) else None
        return voice_data.get('audio_ref') if voice_data else None
    except Exception as e:
        logging.error(f"加载音色文件失败: {e}")
        return None

def validate_input(mode, tts_text, sft_dropdown, prompt_text, prompt_wav, instruct_text):
    """验证输入参数的合法性
    
    Args:
        mode: 推理模式
        tts_text: 合成文本
        sft_dropdown: 预训练音色
        prompt_text: prompt文本
        prompt_wav: prompt音频
        instruct_text: instruct文本
    
    Returns:
        bool: 验证是否通过
        str: 错误信息
    """
    if mode in ['自然语言控制']:
        if not cosyvoice.is_05b and cosyvoice.instruct is False:
            return False, f'您正在使用自然语言控制模式, {args.model_dir}模型不支持此模式'
        if not instruct_text:
            return False, '您正在使用自然语言控制模式, 请输入instruct文本'
            
    elif mode in ['跨语种复刻']:
        if not cosyvoice.is_05b and cosyvoice.instruct is True:
            return False, f'您正在使用跨语种复刻模式, {args.model_dir}模型不支持此模式'
        if not prompt_wav:
            return False, '您正在使用跨语种复刻模式, 请提供prompt音频'
            
    elif mode in ['3s极速复刻', '跨语种复刻']:
        if not prompt_wav:
            return False, 'prompt音频为空，您是否忘记输入prompt音频？'
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            return False, f'prompt音频采样率{torchaudio.info(prompt_wav).sample_rate}低于{prompt_sr}'
            
    elif mode in ['预训练音色']:
        if not sft_dropdown:
            return False, '没有可用的预训练音色！'
            
    if mode in ['3s极速复刻'] and not prompt_text:
        return False, 'prompt文本为空，您是否忘记输入prompt文本？'

    return True, ''

def process_audio(speech_generator, stream):
    """处理音频生成
    
    Args:
        speech_generator: 音频生成器
        stream: 是否流式处理
    
    Returns:
        tuple: (音频数据列表, 总时长)
    """
    tts_speeches = []
    total_duration = 0
    for i in speech_generator:
        tts_speeches.append(i['tts_speech'])
        total_duration += i['tts_speech'].shape[1] / cosyvoice.sample_rate
        if stream:
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None
            
    if not stream:
        audio_data = torch.concat(tts_speeches, dim=1)
        yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())
    
    yield total_duration

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    """生成音频的主函数
    
    Args:
        tts_text: 合成文本
        mode_checkbox_group: 推理模式 
        sft_dropdown: 预训练音色
        prompt_text: prompt文本
        prompt_wav_upload: 上传的prompt音频
        prompt_wav_record: 录制的prompt音频
        instruct_text: instruct文本
        seed: 随机种子
        stream: 是否流式推理
        speed: 语速
    
    Yields:
        tuple: 音频数据
    """
    start_time = time.time()
    logging.info(f"开始生成音频 - 模式: {mode_checkbox_group}, 文本长度: {len(tts_text)}")
    # 处理prompt音频输入
    prompt_wav = prompt_wav_upload if prompt_wav_upload is not None else prompt_wav_record

    # 验证输入
    is_valid, error_msg = validate_input(mode_checkbox_group, tts_text, sft_dropdown, 
                                       prompt_text, prompt_wav, instruct_text)
    if not is_valid:
        gr.Warning(error_msg)
        yield (cosyvoice.sample_rate, default_data), None
        return

    # 设置随机种子
    set_all_random_seed(seed)

    # 根据不同模式处理
    if mode_checkbox_group == '预训练音色':
        # logging.info('get sft inference request')
        generator = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed)
        
    elif mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        # logging.info(f'get {mode_checkbox_group} inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        inference_func = (cosyvoice.inference_zero_shot if mode_checkbox_group == '3s极速复刻' 
                         else cosyvoice.inference_cross_lingual)
        generator = inference_func(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)
        
    else:  # 自然语言控制模式
        # logging.info('get instruct inference request')
        voice_path = f"{ROOT_DIR}/voices/{sft_dropdown}.pt"
        prompt_speech_16k = load_voice_data(voice_path)
        
        if prompt_speech_16k is None:
            gr.Warning('预训练音色文件中缺少prompt_speech数据！')
            yield (cosyvoice.sample_rate, default_data), None
            return
            
        generator = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, 
                                                stream=stream, speed=speed)

    # 处理音频生成并获取总时长
    audio_generator = process_audio(generator, stream)
    total_duration = 0
    
    # 收集所有音频输出
    for output in audio_generator:
        if isinstance(output, (float, int)):  # 如果是总时长
            total_duration = output
        else:  # 如果是音频数据
            yield output

    processing_time = time.time() - start_time
    rtf = processing_time / total_duration if total_duration > 0 else 0
    logging.info(f"音频生成完成 耗时: {processing_time:.2f}秒, rtf: {rtf:.2f}")

def update_audio_visibility(stream_enabled):
    """更新音频组件的可见性"""
    return [
        gr.update(visible=stream_enabled),  # 流式音频组件
        gr.update(visible=not stream_enabled)  # 非流式音频组件
    ]

def main():
    with gr.Blocks() as demo:
        # 页面标题和说明
        gr.Markdown("### 代码库 [CosyVoice2-Ex](https://github.com/journey-ad/CosyVoice2-Ex) 原始项目 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        # 主要输入区域
        tts_text = gr.Textbox(
            label="输入合成文本", 
            lines=1, 
            value="CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities."
        )
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list, 
                label='选择推理模式', 
                value=inference_mode_list[0]
            )
            instruction_text = gr.Text(
                label="操作步骤", 
                value=instruct_dict[inference_mode_list[0]], 
                scale=0.5
            )
            
            # 音色选择部分
            sft_dropdown = gr.Dropdown(
                choices=sft_spk, 
                label='选择预训练音色', 
                value=sft_spk[0], 
                scale=0.25
            )
            refresh_voice_button = gr.Button("刷新音色")
            
            # 流式控制和速度调节
            with gr.Column(scale=0.25):
                stream = gr.Radio(
                    choices=stream_mode_list, 
                    label='是否流式推理', 
                    value=stream_mode_list[0][1]
                )
                speed = gr.Number(
                    value=1, 
                    label="速度调节(仅支持非流式推理)", 
                    minimum=0.5, 
                    maximum=2.0, 
                    step=0.1
                )
                
            # 随机种子控制
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        # 音频输入区域
        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources='upload', 
                type='filepath', 
                label='选择prompt音频文件，注意采样率不低于16khz'
            )
            prompt_wav_record = gr.Audio(
                sources='microphone', 
                type='filepath', 
                label='录制prompt音频文件'
            )
            wavs_dropdown = gr.Dropdown(
                label="参考音频列表", 
                choices=reference_wavs, 
                value="请选择参考音频或者自己上传", 
                interactive=True
            )
            refresh_button = gr.Button("刷新参考音频")

        # 文本输入区域
        prompt_text = gr.Textbox(
            label="输入prompt文本", 
            lines=1, 
            placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", 
            value=''
        )
        instruct_text = gr.Textbox(
            label="输入instruct文本", 
            lines=1, 
            placeholder="请输入instruct文本. 例如: 用四川话说这句话。", 
            value=''
        )

        # 保存音色按钮（默认隐藏）
        with gr.Row(visible=False) as save_spk_btn:
            new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value='', scale=2)
            save_button = gr.Button(value="保存音色模型", scale=1)

        # 生成按钮
        generate_button = gr.Button("生成音频")

        # 音频输出区域
        with gr.Group() as audio_group:
            audio_output_stream = gr.Audio(
                label="合成音频(流式)", 
                value=None,
                streaming=True,
                autoplay=True,
                show_label=True,
                show_download_button=True,
                visible=False
            )
            audio_output_normal = gr.Audio(
                label="合成音频",
                value=None, 
                streaming=False,
                autoplay=True,
                show_label=True,
                show_download_button=True,
                visible=True
            )

        # 绑定事件
        refresh_voice_button.click(fn=refresh_sft_spk, inputs=[], outputs=[sft_dropdown])
        refresh_button.click(fn=refresh_prompt_wav, inputs=[], outputs=[wavs_dropdown])
        wavs_dropdown.change(change_prompt_wav, inputs=[wavs_dropdown], outputs=[prompt_wav_upload])
        save_button.click(save_voice_model, inputs=[new_name])
        seed_button.click(generate_random_seed, inputs=[], outputs=[seed])
        
        generate_button.click(
            generate_audio,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, 
                   prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed],
            outputs=[audio_output_stream, audio_output_normal]
        )
        
        mode_checkbox_group.change(
            fn=change_instruction, 
            inputs=[mode_checkbox_group], 
            outputs=[instruction_text, sft_dropdown, save_spk_btn]
        )
        
        prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])
        
        stream.change(
            fn=update_audio_visibility,
            inputs=[stream],
            outputs=[audio_output_stream, audio_output_normal]
        )

    # 配置队列和启动服务
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port, inbrowser=args.open)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8080)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--open', 
                        action='store_true', 
                        help='open in browser')
    parser.add_argument('--log_level',
                        type=str,
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='set log level')
    args = parser.parse_args()

    
    logging.getLogger().setLevel(getattr(logging, args.log_level))


    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception as e:
        logging.warning(f"尝试加载CosyVoice模型失败: {e}")
        try:
            cosyvoice = CosyVoice2(args.model_dir, load_trt=True, fp16=True)
        except Exception as e:
            logging.error(f"尝试加载CosyVoice2模型也失败: {e}")
            raise TypeError('no valid model_type!')

            
    sft_spk = refresh_sft_spk()['choices']
    reference_wavs = refresh_prompt_wav()['choices']

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)

    model_dir = "iic/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        log_level=args.log_level,
        device="cuda:0")
    main()
