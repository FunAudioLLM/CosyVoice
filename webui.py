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
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import subprocess
import tempfile
from datetime import datetime
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def stream_audio(generator):
    """流式输出音频"""
    for i in generator:
        audio_data = i['tts_speech']
        # 确保音频数据是 numpy array
        if torch.is_tensor(audio_data):
            audio_data = audio_data.numpy()
        audio_data = audio_data.flatten()
        
        # 进行音频后处理（如果需要）
        if np.abs(audio_data).max() > max_val:
            audio_data = audio_data / np.abs(audio_data).max() * max_val
            
        yield (cosyvoice.sample_rate, audio_data)


def save_complete_audio(generator, temp_dir):
    """保存并合并完整音频"""
    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = f'output_{timestamp}.wav'
    
    # 收集所有音频片段
    for i in generator:
        audio_data = i['tts_speech'].numpy().flatten()
        temp_file = os.path.join(temp_dir, f'segment_{len(output_files)}.wav')
        torchaudio.save(temp_file, torch.tensor(audio_data).unsqueeze(0), cosyvoice.sample_rate)
        output_files.append(temp_file)
    
    if output_files:
        try:
            # 创建合并列表
            concat_list = os.path.join(temp_dir, 'concat_list.txt')
            with open(concat_list, 'w') as f:
                for file in output_files:
                    f.write(f"file '{file}'\n")
            
            # 使用ffmpeg合并
            subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list,
                '-c', 'copy',
                final_output
            ], check=True)
            
            return final_output
            
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e}")
            return None
        finally:
            shutil.rmtree(temp_dir)

def get_prompt_wav(prompt_wav_upload, prompt_wav_record):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    return prompt_wav


def generate_audio_stream(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, 
                         prompt_wav_upload, prompt_wav_record, instruct_text,
                         seed, stream, speed):
    prompt_wav = get_prompt_wav(prompt_wav_upload, prompt_wav_record)
    """处理流式音频输出"""
    if mode_checkbox_group == '预训练音色':
        set_all_random_seed(seed)
        generator = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=True, speed=speed)
        for sample_rate, audio_data in stream_audio(generator):
            yield (sample_rate, audio_data)
    elif mode_checkbox_group == '3s极速复刻':
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=True, speed=speed)
        for sample_rate, audio_data in stream_audio(generator):
            yield (sample_rate, audio_data)
    elif mode_checkbox_group == '跨语种复刻':
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        generator = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=True, speed=speed)
        for sample_rate, audio_data in stream_audio(generator):
            yield (sample_rate, audio_data)
    else:
        set_all_random_seed(seed)
        generator = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=True, speed=speed)
        for sample_rate, audio_data in stream_audio(generator):
            yield (sample_rate, audio_data)


def generate_audio_complete(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, 
                          prompt_wav_upload, prompt_wav_record, instruct_text,
                          seed, stream, speed):
    prompt_wav = get_prompt_wav(prompt_wav_upload, prompt_wav_record)
    """处理完整音频文件"""
    temp_dir = tempfile.mkdtemp()
    if mode_checkbox_group == '预训练音色':
        set_all_random_seed(seed)
        generator = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=False, speed=speed)
        return save_complete_audio(generator, temp_dir)
    elif mode_checkbox_group == '3s极速复刻':
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=speed)
        return save_complete_audio(generator, temp_dir)
    elif mode_checkbox_group == '跨语种复刻':
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        generator = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False, speed=speed)
        return save_complete_audio(generator, temp_dir)
    else:
        set_all_random_seed(seed)
        generator = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=False, speed=speed)
        return save_complete_audio(generator, temp_dir)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0] if len(sft_spk) != 0 else '', scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        generate_button = gr.Button("生成音频")

        # 分离流式输出和文件下载
        with gr.Row():
            audio_stream = gr.Audio(
                label="实时预览", 
                autoplay=True,
                streaming=True
            )
            audio_download = gr.File(
                label="下载完整音频"
            )

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            fn=generate_audio_stream,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, 
                   prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed],
            outputs=audio_stream,
            api_name="generate_stream"
        ).then(
            fn=generate_audio_complete,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, 
                   prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed],
            outputs=audio_download,
            api_name="generate_complete"
        )
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice2(args.model_dir) if 'CosyVoice2' in args.model_dir else CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
