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
import datetime
import traceback
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import uvicorn
import uuid
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.ModelManager import ModelManager
from fastapi import FastAPI, File, UploadFile, Query, Body, Form
from fastapi.responses import Response, StreamingResponse, JSONResponse, PlainTextResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from contextlib import asynccontextmanager
from pydub import AudioSegment
from pydub.effects import normalize
from math import log10
from langdetect import detect

# 全局模型管理器
model_manager = ModelManager()
#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制', '语音复刻']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮',
                 '语音复刻': '1. 选择source音频文件\n2. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

def volume_safely(audio: AudioSegment, volume_multiplier: float = 1.0) -> AudioSegment:
    """
    安全地调整音频音量。
    :param audio: AudioSegment 对象，音频数据。
    :param volume_multiplier: float，音量倍数，1.0 为原音量，大于 1 提高音量，小于 1 降低音量。
    :return: 调整后的 AudioSegment 对象。
    """
    if volume_multiplier <= 0:
        raise ValueError("volume_multiplier 必须大于 0")

    # 计算增益（分贝），根据倍数调整
    gain_in_db = 20 * np.log10(volume_multiplier)

    # 应用增益调整音量
    audio = audio.apply_gain(gain_in_db)

    # 确保音频不削波（归一化到峰值 -0.1 dB 以下）
    audio = audio.normalize(headroom=0.1)

    return audio

def detect_language(text):
    lang = detect(text)
    logging.info(f'lang: {lang}')
    return lang
    
def generate_seed():
    seed = random.randint(1, 100000000)
    logging.info(f'seed: {seed}')
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
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def log_error(exception: Exception, log_dir='error'):
    """
    记录错误信息到指定目录，并按日期时间命名文件。

    :param exception: 捕获的异常对象
    :param log_dir: 错误日志存储的目录，默认为 'error'
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    # 获取当前时间戳，格式化为 YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 创建日志文件路径
    log_file_path = os.path.join(log_dir, f'error_{timestamp}.log')
    # 使用 traceback 模块获取详细的错误信息
    error_traceback = traceback.format_exc()
    # 写入错误信息到文件
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"错误发生时间: {timestamp}\n")
        log_file.write(f"错误信息: {str(exception)}\n")
        log_file.write("堆栈信息:\n")
        log_file.write(error_traceback + '\n')
    
    logging.info(f"错误信息已保存至: {log_file_path}")

# 定义一个函数进行显存清理
def clear_cuda_cache():
    torch.cuda.empty_cache()
    logging.info("CUDA cache cleared!")

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav, instruct_text,
                   seed, stream, speed, source_wav):
    errcode = 0
    errmsg = ''
    logging.info(f'prompt_wav: {prompt_wav}')
    logging.info(f'source_wav: {source_wav}')

    cosyvoice = None
    # 获取需要的模型
    if mode_checkbox_group == '预训练音色':
        cosyvoice = model_manager.get_model("cosyvoice_sft")
    elif mode_checkbox_group in ['3s极速复刻', '跨语种复刻', '语音复刻']:
        cosyvoice = model_manager.get_model("cosyvoice")
    else:
        cosyvoice = model_manager.get_model("cosyvoice_instruct")  

    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.frontend.instruct is False:
            errcode = 1
            errmsg = '您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir)
            return errcode, errmsg, (target_sr, default_data)
        
        if instruct_text == '':
            errcode = 2
            errmsg = '您正在使用自然语言控制模式, 请输入instruct文本'
            return errcode, errmsg, (target_sr, default_data)
        
        if prompt_wav is not None or prompt_text != '':
            logging.info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            errcode = 3
            errmsg = '您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir)
            return errcode, errmsg, (target_sr, default_data)
        
        if instruct_text != '':
            logging.info('您正在使用跨语种复刻模式, instruct文本会被忽略')

        if prompt_wav is None:
            errcode = 5
            errmsg = '您正在使用跨语种复刻模式, 请提供prompt音频'
            return errcode, errmsg, (target_sr, default_data)
        
        logging.info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻', '语音复刻']:
        if prompt_wav is None:
            errcode = 6
            errmsg = 'prompt音频为空，您是否忘记输入prompt音频？'
            return errcode, errmsg, (target_sr, default_data)
        
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            errcode = 7
            errmsg = 'prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr)
            return errcode, errmsg, (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            logging.info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            errcode = 8
            errmsg = 'prompt文本为空，您是否忘记输入prompt文本？'
            return errcode, errmsg, (target_sr, default_data)
        
        if instruct_text != '':
            logging.info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group in ['语音复刻']:
        if source_wav is None:
            errcode = 6
            errmsg = 'source音频为空，您是否忘记输入prompt音频？'
            return errcode, errmsg, (target_sr, default_data)
        
        if torchaudio.info(source_wav).sample_rate < prompt_sr:
            errcode = 7
            errmsg = 'source音频采样率{}低于{}'.format(torchaudio.info(source_wav).sample_rate, prompt_sr)
            return errcode, errmsg, (target_sr, default_data)

    generated_audio_list = []  # 用于存储生成的音频片段

    try:
        if mode_checkbox_group == '预训练音色':
            logging.info('get sft inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                generated_audio_list.append(i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '3s极速复刻':
            logging.info('get zero_shot inference request')
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
                generated_audio_list.append(i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '跨语种复刻':
            logging.info('get cross_lingual inference request')
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                generated_audio_list.append(i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '语音复刻':
            logging.info('get vc long inference request')
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            source_speech_16k = postprocess(load_wav(source_wav, prompt_sr))
            set_all_random_seed(seed)
            for i in cosyvoice.inference_vc_long(source_speech_16k, prompt_speech_16k, stream=stream, speed=speed):
                generated_audio_list.append(i)   
        else:
            logging.info('get instruct inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
                generated_audio_list.append(i['tts_speech'].numpy().flatten())

        clear_cuda_cache()
        # 合并所有音频片段为一整段
        if len(generated_audio_list) > 0:
            errcode = 0
            errmsg = 'ok'
            full_audio = np.concatenate(generated_audio_list)
            logging.info(f'full_audio: {full_audio.dtype}')
            return errcode, errmsg, (target_sr, full_audio)
        else:
            errcode = -2
            errmsg = "音频生成失败，未收到有效的音频数据。"
            return errcode, errmsg, (target_sr, default_data)
    except Exception as e:
        log_error(e)
        errcode = -1
        errmsg = f"音频生成失败，错误信息：{str(e)}"
        logging.error(errmsg)
        return errcode, errmsg, (target_sr, default_data)
    
# 包装处理逻辑
def gradio_generate_audio(tts_text, mode_checkbox_group, sft_dropdown, 
                        prompt_text, prompt_wav, 
                        instruct_text, seed, stream, speed,
                        source_wav
    ):
    errcode, errmsg, audio_data = generate_audio(
        tts_text, mode_checkbox_group, sft_dropdown, 
        prompt_text, prompt_wav,
        instruct_text, seed, stream, speed,
        source_wav
    )
    # 根据结果返回 Gradio 的更新
    if errcode == 0:  # 正常
        return (
            gr.update(value="", visible=False),  # 隐藏错误信息
            audio_data                           # 返回音频
        )
    else:  # 异常
        error_display = f"错误码: {errcode}\n错误信息: {errmsg}"
        return (
            gr.update(value=error_display, visible=True), # 显示错误信息
            audio_data                                    # 无音频输出
        )
        
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
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            source_wav = gr.Audio(
                sources=['upload', 'microphone'],
                type='filepath',
                label='上传或录制source音频文件，注意采样率不低于16khz'
            )
            prompt_wav = gr.Audio(
                sources=['upload', 'microphone'],
                type='filepath',
                label='上传或录制prompt音频文件，注意采样率不低于16khz'
            )
      
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        generate_button = gr.Button("生成音频")
        error_output = gr.Textbox(label="错误信息", visible=False)
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)
        # 定义重置函数（用于初始化时隐藏错误信息）
        def reset_error_outputs():
            return (
                gr.update(value="", visible=False)
            )
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            reset_error_outputs,  # 重置错误信息的状态
            inputs=[],
            outputs=[error_output]
        ).then(gradio_generate_audio,
            inputs=[
                tts_text, mode_checkbox_group, sft_dropdown, 
                prompt_text, prompt_wav, 
                instruct_text, seed, stream, speed,
                source_wav
            ],
            outputs=[error_output, 
                    audio_output
            ]
        )
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port, debug=False)

def generate_wav(audio_data, sample_rate, delay=0.0, volume_multiplier = 1.0):
    """
    使用 pydub 将音频数据转换为 WAV 格式，并支持添加延迟。
    :param audio_data: numpy 数组，音频数据
    :param sample_rate: int，采样率
    :param delay: float，延迟时间（单位：秒），默认为 0
    :param volume_multiplier: float，音量倍数，默认为 1.0
    :return: 文件路径，生成的 WAV 文件路径
    """
    # 确保 audio_data 是 numpy 数组
    if not isinstance(audio_data, np.ndarray):
        raise ValueError("audio_data 必须是 numpy 数组。")
    # 生成静音数据（如果有延迟需求）
    if delay > 0:
        num_silence_samples = int(delay * sample_rate)
        silence = np.zeros(num_silence_samples, dtype=audio_data.dtype)
        audio_data = np.concatenate((silence, audio_data), axis=0)
    # 检测音频数据类型并转换
    sample_width = 2
    if audio_data.dtype == np.float32:
        # 如果是 float32 数据，量化到 int16
        audio_data = (audio_data * 32767).astype(np.int16)
        sample_width = 2  # 16-bit (2 bytes per sample)
    elif audio_data.dtype == np.int16:
        sample_width = 2  # 16-bit (2 bytes per sample)
    elif audio_data.dtype == np.int8:
        audio_data = audio_data.astype(np.int16) * 256  # 转换为 int16
        sample_width = 2  # 16-bit
    else:
        raise ValueError("audio_data.dtype 不正确。")
    # 检测声道数
    if len(audio_data.shape) == 1:  # 单声道
        channels = 1
    elif len(audio_data.shape) == 2:  # 多声道
        channels = audio_data.shape[1]
    else:
        raise ValueError("audio_data.shape 格式不正确，必须是 1D 或 2D numpy 数组。")
    # 使用 pydub 生成音频段
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=sample_width,
        channels=channels
    )
    if volume_multiplier != 1.0:
        # 安全地增加音量
        audio_segment = volume_safely(audio_segment, volume_multiplier)
    # 指定保存文件的路径
    filename = f"{str(uuid.uuid4())}.wav"
    wav_dir = "results/output"
    wav_path = os.path.join(wav_dir, filename)
    # 确保目录存在
    os.makedirs(wav_dir, exist_ok=True)
    # 如果文件已存在，先删除
    if os.path.exists(wav_path):
        os.remove(wav_path)
    # 导出 WAV 文件
    audio_segment.export(wav_path, format="wav")

    return wav_path

async def save_upload_to_wav(upload_file: UploadFile, prefix: str, volume_multiplier: float = 1.0):
    """保存上传文件并转换为 WAV 格式（如果需要）"""
    # 指定保存文件的路径
    input_wav_dir = "results\input"
    # 确保目录存在
    os.makedirs(input_wav_dir, exist_ok=True)
    # 构造文件路径
    file_upload_path = os.path.join(input_wav_dir, f'{prefix}{upload_file.filename}')
    # 删除同名已存在的文件
    if os.path.exists(file_upload_path):
        os.remove(file_upload_path)

    logging.info(f"接收上传{upload_file.filename}请求 {file_upload_path}")

    try:
        # 保存上传的音频文件
        with open(file_upload_path, "wb") as f:
            f.write(await upload_file.read())
        # 检查文件格式并转换为 WAV（如果需要）
        if not file_upload_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(file_upload_path)
            wav_path = os.path.splitext(file_upload_path)[0] + ".wav"
            audio.export(wav_path, format="wav")
            os.remove(file_upload_path)  # 删除原始文件
            file_upload_path = wav_path

        if volume_multiplier != 1.0:
            # 加载音频并调整音量
            audio = AudioSegment.from_file(file_upload_path)
            audio = volume_safely(audio, volume_multiplier=volume_multiplier)
            # 保存调整后音量的音频
            audio.export(file_upload_path, format="wav")

        return file_upload_path
    except Exception as e:
        raise Exception(f"{upload_file.filename}音频文件保存或转换失败: {str(e)}")
    
# 定义 FastAPI 应用
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时加载模型
    model_manager.load_models()
    logging.info("Models loaded successfully!")
    yield  # 这里是应用运行的时间段
    logging.info("Application shutting down...")  # 在这里可以释放资源    
app = FastAPI(docs_url="/docs", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。

@app.get('/test')
async def test():
    """
    测试接口，用于验证服务是否正常运行。
    """
    return PlainTextResponse('success')

@app.post('/seed_vc')
async def seed_vc(
    source_wav:UploadFile = File(..., description="选择source音频文件，注意采样率不低于16khz"), 
    prompt_wav:UploadFile = File(..., description="选择prompt音频文件，注意采样率不低于16khz"), 
    spaker:float = Form(1.0, description="语速调节(0.5-2.0)")
):
    """
    用户自定义语音音色复刻接口。
    """
    try:
        prompt_wav_upload = await save_upload_to_wav(prompt_wav, "p", 1.5)
        source_wav_upload = await save_upload_to_wav(source_wav, "s", 1.0)
    except Exception as e:
        return JSONResponse({"errcode": -1, "errmsg": str(e)})
    ############################## generate ##############################
    seed_data = generate_seed()
    seed = seed_data["value"]

    errcode, errmsg, audio = generate_audio(
        tts_text = '', 
        mode_checkbox_group = '语音复刻', 
        sft_dropdown = '', 
        prompt_text = '', 
        prompt_wav = prompt_wav_upload, 
        instruct_text = '', 
        seed = seed, 
        stream = False, 
        speed = spaker, 
        source_wav = source_wav_upload
    )
    # 检查返回值中的错误码
    if errcode != 0:
       return JSONResponse({"errcode": errcode, "errmsg": errmsg})
    # 获取音频数据
    target_sr, audio_data = audio
    # 使用自定义方法生成 WAV 格式
    wav_path = generate_wav(audio_data, target_sr)
    # 返回音频响应
    return JSONResponse({"errcode": 0, "errmsg": "ok", "wav_path": wav_path})

@app.post('/fast_copy')
async def fast_copy(
    text:str = Form(..., description="输入合成文本"), 
    prompt_text:str = Form(..., description="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别"), 
    prompt_wav:UploadFile = File(..., description="选择prompt音频文件，注意采样率不低于16khz"), 
    spaker:float = Form(1.0, description="语速调节(0.5-2.0)"),
    delay: float = Form(0.5, description="文本音频前的延迟时间，单位秒（默认0.5秒）")
):
    """
    用户自定义音色语音合成接口。
    """
    try:
        prompt_wav_upload = await save_upload_to_wav(prompt_wav, "p", 1.5)
    except Exception as e:
        return JSONResponse({"errcode": -1, "errmsg": str(e)})
    ############################## generate ##############################
    lang = detect_language(text)
    if lang == 'en':
        sft_dropdown = '英文男'
    else:
        sft_dropdown = '中文男'

    seed_data = generate_seed()
    seed = seed_data["value"]

    errcode, errmsg, audio = generate_audio(
        tts_text = text, 
        mode_checkbox_group = '预训练音色', 
        sft_dropdown = sft_dropdown, 
        prompt_text = '', 
        prompt_wav = None, 
        instruct_text = '', 
        seed = seed, 
        stream = False, 
        speed = spaker, 
        source_wav = None
    )
    # 检查返回值中的错误码
    if errcode != 0:
       return JSONResponse({"errcode": errcode, "errmsg": errmsg})
    # 获取音频数据
    target_sr, audio_data = audio
    # 使用自定义方法生成 WAV 格式
    source_wav_upload = generate_wav(audio_data, target_sr, delay, 1.0)
    
    seed_data = generate_seed()
    seed = seed_data["value"]

    errcode, errmsg, audio = generate_audio(
        tts_text = '', 
        mode_checkbox_group = '语音复刻', 
        sft_dropdown = '', 
        prompt_text = '', 
        prompt_wav = prompt_wav_upload, 
        instruct_text = '', 
        seed = seed, 
        stream = False, 
        speed = spaker, 
        source_wav = source_wav_upload
    )
    # 检查返回值中的错误码
    if errcode != 0:
       return JSONResponse({"errcode": errcode, "errmsg": errmsg})
    # 获取音频数据
    target_sr, audio_data = audio
    # 使用自定义方法生成 WAV 格式
    wav_path = generate_wav(audio_data, target_sr, 0.0, 3.0)
    # 返回音频响应
    return JSONResponse({"errcode": 0, "errmsg": "ok", "wav_path": wav_path})

@app.post('/zero_shot')
async def zero_shot(
    text:str = Form(..., description="输入合成文本"), 
    prompt_text:str = Form(..., description="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别"), 
    prompt_wav:UploadFile = File(..., description="选择prompt音频文件，注意采样率不低于16khz"), 
    spaker:float = Form(1.0, description="语速调节(0.5-2.0)")
):
    """
    用户自定义音色语音合成接口。
    """
    try:
        prompt_wav_upload = await save_upload_to_wav(prompt_wav, "p", 1.5)
    except Exception as e:
        return JSONResponse({"errcode": -1, "errmsg": str(e)})
    ############################## generate ##############################
    seed_data = generate_seed()
    seed = seed_data["value"]

    errcode, errmsg, audio = generate_audio(
        tts_text = text, 
        mode_checkbox_group = '3s极速复刻', 
        sft_dropdown = '', 
        prompt_text = prompt_text, 
        prompt_wav = prompt_wav_upload, 
        instruct_text = '', 
        seed = seed, 
        stream = False, 
        speed = spaker, 
        source_wav = None
    )
    # 检查返回值中的错误码
    if errcode != 0:
       return JSONResponse({"errcode": errcode, "errmsg": errmsg})
    # 获取音频数据
    target_sr, audio_data = audio
    # 使用自定义方法生成 WAV 格式
    wav_path = generate_wav(audio_data, target_sr)
    # 返回音频响应
    return JSONResponse({"errcode": 0, "errmsg": "ok", "wav_path": wav_path})

@app.post('/tts')
async def tts(
    text:str = Form(..., description="输入合成文本"), 
    sft_dropdown:str = Form('中文女', description="输入预训练音色"), 
    spaker:float = Form(1.0, description="语速调节(0.5-2.0)")
):
    """
    使用预训练音色模型的语音合成接口。
    """
    ############################## generate ##############################
    seed_data = generate_seed()
    seed = seed_data["value"]

    errcode, errmsg, audio = generate_audio(
        tts_text = text, 
        mode_checkbox_group = '预训练音色', 
        sft_dropdown = sft_dropdown, 
        prompt_text = '', 
        prompt_wav = None, 
        instruct_text = '', 
        seed = seed, 
        stream = False, 
        speed = spaker, 
        source_wav = None
    )
    # 检查返回值中的错误码
    if errcode != 0:
       return JSONResponse({"errcode": errcode, "errmsg": errmsg})
  # 获取音频数据
    target_sr, audio_data = audio
    # 使用自定义方法生成 WAV 格式
    wav_path = generate_wav(audio_data, target_sr)
    # 返回音频响应
    return JSONResponse({"errcode": 0, "errmsg": "ok", "wav_path": wav_path})

@app.get('/download')
async def download(
    wav_path:str = Query(..., description="输入wav文件路径"), 
    name:str = Query(..., description="输入wav文件名")
):    
    """
    音频文件下载接口。
    """
    return FileResponse(path=wav_path, filename=name, media_type='application/octet-stream')

parser = argparse.ArgumentParser()
parser.add_argument('--webui',
                    type=bool,
                    default=False)
parser.add_argument('--port',
                    type=int,
                    default=8000)
parser.add_argument('--model_dir',
                    type=str,
                    default='pretrained_models/CosyVoice-300M',
                    help='local path or modelscope repo id')
args = parser.parse_args()
prompt_sr, target_sr = 16000, 22050
default_data = np.zeros(target_sr)

if __name__ == '__main__':
    if args.webui:
        model_manager.load_models()
        sft_spk = model_manager.sft_spk
        main()
    else:
        try:
            uvicorn.run(app="api:app", host="0.0.0.0", port=args.port, workers=1, reload=True, log_level="info")
        except Exception as e:
            log_error(e)
            print(e)
            exit(0)