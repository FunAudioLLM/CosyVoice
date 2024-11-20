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
import uvicorn
import uuid
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
from fastapi import FastAPI, File, UploadFile, Query, Body, Form
from fastapi.responses import Response, StreamingResponse, JSONResponse, PlainTextResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from pydub import AudioSegment
from io import BytesIO
#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

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
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

# 定义一个函数进行显存清理
def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared!")

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    
    errcode = 0
    errmsg = ''

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
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
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
        else:
            logging.info('get instruct inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
                generated_audio_list.append(i['tts_speech'].numpy().flatten())

        clear_cuda_cache()
        # 合并所有音频片段为一整段
        if generated_audio_list:
            errcode = 0
            errmsg = 'ok'
            full_audio = np.concatenate(generated_audio_list)
            return errcode, errmsg, (target_sr, full_audio)
        else:
            errcode = -2
            errmsg = "音频生成失败，未收到有效的音频数据。"
            return errcode, errmsg, (target_sr, default_data)
    except Exception as e:
        errcode = -1
        errmsg = f"音频生成失败，错误信息：{str(e)}"
        logging.error(errmsg)
        return errcode, errmsg, (target_sr, default_data)
    
# 包装处理逻辑
def gradio_generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    # 调用通用的 `generate_audio` 函数
    errcode, errmsg, audio_data = generate_audio(
        tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record,
        instruct_text, seed, stream, speed
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
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
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
                prompt_text, prompt_wav_upload, prompt_wav_record, 
                instruct_text, seed, stream, speed
            ],
            outputs=[error_output, 
                    audio_output
            ]
        )
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

def generate_wav(audio_data, sample_rate):
    """
    使用 pydub 将音频数据转换为 WAV 格式。
    :param audio_data: numpy 数组，音频数据
    :param sample_rate: int，采样率
    :return: BytesIO 对象，包含 WAV 数据
    """
    # 检测音频数据类型
    sample_width = 2
    # audio_data 是 NumPy 数组
    if audio_data.dtype == np.float32:
        # 如果是 float32 数据，量化到 int16
        audio_data = (audio_data * 32767).astype(np.int16)
        sample_width = 2  # 16-bit (2 bytes per sample)
    elif audio_data.dtype == np.int16:
        sample_width = 2  # 16-bit (2 bytes per sample)
    elif audio_data.dtype == np.int8:
        audio_data = audio_data.astype(np.int16) * 256  # 转换为 int16
        sample_width = 2  # 16-bit
    # 检测声道数
    channels = 1        

    if len(audio_data.shape) == 1:  # 单声道
        channels = 1
    elif len(audio_data.shape) == 2:  # 多声道
        channels = audio_data.shape[1]

    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate = sample_rate,
        sample_width = sample_width,
        channels = channels
    )
    # 指定保存文件的路径
    filename = f"{str(uuid.uuid4())}.wav"
    wav_dir = "results\output"
    wav_path = os.path.join(wav_dir, filename)
    # 确保目录存在
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    # 如果文件已存在，先删除
    if os.path.exists(wav_path):
        os.remove(wav_path)

    audio_segment.export(wav_path, format="wav")

    return wav_path

app=FastAPI()
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

@app.post('/fast_copy')
async def fast_copy(
    text:str = Form(..., description="输入合成文本"), 
    prompt_text:str = Form(..., description="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别"), 
    prompt_wav:UploadFile = File(..., description="选择prompt音频文件，注意采样率不低于16khz"), 
    spaker:float = Form(1.0, description="语速调节(0.5-2.0)")
):
    """
    用户自定义音色语音合成接口。
    """
    ###################### 读取上传的音频文件 ######################
    # 指定保存文件的路径
    prompt_wav_dir = "results\input"
    prompt_wav_upload = os.path.join(prompt_wav_dir, prompt_wav.filename)
    # 确保目录存在
    if not os.path.exists(prompt_wav_dir):
        os.makedirs(prompt_wav_dir)
    # 如果文件已存在，先删除
    if os.path.exists(prompt_wav_upload):
        os.remove(prompt_wav_upload)

    print(f"接收上传prompt_wav请求 {prompt_wav_upload}")
    try:
        # 保存上传的音频文件
        with open(prompt_wav_upload, "wb") as f:
            f.write(await prompt_wav.read())
    except Exception as e:
        return JSONResponse({"errcode": -1, "errmsg": f"音频文件保存失败: {str(e)}"})
    
    ###################### generate_audio ######################
    seed_data = generate_seed()
    seed = seed_data["value"]
    # 调用 generate_audio
    errcode, errmsg, audio = generate_audio(
        text, '3s极速复刻', '', prompt_text, prompt_wav_upload, None, '', 
        seed=seed, stream=1, speed=spaker
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
    seed_data = generate_seed()
    seed = seed_data["value"]
    # 调用 generate_audio
    errcode, errmsg, audio = generate_audio(
        text, '预训练音色', sft_dropdown, '', None, None, '', 
        seed=seed, stream=1, speed=spaker
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
cosyvoice = CosyVoice(args.model_dir)
sft_spk = cosyvoice.list_avaliable_spks()
prompt_sr, target_sr = 16000, 22050
default_data = np.zeros(target_sr)

if __name__ == '__main__':
    if args.webui:
        main()
    else:
        try:
            uvicorn.run(app="api:app", host="0.0.0.0", port=args.port, workers=1, reload=True)
        except Exception as e:
            print(e)
            exit(0)