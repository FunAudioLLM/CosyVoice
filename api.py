import time
import io, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 获取当前文件的绝对路径的目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 添加依赖的第三方库到系统路径
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import requests
from pydub import AudioSegment

import numpy as np
# 导入Flask相关库，用于构建Web API服务
from flask import Flask, request, Response, send_from_directory
import torch
import torchaudio

# 导入CosyVoice模型相关的库
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
# import ffmpeg

# 导入CORS处理跨域问题的库
from flask_cors import CORS
from flask import make_response

import shutil

import json

# 初始化CosyVoice2模型，加载预训练模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_trt=True, fp16=True)

# 获取模型内置的预训练音色列表
default_voices = cosyvoice.list_available_spks()

# 获取自定义音色列表
spk_custom = []
for name in os.listdir(f"{ROOT_DIR}/voices/"):
    print(name.replace(".pt", ""))
    spk_custom.append(name.replace(".pt", ""))

print("默认音色", default_voices)
print("自定义音色", spk_custom)

# 创建Flask应用实例
app = Flask(__name__)

# 配置跨域资源共享(CORS)
# 允许所有来源的跨域请求，解决前端与后端API交互时的跨域问题
CORS(app, cors_allowed_origins="*")
# CORS(app, supports_credentials=True)  # 支持携带凭证的跨域请求(如cookie)，当前已注释
def process_audio(tts_speeches, sample_rate=22050, format="wav"):
    """
    处理音频数据并返回响应
    
    参数:
        tts_speeches: 待处理的音频张量列表
        sample_rate: 采样率，默认22050Hz
        format: 输出音频格式，默认wav
    
    返回:
        包含音频数据的内存缓冲区
    """
    buffer = io.BytesIO()  # 创建内存缓冲区，是一个在内存中模拟文件操作的对象
    
    # 合并多个音频片段，dim=1表示在第二个维度上合并
    # 假设tts_speeches中的每个张量形状为[1, L]，合并后audio_data形状仍为[1, L_total]
    # 第0维保留，表示声道数，通常为1（单声道）
    audio_data = torch.concat(tts_speeches, dim=1)
    
    # 将音频数据保存到内存缓冲区，此操作会将文件写入buffer并移动指针到末尾
    torchaudio.save(buffer, audio_data, sample_rate, format=format)
    
    # 将缓冲区指针重置到开始位置，这样后续读取时才能从头开始读取数据
    # 如果不重置，后续读取将从末尾开始，得到空数据
    buffer.seek(0)
    
    return buffer

def create_audio_response(buffer, format="wav"):
    """
    创建音频HTTP响应
    
    参数:
        buffer: 包含音频数据的缓冲区
        format: 音频格式，默认wav
    
    返回:
        Flask响应对象，包含适当的MIME类型和头信息
    """
    if format == "wav":
        # wav格式直接返回Response对象
        return Response(buffer.read(), mimetype="audio/wav")
    else:
        # 其他格式使用make_response创建响应，并设置适当的头信息
        response = make_response(buffer.read())
        response.headers['Content-Type'] = f'audio/{format}'
        response.headers['Content-Disposition'] = f'attachment; filename=sound.{format}'
        return response

def load_voice_data(speaker):
    """
    加载自定义语音数据
    
    参数:
        speaker: 说话人ID/名称
    
    返回:
        加载的语音参考数据，如果加载失败则返回None
    """
    voice_path = f"{ROOT_DIR}/voices/{speaker}.pt"
    try:
        # 检测是否有GPU可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(voice_path):
            return None
        # 加载语音模型数据
        voice_data = torch.load(voice_path, map_location=device)
        return voice_data.get('audio_ref')
    except Exception as e:
        raise ValueError(f"加载音色文件失败: {e}")

# 定义路由，同时处理根路径和/tts路径的GET和POST请求
@app.route("/", methods=['GET', 'POST'])
@app.route("/tts", methods=['GET', 'POST'])
def tts():
    """
    文本转语音(TTS)API端点
    处理文本到语音的转换请求，支持GET和POST方法
    
    请求参数:
        text: 要转换的文本
        speaker: 说话人ID/名称
        instruct: 指令模式下的提示(可选)
        streaming: 是否使用流式输出，默认0(非流式)
        speed: 语速，默认1.0
    
    返回:
        音频数据或错误信息
    """
    # 根据请求方法获取参数，支持GET和POST两种方式
    params = request.get_json() if request.method == 'POST' else request.args
    text = params.get('text')
    speaker = params.get('speaker')
    instruct = params.get('instruct')
    streaming = int(params.get('streaming', 0))
    speed = float(params.get('speed', 1.0))

    # 参数验证
    if not text or not speaker:
        return {"error": "文本和角色名不能为空"}, 400

    # 处理指令模式(可自定义音色)
    if instruct:
        prompt_speech_16k = load_voice_data(speaker)
        if prompt_speech_16k is None:
            return {"error": "预训练音色文件中缺少audio_ref数据！"}, 500
        
        # 定义指令模式下的推理函数
        inference_func = lambda: cosyvoice.inference_instruct2(
            text, instruct, prompt_speech_16k, stream=bool(streaming), speed=speed
        )
    else:
        # 定义标准模式下的推理函数
        inference_func = lambda: cosyvoice.inference_sft(
            text, speaker, stream=bool(streaming), speed=speed
        )

    # 处理流式输出模式
    if streaming:
        def generate():
            """生成器函数，用于流式传输音频片段"""
            # 第一个标志，用于标记是否已经发送WAV头
            first_chunk = True
            
            for _, i in enumerate(inference_func()):
                audio_data = i['tts_speech'].numpy()[0]  # 获取原始音频数据
                
                if first_chunk:
                    # 第一个片段，发送完整WAV头
                    buffer = process_audio([i['tts_speech']], format="wav")
                    yield buffer.read()
                    first_chunk = False
                else:
                    # 后续片段，只发送原始音频数据
                    # 将音频数据转换为字节流
                    audio_bytes = (audio_data * (2 ** 15)).astype(np.int16).tobytes()
                    yield audio_bytes
        
        # 创建流式响应
        response = make_response(generate())
        response.headers.update({
            'Content-Type': 'audio/wav',
            'Transfer-Encoding': 'chunked',  # 使用分块传输编码
            'Content-Disposition': 'attachment; filename=sound.wav'
        })
        return response
    
    # 处理非流式输出模式
    tts_speeches = [i['tts_speech'] for _, i in enumerate(inference_func())]
    buffer = process_audio(tts_speeches, format="wav")
    return create_audio_response(buffer)


@app.route("/speakers", methods=['GET', 'POST'])
def speakers():
    """
    获取可用说话人列表的API端点
    返回系统中所有可用的预训练和自定义音色列表
    
    返回:
        包含音色信息的JSON响应
    """
    voices = []

    # 添加预训练的默认音色
    for x in default_voices:
        voices.append({"name":x,"voice_id":x})

    # 添加自定义音色
    for name in os.listdir("voices"):
        name = name.replace(".pt","")
        voices.append({"name":name,"voice_id":name})

    # 创建JSON响应，确保使用UTF-8编码，并显式设置Content-Type
    response = app.response_class(
        response=json.dumps(voices, ensure_ascii=False),
        status=200,
        mimetype='application/json; charset=utf-8'
    )
    response.headers.set('Content-Type', 'application/json; charset=utf-8')
    return response

# 程序入口点
if __name__ == "__main__":
    # 启动Flask Web服务器
    # host='0.0.0.0'表示接受来自任何IP的连接
    # port=9880指定服务端口
    app.run(host='0.0.0.0', port=9880)
