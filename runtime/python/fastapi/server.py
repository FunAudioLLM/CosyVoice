import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import argparse
import logging
import torch
# logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.DEBUG)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import io
from typing import Iterator, Any, List
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

CosyVoice2_path = '{}/../../..'.format(ROOT_DIR)

app = FastAPI()
# 设置跨域资源共享(CORS)
# 这段代码允许不同域名的前端应用访问这个API服务器
# 如果没有这个设置，当前端应用(如网页)的域名与API服务器域名不同时，
# 浏览器会因安全限制阻止前端访问API，导致用户无法使用语音合成功能
# 例如：如果网页在example.com上，而API在api.example.com上，没有CORS设置会导致请求失败
# "*"表示允许任何网站访问，在生产环境中可能需要限制为特定域名以提高安全性
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

def process_audio_chunk(audio_data) -> bytes:
    """将模型输出的音频数据转换为字节格式"""
    try:
        # 当前使用标准16位PCM格式以确保兼容性
        audio_bytes = (audio_data.numpy() * (2 ** 15)).astype(np.int16).tobytes()
        return audio_bytes
    except Exception as e:
        logging.error(f"处理音频数据时出错: {e}")
        return b''

def generate_data(model_output) -> Iterator[bytes]:
    """
    处理模型输出的语音数据，将其转换为适合流式传输的格式
    
    关于数据格式选择的说明：
    - 确实可以使用float16或float32格式，这样处理流程会更简洁
    - float16相比float32可以节省一半带宽，同时保持足够的精度
    - 音频数据在[-1,1]范围内的浮点数确实可以被现代音频库直接处理
    - 但需要确保客户端能正确解析这种格式的数据
    
    当前我们使用int16是因为：
    1. 广泛兼容性：几乎所有音频系统都支持16位PCM
    2. 客户端代码：当前客户端已配置为接收并解析int16格式
    
    如果整个系统都支持浮点音频，可以考虑简化为float16格式
    """
    try:
        # 检查model_output是否为可迭代对象
        if hasattr(model_output, '__iter__'):
            for i, item in enumerate(model_output):
                if 'tts_speech' in item:
                    audio_bytes = process_audio_chunk(item['tts_speech'])
                    logging.debug(f"发送音频数据片段[{i}]，大小: {len(audio_bytes)} 字节")
                    yield audio_bytes
        else:
            # 处理单个输出的情况
            if hasattr(model_output, 'tts_speech'):
                audio_bytes = process_audio_chunk(model_output.tts_speech)
                logging.debug(f"发送单个音频数据片段，大小: {len(audio_bytes)} 字节")
                yield audio_bytes
    except Exception as e:
        logging.error(f"生成数据时发生错误: {e}")
        raise


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    # 使用流式模式生成语音，inference_sft本身会返回一个生成器
    try:
        logging.info(f"开始语音合成: '{tts_text[:30]}...'")
        
        # 获取所有生成的音频数据
        all_outputs = []
        
        # 对生成的每个片段进行处理
        async def stream_generator():
            for chunk in cosyvoice.inference_sft(tts_text, spk_id, stream=True):
                # 处理单个音频数据块
                audio_bytes = process_audio_chunk(chunk['tts_speech'])
                logging.debug(f"流式发送音频数据片段，大小: {len(audio_bytes)} 字节")
                yield audio_bytes
        
        # 返回流式响应
        return StreamingResponse(
            stream_generator(),
            media_type="audio/wave",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
    except Exception as e:
        logging.error(f"inference_sft处理失败: {e}")
        raise


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    try:
        model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    except Exception as e:
        logging.error(f"inference_instruct处理失败: {e}")
        raise


def load_voice_data(speaker):
    """
    加载自定义语音数据
    
    参数:
        speaker: 说话人ID/名称
    
    返回:
        加载的语音参考数据，如果加载失败则返回None
    """

    voice_path = f"{CosyVoice2_path}/voices/{speaker}.pt"
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


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
# async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File(), spk_id: str = Form()):
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), spk_id: str = Form()):
    try:
        prompt_speech_16k = load_voice_data(spk_id)
        # else:
        #     prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    except Exception as e:
        logging.error(f"inference_instruct2处理失败: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default= CosyVoice2_path+'/pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--timeout',
                        type=int,
                        default=120,
                        help='服务器请求超时时间（秒）')
    args = parser.parse_args()
    try:
        # cosyvoice = CosyVoice(args.model_dir)
        print(f"使用模型目录: {args.model_dir}")
        cosyvoice = CosyVoice2(
            args.model_dir,
            load_jit=False,
            load_trt=True,
            fp16=True,
            use_flow_cache=True,
        )
    except Exception:
        raise TypeError('no valid model_type!')
    
    # 设置更长的超时时间，确保长文本语音合成不会中断
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        timeout_keep_alive=args.timeout,
        timeout_graceful_shutdown=args.timeout,
        limit_concurrency=10
    )
