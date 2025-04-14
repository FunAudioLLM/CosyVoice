import sys

import librosa
import torch 
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from flask import Flask, request, jsonify, send_file
from flask_siwadoc import SiwaDoc
import os
import logging
app = Flask(__name__)
siwa = SiwaDoc(app)
max_val = 0.8
UPLOAD_FOLDER = './reference/voice_id'  # 上传音频的目录
OUTPUT_FOLDER = './output/voice_id'     # 生成音频的目录
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT',load_jit=False, load_trt=False,fp16=False)
#cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B',load_jit=False, load_trt=False,fp16=False,flow_cache=False)

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """
    对音频进行后处理，包括裁剪和归一化。

    :param speech: 输入的音频数据
    :param top_db: 裁剪时使用的分贝阈值
    :param hop_length: 跳跃长度
    :param win_length: 窗口长度
    :return: 处理后的音频数据
    """
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

# 定义上传音频的 API 端点
@app.route('/upload_audio', methods=['POST'])
@siwa.doc()
def upload_audio():
    """
    上传音频文件的 API 端点。

    请求方法: POST
    请求参数:
        - audio: 要上传的音频文件

    响应:
        - 若未提供音频文件，返回 400 错误，包含错误信息
        - 若文件名为空，返回 400 错误，包含错误信息
        - 若上传成功，返回 200 状态码，包含成功信息和文件保存路径
    """
    if request.content_type != 'multipart/form-data':
        return jsonify({"error": "Unsupported media type. Expected 'multipart/form-data'"}), 415
    # 检查请求中是否包含文件
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    # 保存文件到指定目录
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"message": "Audio file uploaded successfully", "file_path": file_path}), 200

# 定义生成音频的 API 端点
@app.route('/generate_audio', methods=['POST', 'GET'])
@siwa.doc()
def genetate_audio():
    """
    生成音频的 API 端点。
    请求方法: POST
    请求参数:
        - text: 要生成的文本
        - audio_path: 用于生成文本的音频文件路径
    响应:
        - 若未提供文本或音频文件路径，返回 400 错误，包含错误信息
        - 若音频文件不存在，返回 400 错误，包含错误信息
        - 若生成成功，返回 200 状态码，包含成功信息和生成的音频文件路径
    """
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported media type. Expected 'application/json'"}), 415
    logging.info('request.json: {}'.format(request.json))
    # 检查请求中是否包含文本
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    text = request.json['text']
    # 检查请求中是否包含音频文件路径
    if 'audio_path' not in request.json:
        return jsonify({"error": "No audio file path provided"}), 400
    audio_path = request.json['audio_path']
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 400
    # 加载音频文件
    audio = postprocess(load_wav(audio_path, cosyvoice.sample_rate))
    prompt_text = "你好吗？今天天气不错，你的心情怎么样？"
    # prompt_text = speech_to_text(audio_path)
    # logging.info('audio_path: {}'.format(audio_path))
    # logging.info('prompt_text: {}'.format(prompt_text))
    print('audio_path:', audio_path)
    print('prompt_text:', prompt_text)
    
    # 生成音频
    #for i, j in enumerate(cosyvoice.inference_instruct2(text, prompt_text, prompt_speech_16k=audio, stream=False, speed=1.0)):
    #for i, j in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech_16k=audio, stream=False, speed=1.0)):
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k=audio, stream=False, speed=1.0)):
        #inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)
        #inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)
        #inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        # audio_path = os.path.join(OUTPUT_FOLDER, 'cross_lingual_{}.wav'.format(i))
        # audio_path = os.path.join(OUTPUT_FOLDER, 'instruct_{}.wav'.format(i))
        audio_path = os.path.join(OUTPUT_FOLDER, 'zero_shot_{}.wav'.format(i))
        print("audio_path:", audio_path)
        torchaudio.save(audio_path, j['tts_speech'], cosyvoice.sample_rate)
    
    return get_audio(), 200



def speech_to_text(audio_path='./reference/voice_id/1.mp3'):
    # 加载音频文件
    audio= load_wav(audio_path, cosyvoice.sample_rate)
    # 生成文本
    for i, j in enumerate(cosyvoice.inference_zero_shot('', '', prompt_speech_16k=audio, stream=False, speed=1.0)):

        torchaudio.save('zero_shot_1_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    return j['tts_text']

@app.route('/speech_to_text', methods=['POST', 'GET'])
@siwa.doc()
def speech_to_text_api():
    """
    语音转文本的 API 端点。
    请求方法: POST
    请求参数:
        - audio_path: 用于生成文本的音频文件路径
    响应:
        - 若未提供音频文件路径，返回 400 错误，包含错误信息
        - 若音频文件不存在，返回 400 错误，包含错误信息
        - 若生成成功，返回 200 状态码，包含成功信息和生成的文本
    """
    # 检查请求中是否包含音频文件路径
    if 'audio_path' not in request.json:
        return jsonify({"error": "No audio file path provided"}), 400
    
    audio_path = request.json['audio_path']
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 400
    # 生成文本
    text = speech_to_text(audio_path)
    return jsonify({"text": text}), 200

@app.route('/get_audio', methods=['GET'])
@siwa.doc()
def get_audio():
    """
    获取生成的音频文件的 API 端点。
    请求方法: GET
    响应:
        - 若生成的音频文件存在，返回 200 状态码，包含音频文件
        - 若生成的音频文件不存在，返回 404 错误，包含错误信息
    """
    audio_path = os.path.join(OUTPUT_FOLDER, 'zero_shot_0.wav')
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    else:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True,port=5000)


'''

# zero shot usage
prompt_speech_16k = load_wav('./asset/huang/1.mp3', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('习近平总书记的重要讲话在与会代表中引发热烈反响。他们结合工作实际，畅谈对会议精神的学习体会与落实打算，表示将深入贯彻总书记重要讲话提出的要求，聚焦构建周边命运共同体，努力开创周边工作新局面。', '你好吗？今天天气不错，你的心情怎么样？', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_1_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('习近平总书记的重要讲话[laughter]在与会代表中[laughter]引发热烈反响。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('习近平总书记的重要讲话在与会代表中引发热烈反响。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '习近平总书记的重要讲话在与会代表中引发热烈反响。'
    yield '他们结合工作实际，'
    yield '畅谈对会议精神的学习体会与落实打算，'
    yield '表示将深入贯彻总书记重要讲话提出的要求，'
    yield '聚焦构建周边命运共同体，'
    yield '努力开创周边工作新局面。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '你好吗？今天天气不错，你的心情怎么样？', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_2_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
'''