import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np
import time
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))

from stream_player import StreamPlayer

logging.basicConfig(level=logging.DEBUG)
player = StreamPlayer(sample_rate=22050, channels=1, block_size=18048)
player.start()

def main():
    url = "http://{}:{}/inference_{}".format(args.host, args.port, args.mode)
    logging.info('请求URL: {}'.format(url))

    time_start = time.time()
    
    if args.mode == 'sft':
        payload = {
            'tts_text': args.tts_text,
            'spk_id': args.spk_id
        }
        response = requests.request("GET", url, data=payload, stream=True, timeout=30)
    elif args.mode == 'zero_shot':
        payload = {
            'tts_text': args.tts_text,
            'prompt_text': args.prompt_text
        }
        files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)
    elif args.mode == 'cross_lingual':
        payload = {
            'tts_text': args.tts_text,
        }
        files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)
    elif args.mode == 'instruct2':
        payload = {
            'tts_text': args.tts_text,
            'instruct_text': args.instruct_text,
            'spk_id': args.spk_id
        }
        # files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        # response = requests.request("GET", url, data=payload, files=files, stream=True)
        response = requests.request("GET", url, data=payload, stream=True, timeout=30)
    else:
        payload = {
            'tts_text': args.tts_text,
            'spk_id': args.spk_id,
            'instruct_text': args.instruct_text
        }
        response = requests.request("GET", url, data=payload, stream=True, timeout=30)
    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        player.play(r)
        tts_audio += r
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    time_end = time.time()
    logging.info('time cost: {}'.format(time_end - time_start))
    logging.info('save response to {}'.format(args.tts_wav))
    torchaudio.save(args.tts_wav, tts_speech, target_sr)
    logging.info('get response')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('--port',
                        type=int,
                        default='50000')
    parser.add_argument('--mode',
                        default='sft',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct', 'instruct2'],
                        help='请求模式')
    parser.add_argument('--tts_text',
                        type=str,
                        default='你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？')
    parser.add_argument('--spk_id',
                        type=str,
                        default='中文女')
    parser.add_argument('--prompt_text',
                        type=str,
                        default='希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav',
                        type=str,
                        default='../../../asset/zero_shot_prompt.wav')
    parser.add_argument('--instruct_text',
                        type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader. \
                                 Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--tts_wav',
                        type=str,
                        default='demo.wav')
    parser.add_argument('--timeout',
                        type=int,
                        default=300,
                        help='请求超时时间(秒)')
    args = parser.parse_args()
    prompt_sr, target_sr = 16000, 22050
    main()
