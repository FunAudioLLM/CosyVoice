# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np
import time

def main():
    url = "http://{}:{}/inference_zero_shot".format(args.host, args.port)
    payload = {
            'tts_text': args.tts_text,
            'person': args.person
        }

    response = requests.request("GET", url, data=payload, stream=True)

    tts_audio = b''
    start = time.time()
    output_file = args.save.replace('.wav', '{}.wav')
    for i, r in enumerate(response.iter_content(chunk_size=16000)):
        if i == 0:
            print('first ack time: {}'.format(time.time() - start))
        tts_audio += r
        r = torch.from_numpy(np.array(np.frombuffer(r, dtype=np.int16))).unsqueeze(dim=0)
        torchaudio.save(output_file.format(i), r, 24000)
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    logging.info('save response')
    torchaudio.save(output_file.format(''), tts_speech, 24000)
    logging.info('get response')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('--port',
                        type=int,
                        default='21559')
    parser.add_argument('--tts_text',
                        type=str,
                        default='"세상은 매일 진화하고 있습니다. AI는 이제 단순한 도구가 아니라, 우리 삶의 일부가 되었죠. 오늘, 그 놀라운 변화를 함께 만나봅니다."')
    parser.add_argument('--person',
                        type=str,
                        default='woon',
                        help='speaker name')
    parser.add_argument('--save',
                        type=str,
                        default='/tmp/tts.wav',
                        help='path to save the response')
    args = parser.parse_args()
    main()
