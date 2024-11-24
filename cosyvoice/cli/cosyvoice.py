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
import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.file_utils import logging
import torch
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.dirname(parent_dir)



def time_it(func):
  """
  这是一个装饰器，用来计算类方法运行的时长，单位秒.
  """
  def wrapper(self, *args, **kwargs):
    start_time = time.time()
    result = func(self, *args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"推理方法 {func.__name__} 运行时长: {duration:.4f} 秒")
    return result
  return wrapper


def ms_to_srt_time(ms):
    N = int(ms)
    hours, remainder = divmod(N, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    timesrt = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    # print(timesrt)
    return timesrt


class CosyVoice:

    def __init__(self, model_dir, load_jit=True, load_onnx=False):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                '{}/llm.llm.fp16.zip'.format(model_dir),
                                '{}/flow.encoder.fp32.zip'.format(model_dir))
        if load_onnx:
            self.model.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0,new_dropdown="无"):

        default_voices = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

        tts_speeches = []
        audio_opt = []
        audio_samples = 0
        srtlines = []
        
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):

            if new_dropdown != "无" or spk_id not in default_voices:

                model_input = self.frontend.frontend_sft(i,"中文女")

                print({grandparent_dir})

                if new_dropdown != "无":

                    newspk = torch.load(f'{grandparent_dir}/voices/{new_dropdown}.pt')
                else:
                    newspk = torch.load(f'{grandparent_dir}/voices/{spk_id}.pt')

                model_input["flow_embedding"] = newspk["flow_embedding"] 
                model_input["llm_embedding"] = newspk["llm_embedding"]

                model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                model_input["prompt_text"] = newspk["prompt_text"]
                model_input["prompt_text_len"] = newspk["prompt_text_len"]
                
            else:
            
                model_input = self.frontend.frontend_sft(i, spk_id)
            
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                
                # 使用 .numpy() 方法将 tensor 转换为 numpy 数组
                numpy_array = model_output['tts_speech'].numpy()
                # 使用 np.ravel() 方法将多维数组展平成一维数组
                audio = numpy_array.ravel()
                print(audio)
                srtline_begin=ms_to_srt_time(audio_samples*1000.0 / 22050)
                audio_samples += audio.size
                srtline_end=ms_to_srt_time(audio_samples*1000.0 / 22050)
                audio_opt.append(audio)

                srtlines.append(f"{len(audio_opt):02d}\n")
                srtlines.append(srtline_begin+' --> '+srtline_end+"\n")

                srtlines.append(i.replace("、。","")+"\n\n")

                tts_speeches.append(model_output['tts_speech'])

                
                yield model_output
                start_time = time.time()
            
            audio_data = torch.concat(tts_speeches, dim=1)

            torchaudio.save("音频输出/output.wav", audio_data, 22050)
            with open('音频输出/output.srt', 'w', encoding='utf-8') as f:
                f.writelines(srtlines)

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            # 保存数据
            torch.save(model_input, 'output.pt') 
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / 22050
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()
