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
from typing import Generator
from tqdm import tqdm
from math import ceil
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
import torchaudio
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.dirname(parent_dir)

class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.is_05b = True if 'CosyVoice2-0.5B' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def _process_with_progress(self, model_input, text_segment, stream, speed):
        """处理带进度条的TTS生成
        Args:
            model_input: 模型输入
            text_segment: 当前文本片段
            stream: 是否流式输出
            speed: 语速
        """
        start_time = time.time()
        tqdm.write(f'{text_segment}\n')
        
        # 初始化进度条参数
        estimated_iterations = 1 if not stream else max(1, len(text_segment) // 10)
        
        with tqdm(total=estimated_iterations, leave=False, desc='当前片段', disable=not stream) as pbar:
            iter_count = 0
            
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech = model_output['tts_speech']
                speech_len = speech.shape[1] / self.sample_rate
                iter_count += 1

                if stream:
                    # 更新进度条
                    rtf = (time.time() - start_time) / speech_len
                    pbar.set_postfix_str(f'rtf={rtf:.2f}')
                    
                    # 仅在第一次迭代更新预估值
                    if iter_count == 1:
                        text_to_speech_ratio = speech_len / len(text_segment)
                        new_estimate = max(1, ceil(len(text_segment) * text_to_speech_ratio * self.sample_rate / speech.shape[1]))
                        pbar.total = max(estimated_iterations, new_estimate)
                    
                    # 动态调整总迭代次数
                    pbar.total = max(pbar.total, iter_count)
                    pbar.update(1)
                else:
                    pbar.close()

                yield model_output
                start_time = time.time()

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        default_voices = self.list_available_spks()
        
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend), desc='生成进度'):

            # 根据音色ID获取模型输入
            spk = default_voices[0] if spk_id not in default_voices else spk_id
            model_input = self.frontend.frontend_sft(text_segment, spk)

            # 如果是自定义音色,加载并更新音色相关特征
            if spk_id not in default_voices:
                newspk = torch.load(
                    f'{grandparent_dir}/voices/{spk_id}.pt',
                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )
                
                # 更新模型输入中的音色特征
                spk_fields = [
                    "flow_embedding", "llm_embedding",
                    "llm_prompt_speech_token", "llm_prompt_speech_token_len",
                    "flow_prompt_speech_token", "flow_prompt_speech_token_len", 
                    "prompt_speech_feat_len", "prompt_speech_feat",
                    "prompt_text", "prompt_text_len"
                ]
                
                for field in spk_fields:
                    model_input[field] = newspk[field]

            yield from self._process_with_progress(model_input, text_segment, stream, speed)

    def _save_voice_model(self, model_input, prompt_speech_16k, text_ref=None, save_path='output.pt'):
        """保存音色模型到文件
        Args:
            model_input: 包含音色信息的模型输入
            prompt_speech_16k: 参考音频
            text_ref: 参考文本（可选）
            save_path: 保存路径，默认为output.pt
        """
        model_input['audio_ref'] = prompt_speech_16k
        if text_ref is not None:
            model_input['text_ref'] = text_ref
        
        torch.save(model_input, save_path)

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        
        # 先获取所有分段，找出最长的一段
        text_parts = list(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))
        longest_segment = max(text_parts, key=len)
        longest_idx = text_parts.index(longest_segment)
        
        for idx, text_segment in enumerate(tqdm(text_parts, desc='生成进度')):
            if (not isinstance(text_segment, Generator)) and len(text_segment) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(text_segment, prompt_text))
            model_input = self.frontend.frontend_zero_shot(text_segment, prompt_text, prompt_speech_16k, self.sample_rate)

            if idx == 0 or idx == longest_idx:  # 保存第一段或最长段作为音色模型
                self._save_voice_model(model_input, prompt_speech_16k, prompt_text)

            yield from self._process_with_progress(model_input, text_segment, stream, speed)

    def inference_cross_lingual(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        # 先获取所有分段，找出最长的一段
        text_parts = list(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))
        longest_segment = max(text_parts, key=len)
        longest_idx = text_parts.index(longest_segment)
        
        for idx, text_segment in enumerate(tqdm(text_parts, desc='生成进度')):
            model_input = self.frontend.frontend_cross_lingual(text_segment, prompt_speech_16k, self.sample_rate)
            
            if idx == 0 or idx == longest_idx:  # 保存第一段或最长段作为音色模型
                self._save_voice_model(model_input, prompt_speech_16k)

            yield from self._process_with_progress(model_input, text_segment, stream, speed)

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(text_segment, spk_id, instruct_text)
            yield from self._process_with_progress(model_input, text_segment, stream, speed)

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.is_05b = True if 'CosyVoice2-0.5B' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(text_segment, instruct_text, prompt_speech_16k, self.sample_rate)
            yield from self._process_with_progress(model_input, text_segment, stream, speed)
