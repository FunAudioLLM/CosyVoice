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
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.model import CosyVoiceModel
from custom.CosyVoiceFrontEnd import CosyVoiceFrontEnd
from custom.file_utils import logging

class CosyVoice:

    def __init__(self, model_dir, load_jit=True, load_onnx=False, fp16=True):
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
        if torch.cuda.is_available() is False and (fp16 is True or load_jit is True):
            load_jit = False
            fp16 = False
            logging.warning('cpu do not support fp16 and jit, force set to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
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

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self.frontend._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self.frontend._extract_speech_token(prompt_speech_16k)
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            if len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, embedding, 
                                                           (speech_feat, speech_feat_len), 
                                                           (speech_token, speech_token_len))
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))

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

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0, 
            embedding = None, 
            prompt_speech_feat_obj = None, 
            prompt_speech_token_obj = None
        ):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k,
                                                embedding = embedding, 
                                                prompt_speech_feat_obj = prompt_speech_feat_obj, 
                                                prompt_speech_token_obj = prompt_speech_token_obj)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / 22050
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()

    def inference_vc_long(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):    
        prompt_sr, target_sr = 16000, 22050
        overlap = 5  
        segment_length = 30

        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self.frontend._extract_speech_feat(prompt_speech_22050)
        prompt_speech_token, prompt_speech_token_len = self.frontend._extract_speech_token(prompt_speech_16k)
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)

        segments = self.segment_audio_with_overlap(source_speech_16k, segment_length, overlap, prompt_sr)
        generated_segments = []
        for segment in tqdm(segments):
            for i in self.inference_vc(segment, prompt_speech_16k, stream=stream, speed=speed,
                                       embedding = embedding,
                                       prompt_speech_feat_obj = (prompt_speech_feat, prompt_speech_feat_len),
                                       prompt_speech_token_obj = (prompt_speech_token, prompt_speech_token_len)
                                    ):
                generated_segments.append(i['tts_speech'].numpy().flatten())

        final_audio = self.crossfade_segments(generated_segments, overlap, target_sr)

        logging.info(f"Final length: {len(final_audio)}")

        yield final_audio

    def segment_audio_with_overlap(self, audio, segment_length=30, overlap=5, sample_rate=16000):
        """带重叠的音频分段，处理不足一个段长度的情况"""
        samples_per_segment = int(segment_length * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        total_samples = audio.size(1)
        logging.info(f'total_samples: {total_samples}')
        if total_samples <= samples_per_segment:
            # 如果音频小于等于一个段长度，直接返回整段
            return [audio]
        
        segments = []
        for i in range(0, total_samples - overlap_samples, samples_per_segment - overlap_samples):
            segments.append(audio[:, i:i + samples_per_segment])

        return segments

    def crossfade_segments(self, segments, overlap=5, sample_rate=16000):    
        """对生成的音频片段进行交叉淡化合并"""
        if len(segments) == 0:
            raise ValueError("Segments list is empty.")
        if len(segments) == 1:
            return segments[0]
        
        overlap_samples = int(overlap * sample_rate)
        result = segments[0]

        for i in range(1, len(segments)):
            prev_length = len(result)
            curr_length = len(segments[i])
            actual_overlap = min(overlap_samples, prev_length, curr_length)

            if actual_overlap == 0:
                # No overlap, just concatenate
                result = np.concatenate([result, segments[i]])
            else:
                # Generate fade-in and fade-out windows (using cosine for smoother transitions)
                fade_out = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, actual_overlap, dtype=np.float32)))
                fade_in = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, actual_overlap, dtype=np.float32)))

                # Crossfade the overlapping regions
                crossfaded = result[-actual_overlap:] * fade_out + segments[i][:actual_overlap] * fade_in

                # Concatenate the non-overlapping parts
                result = np.concatenate([result[:-actual_overlap], crossfaded, segments[i][actual_overlap:]])
        
        logging.info(f'crossfade_segments: {len(result)}')
        # Ensure the final length is within the expected range
        total_samples = sum(len(segment) for segment in segments)

        return result[:total_samples]  # Ensure we don't exceed the original length
