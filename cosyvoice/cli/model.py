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
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # here we fix set flow.decoder.estimator.static_chunk_size = 0 for compatibability
        self.flow.decoder.estimator.static_chunk_size = 0
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        """
        LLM推理任务函数，在单独的CUDA流中执行以实现并行处理。
        
        llm_context是一个CUDA流上下文，用于以下目的：
        1. 并行执行：允许LLM推理与主线程中的其他操作（如音频合成）并行执行
        2. 内存隔离：为LLM操作提供独立的CUDA流，避免与主流的资源竞争
        3. 异步处理：使LLM推理可以在后台进行，不阻塞主线程的其他操作
        
        在CPU模式下，llm_context被设置为nullcontext()，不提供实际功能。
        """
        with self.llm_context:  # 在独立CUDA流中执行LLM推理，实现与主线程的并行处理
            if isinstance(text, Generator):
                assert isinstance(self, CosyVoice2Model), 'streaming input text is only implemented for CosyVoice2!'
                for i in self.llm.inference_bistream(text=text,
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                for i in self.llm.inference(text=text.to(self.device),
                                            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_text=prompt_text.to(self.device),
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel, flow_cache = self.flow.inference(token=token.to(self.device),
                                                  token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_token=prompt_token.to(self.device),
                                                  prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_feat=prompt_feat.to(self.device),
                                                  prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                  embedding=embedding.to(self.device),
                                                  flow_cache=self.flow_cache_dict[uuid])
        self.flow_cache_dict[uuid] = flow_cache

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        torch.cuda.empty_cache()

    def vc(self, source_speech_token, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = source_speech_token.flatten().tolist(), True
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        torch.cuda.empty_cache()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        """
        文本转语音(TTS)的主函数，支持流式和非流式推理模式。
        
        参数说明:
        - text: 输入文本的token表示
        - flow_embedding: 流模型使用的说话人嵌入向量
        - llm_embedding: 大语言模型使用的说话人嵌入向量，默认为零向量
        - prompt_text: 提示文本的token表示，默认为空
        - llm_prompt_speech_token: LLM使用的提示语音token，默认为空
        - flow_prompt_speech_token: 流模型使用的提示语音token，默认为空
        - prompt_speech_feat: 提示语音特征，默认为空
        - stream: 是否使用流式推理模式，默认为False
        - speed: 语音速度调整因子，默认为1.0
        """
        # 创建唯一标识符，用于跟踪当前推理线程的相关变量
        this_uuid = str(uuid.uuid1())
        with self.lock:
            # 初始化与此推理相关的状态变量
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        
        # 启动LLM推理线程，生成语音token序列
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        
        if stream is True:
            # 流式推理模式：边生成边输出语音
            token_offset = 0
            while True:
                time.sleep(0.1)  # 短暂休眠，避免CPU占用过高
                
                # 当累积的token数量足够时，处理一批token
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    # 获取当前批次的token
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    
                    # 将token转换为语音波形
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     token_offset=token_offset,
                                                     finalize=False)
                    
                    # 更新已处理的token偏移量
                    token_offset += self.token_hop_len
                    
                    # 生成器返回当前批次的语音
                    yield {'tts_speech': this_tts_speech.cpu()}
                
                # 当LLM生成结束且剩余token不足以形成新批次时，退出循环
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len:
                    break
            
            # 等待LLM线程完成
            p.join()
            
            # 处理剩余的token
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=token_offset,
                                             finalize=True)
            
            # 返回最后一批语音
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # 非流式推理模式：等待所有token生成完毕后一次性处理
            p.join()
            
            # 将所有token转换为语音波形
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=speed)
            
            # 返回完整的语音
            yield {'tts_speech': this_tts_speech.cpu()}
        
        # 清理资源：删除与此推理相关的临时变量
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
        
        # 释放GPU缓存
        torch.cuda.empty_cache()
