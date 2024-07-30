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
import torch
import numpy as np
import threading
import time
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.token_min_hop_len = 100
        self.token_max_hop_len = 400
        self.token_overlap_len = 20
        self.speech_overlap_len = 34 * 256
        self.window = np.hamming(2 * self.speech_overlap_len)
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.flow_hift_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token = {}
        self.llm_end = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def llm_job(self, text, text_len, prompt_text, prompt_text_len, llm_prompt_speech_token, llm_prompt_speech_token_len, llm_embedding, this_uuid):
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                                text_len=text_len.to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=prompt_text_len.to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3):
                self.tts_speech_token[this_uuid].append(i)
        self.llm_end[this_uuid] = True

    def token2wav(self, token, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding):
        with self.flow_hift_context:
            tts_mel = self.flow.inference(token=token.to(self.device),
                                        token_len=torch.tensor([token.size(1)], dtype=torch.int32).to(self.device),
                                        prompt_token=prompt_token.to(self.device),
                                        prompt_token_len=prompt_token_len.to(self.device),
                                        prompt_feat=prompt_feat.to(self.device),
                                        prompt_feat_len=prompt_feat_len.to(self.device),
                                        embedding=embedding.to(self.device))
            tts_speech = self.hift.inference(mel=tts_mel).cpu()
        return tts_speech

    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32), stream=False):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token[this_uuid], self.llm_end[this_uuid] = [], False
        p = threading.Thread(target=self.llm_job, args=(text.to(self.device), text_len.to(self.device), prompt_text.to(self.device), prompt_text_len.to(self.device),
                                                    llm_prompt_speech_token.to(self.device), llm_prompt_speech_token_len.to(self.device), llm_embedding.to(self.device), this_uuid))
        p.start()
        if stream is True:
            cache_speech, cache_token, token_hop_len = None, None, self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.concat(self.tts_speech_token[this_uuid][:token_hop_len + self.token_overlap_len], dim=1)
                    with self.flow_hift_context:
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                    prompt_token=flow_prompt_speech_token.to(self.device),
                                                    prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                                    prompt_feat=prompt_speech_feat.to(self.device),
                                                    prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                                    embedding=flow_embedding.to(self.device))
                    # fade in/out if necessary
                    if cache_speech is not None:
                        this_tts_speech = fade_in_out(this_tts_speech, cache_speech, self.window)
                    yield  {'tts_speech': this_tts_speech[:, :-self.speech_overlap_len]}
                    cache_speech = this_tts_speech[:, -self.speech_overlap_len:]
                    cache_token = self.tts_speech_token[this_uuid][:token_hop_len]
                    with self.lock:
                        self.tts_speech_token[this_uuid] = self.tts_speech_token[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end[this_uuid] is True and len(self.tts_speech_token[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.concat(self.tts_speech_token[this_uuid], dim=1)
            if this_tts_speech_token.shape[1] < self.token_min_hop_len + self.token_overlap_len and cache_token is not None:
                cache_token_len = self.token_min_hop_len + self.token_overlap_len - this_tts_speech_token.shape[1]
                this_tts_speech_token = torch.concat([torch.concat(cache_token[-cache_token_len:], dim=1), this_tts_speech_token], dim=1)
            else:
                cache_token_len = 0
            with self.flow_hift_context:
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token.to(self.device),
                                            prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                            prompt_feat=prompt_speech_feat.to(self.device),
                                            prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                            embedding=flow_embedding.to(self.device))
                this_tts_speech = this_tts_speech[:, int(cache_token_len / this_tts_speech_token.shape[1] * this_tts_speech.shape[1]):]
            if cache_speech is not None:
                this_tts_speech = fade_in_out(this_tts_speech, cache_speech, self.window)
            yield {'tts_speech': this_tts_speech}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.concat(self.tts_speech_token[this_uuid], dim=1)
            with self.flow_hift_context:
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token.to(self.device),
                                            prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                            prompt_feat=prompt_speech_feat.to(self.device),
                                            prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                            embedding=flow_embedding.to(self.device))
            yield {'tts_speech': this_tts_speech}
        with self.lock:
            self.tts_speech_token.pop(this_uuid)
            self.llm_end.pop(this_uuid)
        torch.cuda.synchronize()
