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


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.stream_win_len = 60
        self.stream_hop_len = 50
        self.overlap = 4395 # 10 token equals 4395 sample point
        self.window = np.hamming(2 * self.overlap)

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32), stream=False):
        if stream is True:
            tts_speech_token, cache_speech = [], None
            for i in self.llm.inference(text=text.to(self.device),
                                                text_len=text_len.to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=prompt_text_len.to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                beam_size=1,
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3,
                                                stream=stream):
                tts_speech_token.append(i)
                if len(tts_speech_token) == self.stream_win_len:
                    this_tts_speech_token = torch.concat(tts_speech_token, dim=1)
                    this_tts_mel = self.flow.inference(token=this_tts_speech_token,
                                                token_len=torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                                prompt_token=flow_prompt_speech_token.to(self.device),
                                                prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                                prompt_feat=prompt_speech_feat.to(self.device),
                                                prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                                embedding=flow_embedding.to(self.device))
                    this_tts_speech = self.hift.inference(mel=this_tts_mel).cpu()
                    # fade in/out if necessary
                    if cache_speech is not None:
                        this_tts_speech[:, :self.overlap] = this_tts_speech[:, :self.overlap] * self.window[:self.overlap] + cache_speech * self.window[-self.overlap:]
                    yield  {'tts_speech': this_tts_speech[:, :-self.overlap]}
                    cache_speech = this_tts_speech[:, -self.overlap:]
                    tts_speech_token = tts_speech_token[-(self.stream_win_len - self.stream_hop_len):]
            # deal with remain tokens
            if cache_speech is None or len(tts_speech_token) > self.stream_win_len - self.stream_hop_len:
                this_tts_speech_token = torch.concat(tts_speech_token, dim=1)
                this_tts_mel = self.flow.inference(token=this_tts_speech_token,
                                            token_len=torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                            prompt_token=flow_prompt_speech_token.to(self.device),
                                            prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                            prompt_feat=prompt_speech_feat.to(self.device),
                                            prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                            embedding=flow_embedding.to(self.device))
                this_tts_speech = self.hift.inference(mel=this_tts_mel).cpu()
                if cache_speech is not None:
                    this_tts_speech[:, :self.overlap] = this_tts_speech[:, :self.overlap] * self.window[:self.overlap] + cache_speech * self.window[-self.overlap:]
                yield {'tts_speech': this_tts_speech}
            else:
                assert len(tts_speech_token) == self.stream_win_len - self.stream_hop_len, 'tts_speech_token not equal to {}'.format(self.stream_win_len - self.stream_hop_len)
                yield {'tts_speech': cache_speech}
        else:
            tts_speech_token = []
            for i in self.llm.inference(text=text.to(self.device),
                                                text_len=text_len.to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=prompt_text_len.to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                beam_size=1,
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3,
                                                stream=stream):
                tts_speech_token.append(i)
            assert len(tts_speech_token) == 1, 'tts_speech_token len should be 1 when stream is {}'.format(stream)
            tts_speech_token = torch.concat(tts_speech_token, dim=1)
            tts_mel = self.flow.inference(token=tts_speech_token,
                                        token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                        prompt_token=flow_prompt_speech_token.to(self.device),
                                        prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                        prompt_feat=prompt_speech_feat.to(self.device),
                                        prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                        embedding=flow_embedding.to(self.device))
            tts_speech = self.hift.inference(mel=tts_mel).cpu()
            torch.cuda.empty_cache()
            yield {'tts_speech': tts_speech}
