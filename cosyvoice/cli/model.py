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

class CosyVoiceModel:
    def __init__(self, llm: torch.nn.Module, flow: torch.nn.Module, hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.models = [self.llm, self.flow, self.hift]  # 存储所有模型以便于统一管理

    def load(self, llm_model, flow_model, hift_model):
        try:
            self.load_model(self.llm, llm_model)
            self.load_model(self.flow, flow_model)
            self.load_model(self.hift, hift_model)
        except Exception as e:
            print(f"An error occurred during model loading: {e}")

    def load_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()

    def inference(self, text, text_len, flow_embedding, **kwargs):
        try:
            # 将所有输入数据一次性转移到GPU
            text = text.to(self.device)
            text_len = text_len.to(self.device)
            # 处理kwargs中的所有参数
            params = {k: v.to(self.device) if v is not None else v for k, v in kwargs.items()}

            # 执行推理
            tts_speech_token = self.llm.inference(
                text=text,
                text_len=text_len,
                **params,
                embedding=flow_embedding.to(self.device),
                beam_size=1,
                sampling=25,
                max_token_text_ratio=30,
                min_token_text_ratio=3
            )

            tts_mel = self.flow.inference(
                token=tts_speech_token,
                token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                **params,
                embedding=flow_embedding.to(self.device)
            )

            tts_speech = self.hift.inference(mel=tts_mel).cpu()

            return {'tts_speech': tts_speech}
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            # 可以在这里添加更多的错误处理逻辑

    def clear_cache(self):
        # 清理未使用的缓存
        torch.cuda.empty_cache()

# 使用示例
# cosy_voice_model = CosyVoiceModel(llm_module, flow_module, hift_module)
# cosy_voice_model.load(llm_model_path, flow_model_path, hift_model_path)
# result = cosy_voice_model.inference(text_tensor, text_len_tensor, flow_embedding_tensor)
# cosy_voice_model.clear_cache()
