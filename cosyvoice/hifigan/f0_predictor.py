# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
from cosyvoice.transformer.convolution import CausalConv1d


class ConvRNNF0Predictor(nn.Module):
    def __init__(self,
                 num_class: int = 1,
                 in_channels: int = 80,
                 cond_channels: int = 512
                 ):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self,
                 num_class: int = 1,
                 in_channels: int = 80,
                 cond_channels: int = 512
                 ):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type='right')
            ),
            nn.ELU(),
            weight_norm(
                CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')
            ),
            nn.ELU(),
            weight_norm(
                CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')
            ),
            nn.ELU(),
            weight_norm(
                CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')
            ),
            nn.ELU(),
            weight_norm(
                CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type='left')
            ),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        if finalize is True:
            x = self.condnet[0](x)
        else:
            x = self.condnet[0](x[:, :, :-self.condnet[0].causal_padding], x[:, :, -self.condnet[0].causal_padding:])
        for i in range(1, len(self.condnet)):
            x = self.condnet[i](x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))
