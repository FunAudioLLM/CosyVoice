# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
from typing import Tuple
import torch.nn as nn
from torch.nn import functional as F
from cosyvoice.utils.mask import make_pad_mask


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens=None):
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens
