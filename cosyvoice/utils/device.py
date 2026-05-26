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
"""Unified device management for CUDA, MPS (Apple Silicon), and CPU backends."""

import random
from contextlib import nullcontext

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def is_cuda() -> bool:
    return torch.cuda.is_available()


def is_mps() -> bool:
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def is_gpu_available() -> bool:
    return is_cuda() or is_mps()


def get_stream_context(device: torch.device):
    """Return a CUDA stream context or nullcontext for non-CUDA devices."""
    if device.type == 'cuda':
        return torch.cuda.stream(torch.cuda.Stream(device))
    return nullcontext()


def get_autocast_context(enabled: bool, device: torch.device):
    """Return the appropriate autocast context for the device."""
    if not enabled:
        return nullcontext()
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(enabled=True)
    if device.type == 'mps':
        return torch.autocast(device_type='mps', dtype=torch.float16)
    return nullcontext()


def empty_cache(device: torch.device):
    """Clear device cache and synchronize."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.current_stream().synchronize()
    elif device.type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()


def set_all_random_seed(seed: int):
    """Set random seed across all available backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
