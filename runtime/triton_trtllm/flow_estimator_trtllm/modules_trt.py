"""
TensorRT-LLM modules for CosyVoice DiT
Converted from cosyvoice/flow/DiT/modules.py
"""

import math
import numpy as np
import tensorrt as trt

from tensorrt_llm.module import Module
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.layers import Linear, Conv1d
from tensorrt_llm.layers.activation import Mish
from tensorrt_llm.functional import (
    concat, cos, sin, arange, unsqueeze, pad, silu, constant
)
from tensorrt_llm._utils import str_dtype_to_trt


# Sinusoidal Position Embedding
class SinusPositionEmbedding(Module):
    def __init__(self, dim, dtype=None):
        super().__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x, scale=1000):
        """
        Args:
            x: Tensor of shape [batch]
            scale: Scaling factor (default 1000)
        Returns:
            Embedding of shape [batch, dim]
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)

        # Create frequency tensor
        emb_const = constant(np.exp(np.arange(half_dim, dtype=np.float32) * -emb))

        # Compute in float32 for numerical stability
        x_expanded = unsqueeze(x.cast(trt.float32), -1)  # [batch, 1]
        emb_expanded = unsqueeze(emb_const, 0)  # [1, half_dim]

        emb_result = x_expanded * scale * emb_expanded  # [batch, half_dim]

        # Concatenate sin and cos
        emb_sin = sin(emb_result)
        emb_cos = cos(emb_result)
        result = concat([emb_sin, emb_cos], dim=-1)  # [batch, dim]

        # Cast to model dtype (following DiT pattern)
        if self.dtype is not None:
            result = result.cast(self.dtype)

        return result


# Timestep Embedding
class TimestepEmbedding(Module):
    def __init__(self, dim, freq_embed_dim=256, dtype=None):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim, dtype=dtype)
        self.time_mlp = Linear(freq_embed_dim, dim, bias=True, dtype=dtype)
        self.time_mlp2 = Linear(dim, dim, bias=True, dtype=dtype)

    def forward(self, timestep):
        """
        Args:
            timestep: Tensor of shape [batch]
        Returns:
            Time embedding of shape [batch, dim]
        """
        time_hidden = self.time_embed(timestep)
        time_hidden = self.time_mlp(time_hidden)
        time_hidden = silu(time_hidden)
        time = self.time_mlp2(time_hidden)
        return time


# Causal Convolutional Position Embedding
class CausalConvPositionEmbedding(Module):
    def __init__(self, dim, kernel_size=31, groups=16, dtype=None):
        super().__init__()
        assert kernel_size % 2 != 0, "kernel_size must be odd"
        self.kernel_size = kernel_size

        # First conv block
        self.conv1 = Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=groups,
            padding=0,
            bias=True,
            dtype=dtype
        )
        self.mish1 = Mish()

        # Second conv block
        self.conv2 = Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=groups,
            padding=0,
            bias=True,
            dtype=dtype
        )
        self.mish2 = Mish()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, dim]
        Returns:
            Output tensor of shape [batch, seq_len, dim]
        """
        # Permute to [batch, dim, seq_len] for Conv1d
        x = x.transpose(1, 2)  # [b, d, n]

        # First conv block with causal padding
        x = pad(x, [self.kernel_size - 1, 0])
        x = self.conv1(x)
        x = self.mish1(x)

        # Second conv block with causal padding
        x = pad(x, [self.kernel_size - 1, 0])
        x = self.conv2(x)
        x = self.mish2(x)

        # Permute back to [batch, seq_len, dim]
        out = x.transpose(1, 2)  # [b, n, d]

        return out


# Input Embedding
class InputEmbedding(Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim, dtype=None):
        super().__init__()
        self.spk_dim = spk_dim
        self.proj = Linear(mel_dim * 2 + text_dim + spk_dim, out_dim, bias=True, dtype=dtype)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim, dtype=dtype)

    def forward(self, x, cond, text_embed, spks):
        """
        Args:
            x: Noised mel-spec, shape [batch, seq_len, mel_dim]
            cond: Conditional audio, shape [batch, seq_len, mel_dim]
            text_embed: Text embeddings, shape [batch, seq_len, text_dim]
            spks: Speaker embeddings, shape [batch, spk_dim]
        Returns:
            Combined embedding of shape [batch, seq_len, out_dim]
        """
        from tensorrt_llm.functional import expand, shape as get_shape

        # Repeat speaker embeddings for each timestep
        # spks: [b, spk_dim] -> [b, 1, spk_dim] -> [b, seq_len, spk_dim]
        spks_expanded = unsqueeze(spks, 1)  # [b, 1, spk_dim]

        # Expand to match sequence length (much simpler!)
        # Build target shape: [batch, seq_len, spk_dim]
        target_shape = concat([
            get_shape(x, 0),           # batch
            get_shape(x, 1),           # seq_len (dynamic!)
            get_shape(spks_expanded, 2) # spk_dim
        ])
        spks_tiled = expand(spks_expanded, target_shape)

        # Concatenate all inputs
        combined = concat([x, cond, text_embed, spks_tiled], dim=-1)

        # Project
        x = self.proj(combined)

        # Add convolutional positional embedding
        x = self.conv_pos_embed(x) + x

        return x
