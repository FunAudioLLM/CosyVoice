"""
TensorRT-LLM model for CosyVoice DiT

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorrt as trt
from collections import OrderedDict

from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.models.modeling_utils import PretrainedModel, PretrainedConfig
from tensorrt_llm.functional import Tensor
from tensorrt_llm._utils import str_dtype_to_trt

from modules_trt import TimestepEmbedding, InputEmbedding
from dit_block_trt import DiTBlock, FinalLayer


class CosyVoiceDiT(PretrainedModel):
    """
    CosyVoice DiT model for TensorRT-LLM

    Phase 2: Embeddings + 1 Transformer Block + Output
    """

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        super().__init__(config)

        self.dim = config.hidden_size
        self.mel_dim = config.mel_dim
        self.mu_dim = config.mu_dim if hasattr(config, 'mu_dim') else config.mel_dim
        self.spk_dim = config.spk_dim
        self.dtype = str_dtype_to_trt(config.dtype)

        # Get architecture parameters
        self.heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else 16
        self.dim_head = self.dim // self.heads
        self.ff_mult = getattr(config, 'ff_mult', 2)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 1000)
        self.mapping = config.mapping

        # Matches CosyVoice naming: time_embed, input_embed, transformer_blocks, norm_out, proj_out
        self.time_embed = TimestepEmbedding(self.dim, dtype=self.dtype)
        self.input_embed = InputEmbedding(
            mel_dim=self.mel_dim,
            text_dim=self.mu_dim,
            out_dim=self.dim,
            spk_dim=self.spk_dim,
            dtype=self.dtype
        )

        self.transformer_blocks = []

        for i in range(22):
            self.transformer_blocks.append(DiTBlock(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                ff_mult=self.ff_mult,
                mapping=self.mapping,
                dtype=self.dtype,
                max_position_embeddings=self.max_position_embeddings,
            ))
        self.transformer_blocks = ModuleList(self.transformer_blocks)

        # Final output layer (matches CosyVoice naming: norm_out + proj_out)
        self.final_layer = FinalLayer(
            dim=self.dim,
            out_dim=self.mel_dim,
            mapping=self.mapping,
            dtype=self.dtype
        )

    def check_config(self, config: PretrainedConfig):
        """Set default config values (from actual CosyVoice model)"""
        config.set_if_not_exist('hidden_size', 1024)
        config.set_if_not_exist('mel_dim', 80)
        config.set_if_not_exist('mu_dim', None)
        config.set_if_not_exist('spk_dim', 80)
        config.set_if_not_exist('dtype', 'float16')
        config.set_if_not_exist('num_attention_heads', 16)  # Actual: 16 heads
        config.set_if_not_exist('num_hidden_layers', 22)  # 22 DiTBlocks in actual model
        config.set_if_not_exist('ff_mult', 2)  # Actual: 2048/1024 = 2, not 4!
        config.set_if_not_exist('max_position_embeddings', 1000)

    def forward(self, x, mu, t, spks, cond):
        """
        Forward pass - Phase 2: Embeddings + 1 Block + Output

        Args:
            x: Noised mel-spec input [batch, mel_dim, seq_len]
            mu: Text embeddings [batch, mu_dim, seq_len]
            t: Timestep [batch]
            spks: Speaker embeddings [batch, spk_dim]
            cond: Conditional audio [batch, mel_dim, seq_len]

        Returns:
            output: Predicted noise [batch, mel_dim, seq_len]
        """
        # Transpose inputs from [b, c, n] to [b, n, c]
        x = x.transpose(1, 2)      # [b, seq_len, mel_dim]
        mu = mu.transpose(1, 2)    # [b, seq_len, mu_dim]
        cond = cond.transpose(1, 2)  # [b, seq_len, mel_dim]

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, hidden_size]

        # Input embedding
        x_emb = self.input_embed(x, cond, mu, spks)  # [batch, seq_len, hidden_size]

        # Pass through 1 transformer block (RoPE applied inside)
        for block in self.transformer_blocks:
            x_emb = block(x_emb, t_emb)

        # Final layer with time conditioning
        output = self.final_layer(x_emb, t_emb)  # [batch, seq_len, mel_dim]

        # Transpose back to [batch, mel_dim, seq_len]
        output = output.transpose(1, 2)

        # Mark output
        output.mark_output('output', self.dtype)

        return output

    def prepare_inputs(self, max_batch_size, max_seq_len, **kwargs):
        """
        Prepare input tensors with dynamic shapes

        Args:
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
        """
        def default_range(max_val):
            return [1, (max_val + 1) // 2, max_val]

        # Noised mel-spec input
        x = Tensor(
            name='x',
            dtype=self.dtype,
            shape=[-1, self.mel_dim, -1],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('mel_dim', [[self.mel_dim] * 3]),
                ('seq_len', [default_range(max_seq_len)]),
            ])
        )

        # Text embeddings
        mu = Tensor(
            name='mu',
            dtype=self.dtype,
            shape=[-1, self.mu_dim, -1],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('mu_dim', [[self.mu_dim] * 3]),
                ('seq_len', [default_range(max_seq_len)]),
            ])
        )

        # Timestep
        t = Tensor(
            name='t',
            dtype=trt.float32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
            ])
        )

        # Speaker embeddings
        spks = Tensor(
            name='spks',
            dtype=self.dtype,
            shape=[-1, self.spk_dim],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('spk_dim', [[self.spk_dim] * 3]),
            ])
        )

        # Conditional audio
        cond = Tensor(
            name='cond',
            dtype=self.dtype,
            shape=[-1, self.mel_dim, -1],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('mel_dim', [[self.mel_dim] * 3]),
                ('seq_len', [default_range(max_seq_len)]),
            ])
        )

        return {
            'x': x,
            'mu': mu,
            't': t,
            'spks': spks,
            'cond': cond
        }
