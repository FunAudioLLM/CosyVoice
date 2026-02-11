"""
TensorRT-LLM DiTBlock implementation
Single transformer block with adaptive layer norm
"""

import numpy as np
import torch
import math
import tensorrt as trt
from tensorrt_llm.module import Module
from tensorrt_llm.layers import Linear, MLP, LayerNorm
from tensorrt_llm.layers.attention import BertAttention
from tensorrt_llm.functional import (Tensor, silu, chunk, unsqueeze, constant, shape, expand,
                                     concat, split, allgather, cast, expand_mask, softmax, matmul, arange,
                                     where, minimum,  embedding, slice as trt_slice)
from tensorrt_llm.functional import stack as trt_stack
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch, trt_dtype_to_str, fp32_array, int32_array
from tensorrt_llm._common import default_net
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.layers.lora import LoraRuntimeParams
from tensorrt_llm.layers.attention import bert_attention


def modulate(x, shift, scale, dtype):
    """
    Modulation helper function (from tensorrt_llm/models/dit/model.py)

    Applies: x * (1 + scale) + shift
    """
    ones = 1.0
    if dtype is not None:
        ones = constant(np.ones(1, dtype=np.float32)).cast(dtype)
    return x * (ones + unsqueeze(scale, 1)) + unsqueeze(shift, 1)


def rotate_half(x):
    """
    Rotate half the hidden dims of the input (for RoPE)

    Matches x-transformers: interleaved pairs [a0, b0, a1, b1, ...] -> [-b0, a0, -b1, a1, ...]
    NOT block rotation!
    """
    # x shape: [B, T, D] where D=64
    # Reshape to [B, T, D//2, 2] to separate pairs
    B = shape(x, 0)
    T = shape(x, 1)
    D = shape(x, 2)

    # Use Python int instead of constant() - auto-converts to match Tensor dtype
    x_reshaped = x.view(concat([B, T, D // 2, 2]))

    # Split into x1 and x2: [B, T, D//2, 2] -> 2 x [B, T, D//2]
    # Use proper Tensor arguments for dynamic slicing
    x1_starts = constant(np.array([0, 0, 0, 0], dtype=np.int32))
    x1_sizes = concat([B, T, D // 2, 1])  # Python ints auto-convert
    x1 = trt_slice(x_reshaped, starts=x1_starts, sizes=x1_sizes)

    x2_starts = constant(np.array([0, 0, 0, 1], dtype=np.int32))
    x2_sizes = concat([B, T, D // 2, 1])  # Python ints auto-convert
    x2 = trt_slice(x_reshaped, starts=x2_starts, sizes=x2_sizes)

    x1 = x1.view(concat([B, T, D // 2]))
    x2 = x2.view(concat([B, T, D // 2]))

    # Stack as [-x2, x1] to create pairs: [B, T, D//2, 2]
    result = trt_stack([-1 * x2, x1], dim=-1)  # [B, T, D//2, 2]

    # Reshape back to [B, T, D]
    result = result.view(concat([B, T, D]))

    return result


def compute_relative_bias(query_length,
                          key_length,
                          num_buckets,
                          max_distance,
                          bidirectional,
                          rel_attn_table,
                          tp_size=1,
                          tp_group=None,
                          tp_rank=None):

    def make_relative_position_bucket(relative_position, bidirectional,
                                      num_buckets, max_distance):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += where(relative_position > 0, num_buckets, 0)
            relative_position = relative_position.abs()
        else:
            relative_position = 0 - minimum(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        max_exact_fp = constant(fp32_array(max_exact))
        tmp = cast(relative_position, "float32") / max_exact_fp
        tmp = tmp.log()
        const1 = math.log(max_distance / max_exact)
        const2 = constant(fp32_array(num_buckets - max_exact))
        relative_position_if_large = tmp / const1 * const2
        relative_position_if_large = cast(relative_position_if_large, "int32")
        relative_position_if_large = max_exact + relative_position_if_large
        relative_position_if_large = minimum(relative_position_if_large,
                                             num_buckets - 1)

        relative_buckets += where(is_small, relative_position,
                                  relative_position_if_large)
        return relative_buckets

    context_position = arange(start=constant(int32_array(0)),
                              end=query_length,
                              dtype=trt_dtype_to_str(trt.int32))
    context_position = unsqueeze(context_position, -1)
    memory_position = arange(start=constant(int32_array(0)),
                             end=key_length,
                             dtype=trt_dtype_to_str(trt.int32))
    memory_position = unsqueeze(memory_position, 0)
    relative_position = memory_position - context_position
    relative_position_bucket = make_relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional,
        num_buckets,
        max_distance,
    )
    # shape (query_length, key_length, num_heads)
    values = embedding(relative_position_bucket,
                       rel_attn_table,
                       tp_size=tp_size,
                       tp_group=tp_group,
                       tp_rank=tp_rank)
    # shape (1, num_heads, query_length, key_length)
    values = unsqueeze(values.permute([2, 0, 1]), 0)
    return values

class CosyVoiceAttention(BertAttention):
    """
    BertAttention with partial RoPE (x-transformers style)

    Only applies RoPE to first dim_head dimensions (head 0),
    matching CosyVoice's x-transformers implementation.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings=1024,
                 num_layers=1,
                 attention_head_size=None,
                 num_kv_heads=None,
                 q_scaling=1.0,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 cp_group=None,
                 cp_size=1,
                 cp_rank=0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0,
                 quant_mode=QuantMode(0)):

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            num_layers=num_layers,
            attention_head_size=attention_head_size,
            num_kv_heads=num_kv_heads,
            q_scaling=q_scaling,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            cp_group=cp_group,
            cp_size=cp_size,
            cp_rank=cp_rank,
            relative_attention=relative_attention,
            max_distance=max_distance,
            num_buckets=num_buckets,
            quant_mode=quant_mode
        )

        # Precompute RoPE frequencies at build time
        # This is constant and only depends on position, not input data
        dim = self.attention_head_size  # 64
        base = 10000.0

        # Precompute RoPE cos/sin at build time
        # This avoids runtime trig computation AND dtype conversion issues
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum('i, j -> i j', t, inv_freq)  # [max_pos, dim//2]
        freqs = torch.stack([freqs, freqs], dim=-1)  # [max_pos, dim//2, 2]
        freqs = freqs.view(max_position_embeddings, dim)  # [max_pos, 64]

        # Precompute cos and sin (avoids runtime trig + ensures correct dtype)
        freqs_cos = torch.cos(freqs)  # [max_pos, 64] - float32 by default
        freqs_sin = torch.sin(freqs)  # [max_pos, 64] - float32 by default

        # Convert to target dtype BEFORE creating Parameter
        if dtype is not None:
            torch_dtype = trt_dtype_to_torch(dtype)
            freqs_cos = freqs_cos.to(torch_dtype)
            freqs_sin = freqs_sin.to(torch_dtype)

        # Store as buffers (is_buffer=True means not loaded from checkpoint)
        self.rope_freqs_cos = Parameter(freqs_cos, dtype=dtype, is_buffer=True)
        self.rope_freqs_sin = Parameter(freqs_sin, dtype=dtype, is_buffer=True)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None,
                lora_layer_params=None):
        assert isinstance(hidden_states, Tensor)

        qkv_lora_params = None
        if lora_layer_params is not None:
            qkv_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_qkv")

        qkv = self.qkv(hidden_states, qkv_lora_params)

        if default_net().plugin_config.remove_input_padding:
            assert qkv.ndim() == 2

        if default_net(
        ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
            q_lora_params = lora_layer_params.get_runtime_params(0, "attn_q")
            k_lora_params = lora_layer_params.get_runtime_params(0, "attn_k")
            v_lora_params = lora_layer_params.get_runtime_params(0, "attn_v")

            assert (q_lora_params is not None and k_lora_params is not None and v_lora_params is not None) or \
                (q_lora_params is None and k_lora_params is None and v_lora_params is None), "q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time."

            if q_lora_params is not None and k_lora_params is not None and v_lora_params is not None:
                qkv_lora_params = LoraRuntimeParams(
                    lora_ranks=[
                        q_lora_params.lora_ranks[0],
                        k_lora_params.lora_ranks[0],
                        v_lora_params.lora_ranks[0],
                    ],
                    lora_weights_pointers=[
                        q_lora_params.lora_weights_pointers[0],
                        k_lora_params.lora_weights_pointers[0],
                        v_lora_params.lora_weights_pointers[0],
                    ],
                    host_request_types=q_lora_params.host_request_types,
                    host_context_lengths=q_lora_params.host_context_lengths)

                q_lora, k_lora, v_lora = self.qkv_lora(hidden_states,
                                                       qkv_lora_params)
                qkv_lora = concat([q_lora, k_lora, v_lora],
                                  dim=q_lora.rank() - 1)
                qkv = qkv + qkv_lora

        B = shape(hidden_states, 0)
        N = shape(hidden_states, 1)  # sequence length

        # Compute input_lengths if not provided
        if input_lengths is None:
            input_lengths = expand(unsqueeze(N, 0).cast('int32'), unsqueeze(B, 0))

        # Split into Q, K, V
        kv_size = self.attention_head_size * self.num_attention_kv_heads
        query, key, value = split(
            qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)

        # ========== Apply Partial RoPE (x-transformers style) ==========
        # Only rotate first dim_head (64) dimensions
        # Query/Key shape: [batch, seq, hidden_size]

        # Slice precomputed cos/sin based on sequence length
        # Build dynamic sizes tensor: [N, 64] where N is dynamic
        slice_starts = constant(np.array([0, 0], dtype=np.int32))
        slice_sizes = concat([N, self.attention_head_size])  # Python int auto-converts

        # Slice precomputed cos and sin (access .value to get the tensor)
        freqs_cos = trt_slice(self.rope_freqs_cos.value,
                             starts=slice_starts,
                             sizes=slice_sizes)  # [seq_len, 64]
        freqs_sin = trt_slice(self.rope_freqs_sin.value,
                             starts=slice_starts,
                             sizes=slice_sizes)  # [seq_len, 64]

        # Broadcast to batch: [seq_len, 64] -> [batch, seq_len, 64]
        freqs_cos = unsqueeze(freqs_cos, 0)  # [1, seq_len, 64]
        freqs_sin = unsqueeze(freqs_sin, 0)  # [1, seq_len, 64]

        # Split query/key into rotated and unrotated parts
        rot_dim = self.attention_head_size  # 64

        # Query - split into rotated (first 64 dims) and unrotated parts
        q_rot_starts = constant(np.array([0, 0, 0], dtype=np.int32))
        q_rot_sizes = concat([B, N, rot_dim])  # Python int auto-converts
        q_rot = trt_slice(query, starts=q_rot_starts, sizes=q_rot_sizes)

        q_unrot_starts = constant(np.array([0, 0, rot_dim], dtype=np.int32))
        q_unrot_sizes = concat([B, N, self.attention_hidden_size - rot_dim])
        q_unrot = trt_slice(query, starts=q_unrot_starts, sizes=q_unrot_sizes)

        # Apply RoPE to first 64 dims (using precomputed cos/sin)
        q_rot = q_rot * freqs_cos + rotate_half(q_rot) * freqs_sin

        # Concat back
        query = concat([q_rot, q_unrot], dim=2)

        # Key - split into rotated (first 64 dims) and unrotated parts
        k_rot_starts = constant(np.array([0, 0, 0], dtype=np.int32))
        k_rot_sizes = concat([B, N, rot_dim])  # Python int auto-converts
        k_rot = trt_slice(key, starts=k_rot_starts, sizes=k_rot_sizes)

        k_unrot_starts = constant(np.array([0, 0, rot_dim], dtype=np.int32))
        k_unrot_sizes = concat([B, N, kv_size - rot_dim])
        k_unrot = trt_slice(key, starts=k_unrot_starts, sizes=k_unrot_sizes)

        # Apply RoPE to first 64 dims (using precomputed cos/sin)
        k_rot = k_rot * freqs_cos + rotate_half(k_rot) * freqs_sin

        # Concat back
        key = concat([k_rot, k_unrot], dim=2)

        # ========== Rebuild QKV and call BertAttention plugin ==========
        qkv = concat([query, key, value], dim=2)

        if default_net().plugin_config.bert_attention_plugin:
            # TRT plugin mode
            assert input_lengths is not None
            context = bert_attention(
                qkv,
                input_lengths,
                self.num_attention_heads,
                self.attention_head_size,
                q_scaling=self.q_scaling,
                relative_attention=self.relative_attention,
                max_distance=self.max_distance,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_input_length=max_input_length,
                cp_group=self.cp_group,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank)
        else:
            # plain TRT mode
            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            kv_size = self.attention_head_size * self.num_attention_kv_heads
            query, key, value = split(
                qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)
            if self.cp_size > 1 and self.cp_group is not None:
                key = allgather(key, self.cp_group, gather_dim=1)
                value = allgather(value, self.cp_group, gather_dim=1)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            key = key.permute([0, 1, 3, 2])
            attention_scores = matmul(query, key, use_fp32_acc=False)
            attention_scores = attention_scores / (self.q_scaling *
                                                   self.norm_factor)

            if self.relative_attention:
                query_len = shape(attention_scores, 2)
                key_len = shape(attention_scores, 3)
                bias = compute_relative_bias(
                    query_len,
                    key_len,
                    self.num_buckets,
                    self.max_distance,
                    True,  # bidirectional
                    self.rel_attn_table.value.transpose(1, 0),
                    tp_size=self.tp_size,
                    tp_group=self.tp_group,
                    tp_rank=self.tp_rank)
                attention_scores = attention_scores + bias

            if attention_mask is not None:
                attention_mask = expand_mask(attention_mask, shape(query, 2))
                attention_mask = cast(attention_mask, attention_scores.dtype)
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1), self.attention_hidden_size
                ]))

        dense_lora_params = None
        if lora_layer_params is not None:
            dense_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_dense")
        context = self.dense(context, lora_runtime_params=dense_lora_params)

        return context


class DiTBlock(Module):
    """
    DiT Transformer Block - matches CosyVoice structure

    Based on: cosyvoice/flow/DiT/modules.py:DiTBlock
    Uses BertAttention with partial RoPE (x-transformers style)

    Original PyTorch:
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(...)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, mult=ff_mult)
    """

    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 ff_mult=4,
                 mapping=Mapping(),
                 dtype=None,
                 max_position_embeddings=1000):
        super().__init__()
        self.dtype = dtype

        # Adaptive LayerNorm for attention (outputs 6 modulation params)
        self.attn_norm_modulation = Linear(
            dim,
            6 * dim,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            bias=True,
            dtype=dtype
        )
        self.attn_norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Self-Attention with partial RoPE
        self.attn = CosyVoiceAttention(
            hidden_size=dim,
            num_attention_heads=heads,
            attention_head_size=dim_head,
            bias=True,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            quant_mode=QuantMode(0),
            max_position_embeddings=max_position_embeddings
        )

        # LayerNorm for feed-forward (no affine parameters)
        self.ff_norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Feed-Forward Network (called 'ff' in CosyVoice, 'mlp' in DiT)
        self.ff = MLP(
            hidden_size=dim,
            ffn_hidden_size=int(dim * ff_mult),
            hidden_act='gelu',
            bias=True,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
        )

    def forward(self, x, t):
        """
        Forward pass - matches CosyVoice structure

        Original PyTorch forward:
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
            attn_output = self.attn(x=norm, mask=mask, rope=rope)
            x = x + gate_msa.unsqueeze(1) * attn_output
            ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(ff_norm)
            x = x + gate_mlp.unsqueeze(1) * ff_output

        Args:
            x: Input tensor [batch, seq_len, dim]
            t: Time embedding [batch, dim]

        Returns:
            x: Output tensor [batch, seq_len, dim]
        """
        # Pre-norm & modulation for attention input
        modulation = self.attn_norm_modulation(silu(t))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(modulation, 6, dim=1)

        norm = modulate(self.attn_norm(x), shift_msa, scale_msa, self.dtype)

        # Attention (partial RoPE applied inside)
        attn_output = self.attn(norm)

        # Process attention output
        x = x + unsqueeze(gate_msa, 1) * attn_output

        # Feed-forward with modulation
        ff_norm = modulate(self.ff_norm(x), shift_mlp, scale_mlp, self.dtype)
        ff_output = self.ff(ff_norm)
        x = x + unsqueeze(gate_mlp, 1) * ff_output

        return x


class FinalLayer(Module):
    """
    Final layer with adaptive layer norm and output projection

    Based on: cosyvoice/flow/DiT/modules.py:AdaLayerNormZero_Final
              and cosyvoice/flow/DiT/dit.py (norm_out + proj_out)

    Original PyTorch:
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
    """

    def __init__(self,
                 dim,
                 out_dim,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.dtype = dtype

        # AdaLayerNormZero_Final modulation (outputs 2 params: scale, shift)
        self.norm_out_modulation = Linear(
            dim,
            2 * dim,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            bias=True,
            dtype=dtype
        )

        # LayerNorm (no affine parameters)
        self.norm_out = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Output projection (called 'proj_out' in CosyVoice)
        self.proj_out = Linear(
            dim,
            out_dim,
            bias=True,
            dtype=dtype
        )

    def forward(self, x, t):
        """
        Forward pass - matches CosyVoice structure

        Original PyTorch forward:
            x = self.norm_out(x, t)
            output = self.proj_out(x)

        Args:
            x: Input tensor [batch, seq_len, dim]
            t: Time embedding [batch, dim]

        Returns:
            Output tensor [batch, seq_len, out_dim]
        """
        # Compute modulation parameters
        modulation = self.norm_out_modulation(silu(t))
        scale, shift = chunk(modulation, 2, dim=1)

        x = modulate(self.norm_out(x), shift, scale, self.dtype)

        # Output projection
        output = self.proj_out(x)

        return output
