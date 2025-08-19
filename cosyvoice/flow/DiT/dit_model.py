"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from x_transformers.x_transformers import RotaryEmbedding
from funasr.models.transformer.utils.mask import causal_block_mask

from cosyvoice.flow.DiT.dit_modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    CausalConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(
            self,
            x: float["b n d"],
            cond: float["b n d"],
            text_embed: float["b n d"],
            spks: float["b d"],
    ):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)

        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        **kwargs
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.causal_mask_type = kwargs.get("causal_mask_type", None)

    def build_mix_causal_mask(self, attn_mask, rand=None, ratio=None):
        b, _, _, t = attn_mask.shape
        if rand is None:
            rand = torch.rand((b, 1, 1, 1), device=attn_mask.device, dtype=torch.float32)
        mixed_mask = attn_mask.clone()
        for item in self.causal_mask_type:
            prob_min, prob_max = item["prob_min"], item["prob_max"]
            _ratio = 1
            if "ratio" in item:
                _ratio = item["ratio"]
            if ratio is not None:
                _ratio = ratio
            block_size = item["block_size"] * _ratio
            if block_size <= 0:
                causal_mask = attn_mask
            else:
                causal_mask = causal_block_mask(
                    t, block_size, attn_mask.device, torch.float32
                ).unsqueeze(0).unsqueeze(1)  # 1,1,T,T
            flag = (prob_min <= rand) & (rand < prob_max)
            mixed_mask = mixed_mask * (~flag) + (causal_mask * attn_mask) * flag

        return mixed_mask

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        mu: int["b nt d"],  # mu
        spks: float["b 1 d"],  # spk xvec
        time: float["b"] | float[""],  # time step
        return_hidden: bool = False,
        mask: bool["b 1 n"] | None = None,
        mask_rand: float["b 1 1"] = None,  # for mask flag type
        **kwargs,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        mask = mask.unsqueeze(1)  # B,1,1,T
        if self.causal_mask_type is not None:
            mask = self.build_mix_causal_mask(mask, rand=mask_rand.unsqueeze(-1))

        for block in self.transformer_blocks:
            # mask-out padded values for amp training
            x = x * mask[:, 0, -1, :].unsqueeze(-1)
            x = block(x, t, mask=mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if return_hidden:
            return output, None

        return output
