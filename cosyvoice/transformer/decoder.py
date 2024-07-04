# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2024 Alibaba Inc (Xiang Lyu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Decoder definition."""
from typing import Tuple, List, Optional

import torch
import torch.utils.checkpoint as ckpt
import logging

from cosyvoice.transformer.decoder_layer import DecoderLayer
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from cosyvoice.utils.mask import (subsequent_mask, make_pad_mask)


class TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, encoder-decoder cross attention is not
                       applied, such as CIF model
        key_bias: whether use bias in attention.linear_k, False for whisper models.
        gradient_checkpointing: rerunning a forward-pass segment for each
            checkpointed segment during backward.
        tie_word_embedding: Tie or clone module weights depending of whether we are
            using TorchScript or not
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        src_attention: bool = True,
        key_bias: bool = True,
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        tie_word_embedding: bool = False,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()

        self.embed = torch.nn.Sequential(
            torch.nn.Identity() if input_layer == "no_pos" else
            torch.nn.Embedding(vocab_size, attention_dim),
            COSYVOICE_EMB_CLASSES[input_layer](attention_dim,
                                               positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                COSYVOICE_ATTENTION_CLASSES["selfattn"](
                    attention_heads, attention_dim,
                    self_attention_dropout_rate, key_bias),
                COSYVOICE_ATTENTION_CLASSES["selfattn"](
                    attention_heads, attention_dim, src_attention_dropout_rate,
                    key_bias) if src_attention else None,
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate, activation),
                dropout_rate,
                normalize_before,
            ) for _ in range(self.num_blocks)
        ])

        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embedding = tie_word_embedding

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        if self.gradient_checkpointing and self.training:
            x = self.forward_layers_checkpointed(x, tgt_mask, memory,
                                                 memory_mask)
        else:
            x = self.forward_layers(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_layers(self, x: torch.Tensor, tgt_mask: torch.Tensor,
                       memory: torch.Tensor,
                       memory_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        return x

    @torch.jit.ignore(drop=True)
    def forward_layers_checkpointed(self, x: torch.Tensor,
                                    tgt_mask: torch.Tensor,
                                    memory: torch.Tensor,
                                    memory_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = ckpt.checkpoint(
                layer.__call__, x, tgt_mask, memory, memory_mask)
        return x

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache

    def tie_or_clone_weights(self, jit_mode: bool = True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        if not self.use_output_layer:
            return
        if jit_mode:
            logging.info("clone emb.weight to output.weight")
            self.output_layer.weight = torch.nn.Parameter(
                self.embed[0].weight.clone())
        else:
            logging.info("tie emb.weight with output.weight")
            self.output_layer.weight = self.embed[0].weight

        if getattr(self.output_layer, "bias", None) is not None:
            self.output_layer.bias.data = torch.nn.functional.pad(
                self.output_layer.bias.data,
                (
                    0,
                    self.output_layer.weight.shape[0] -
                    self.output_layer.bias.shape[0],
                ),
                "constant",
                0,
            )


class BiTransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        key_bias: whether use bias in attention.linear_k, False for whisper models.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
        tie_word_embedding: bool = False,
    ):

        super().__init__()
        self.tie_word_embedding = tie_word_embedding
        self.left_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            use_output_layer,
            normalize_before,
            key_bias=key_bias,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embedding=tie_word_embedding)

        self.right_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            r_num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            use_output_layer,
            normalize_before,
            key_bias=key_bias,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embedding=tie_word_embedding)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask,
                                               r_ys_in_pad, ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)

    def tie_or_clone_weights(self, jit_mode: bool = True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        self.left_decoder.tie_or_clone_weights(jit_mode)
        self.right_decoder.tie_or_clone_weights(jit_mode)
