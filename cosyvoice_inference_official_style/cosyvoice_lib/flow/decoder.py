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
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import logging # Added
import onnxruntime # Added
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from ..utils.common import mask_to_bias
from ..utils.mask import add_optional_chunk_mask
from matcha.models.components.decoder import SinusoidalPosEmb, Block1D, ResnetBlock1D, Downsample1D, TimestepEmbedding, Upsample1D
from matcha.models.components.transformer import BasicTransformerBlock


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_padding, 0), value=0.0)
        x = super(CausalConv1d, self).forward(x)
        return x


class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = ResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class CausalConditionalDecoder(ConditionalDecoder):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        static_chunk_size=50,
        num_decoding_left_chunks=2,
        # ONNX specific params
        onnx_model_path: Optional[str] = None,
        onnx_providers: Optional[List[str]] = None,
    ):
        torch.nn.Module.__init__(self)

        _channels_tuple = tuple(channels)
        # self.in_channels is the number of channels for the UNet input after all conditions are packed.
        # This should come from the YAML config, e.g., decoder_conf.in_channels (like 240)
        self.in_channels = in_channels
        self.out_channels = out_channels # Final output channels, e.g., 80 for mel

        # Time embeddings in_channels should be based on the UNet's operating channel dimension,
        # not necessarily the raw input 'x' (e.g. 80 for mel) if packing happens before time embedding.
        # The original ConditionalDecoder uses its 'in_channels' for SinusoidalPosEmb.
        # If CausalConditionalDecoder's 'in_channels' argument *is* the UNet's main channel dim (e.g. 240), this is fine.
        self.time_embeddings = SinusoidalPosEmb(self.in_channels)
        time_embed_dim = _channels_tuple[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=self.in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # --- Start of PyTorch UNet Layer Definitions (from original CausalConditionalDecoder) ---
        current_unet_channel = self.in_channels # This is the channel dim for the first layer of UNet
        for i in range(len(_channels_tuple)):
            input_ch_unet = current_unet_channel
            output_ch_unet = _channels_tuple[i]
            is_last = i == len(_channels_tuple) - 1
            resnet = CausalResnetBlock1D(dim=input_ch_unet, dim_out=output_ch_unet, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_ch_unet, num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim, dropout=dropout, activation_fn=act_fn,
                    ) for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_ch_unet) if not is_last else CausalConv1d(output_ch_unet, output_ch_unet, 3)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
            current_unet_channel = output_ch_unet

        for _ in range(num_mid_blocks):
            resnet = CausalResnetBlock1D(dim=current_unet_channel, dim_out=current_unet_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=current_unet_channel, num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim, dropout=dropout, activation_fn=act_fn,
                    ) for _ in range(n_blocks)
                ]
            )
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        reversed_channels_unet = _channels_tuple[::-1] + (_channels_tuple[0],)
        for i in range(len(reversed_channels_unet) - 1):
            # Input channel for up-block resnet is current_unet_channel (from previous layer) + skip connection channel
            # Skip connection channel is _channels_tuple[len(_channels_tuple) - 1 - i]
            input_ch_unet = current_unet_channel + _channels_tuple[len(_channels_tuple) - 1 - i] if i < len(_channels_tuple) else current_unet_channel # Correct skip logic needed
            # More simply, the input to ResNet in up block is `channels[i] * 2` from original `ConditionalDecoder`
            # where `channels` is the reversed list. So `reversed_channels_unet[i] * 2`.
            input_ch_unet = reversed_channels_unet[i] * 2
            output_ch_unet = reversed_channels_unet[i+1]
            is_last = i == len(reversed_channels_unet) - 2
            resnet = CausalResnetBlock1D(dim=input_ch_unet, dim_out=output_ch_unet, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_ch_unet, num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim, dropout=dropout, activation_fn=act_fn,
                    ) for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_ch_unet, use_conv_transpose=True) if not is_last
                else CausalConv1d(output_ch_unet, output_ch_unet, 3)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
            current_unet_channel = output_ch_unet

        self.final_block = CausalBlock1D(current_unet_channel, current_unet_channel)
        self.final_proj = nn.Conv1d(current_unet_channel, self.out_channels, 1)
        # --- End of PyTorch UNet Layer Definitions ---

        # Call initialize_weights from ConditionalDecoder (base class) or define it here
        self.initialize_weights()

        # ONNX Session
        self.onnx_session = None
        if onnx_model_path:
            assert torch.cuda.is_available(), "CUDA is required for ONNX flow decoder estimator."
            if onnx_providers is None:
                onnx_providers = ['CUDAExecutionProvider']
            try:
                self.onnx_session = onnxruntime.InferenceSession(onnx_model_path, providers=onnx_providers)
                logging.info(f"ONNX session initialized for CausalConditionalDecoder: {onnx_model_path} with providers: {self.onnx_session.get_providers()}")
            except Exception as e:
                logging.error(f"Failed to initialize ONNX session for CausalConditionalDecoder {onnx_model_path}: {e}")
                self.onnx_session = None

    def initialize_weights(self): # Added from ConditionalDecoder
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        if self.onnx_session:
            ort_inputs = {
                'x': x.cpu().numpy(),
                'mask': mask.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                't': t.cpu().numpy()
            }
            # The ONNX model expects 6 inputs: x, mask, mu, t, spks, cond
            # These are direct inputs to the "estimator"
            # Check export_onnx.py: dummy spks is (B, EstOutChannels), cond is (B, EstOutChannels, SeqLen)
            # `EstOutChannels` is likely the feature dim (e.g. 80 for mel) if conditions are not pre-packed for ONNX.
            # `spks` to this function is (B, ProjSpkDim e.g. 80).
            # `cond` to this function is (B, MelChannels, PromptSeqLen e.g. 80).
            # This seems to align.

            onnx_input_names = {inp.name for inp in self.onnx_session.get_inputs()}

            if 'spks' in onnx_input_names:
                if spks is not None:
                    ort_inputs['spks'] = spks.cpu().numpy()
                else:
                    raise ValueError("ONNX model requires 'spks' input, but it was None.")

            if 'cond' in onnx_input_names:
                if cond is not None:
                    ort_inputs['cond'] = cond.cpu().numpy()
                else:
                    raise ValueError("ONNX model requires 'cond' input, but it was None.")

            missing_inputs = onnx_input_names - set(ort_inputs.keys())
            if missing_inputs:
                raise ValueError(f"ONNX model requires inputs {missing_inputs} which were not provided.")

            output_onnx = self.onnx_session.run(None, ort_inputs)[0]
            return torch.from_numpy(output_onnx).to(x.device)

        # PyTorch Path (original CausalConditionalDecoder logic)
        t_emb = self.time_embeddings(t).to(t.dtype)
        t_emb = self.time_mlp(t_emb)

        # Packing logic from original ConditionalDecoder.forward
        # x is (B, C_mel, T), mu is (B, C_encoder_proj, T)
        # spks is (B, C_spk_proj), cond is (B, C_mel, T_prompt_aligned)
        # All these C_... should sum up to self.in_channels (e.g. 240) for the UNet.
        x_packed = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks_repeated = repeat(spks, "b c -> b c t", t=x_packed.shape[-1])
            x_packed = pack([x_packed, spks_repeated], "b * t")[0]
        if cond is not None:
            _cond = cond
            if cond.shape[-1] != x_packed.shape[-1]: # Align cond length
                if cond.shape[-1] < x_packed.shape[-1]:
                    _cond = F.pad(cond, (0, x_packed.shape[-1] - cond.shape[-1]))
                else:
                    _cond = cond[:, :, :x_packed.shape[-1]]
            x_packed = pack([x_packed, _cond], "b * t")[0]

        # UNet processing (from original CausalConditionalDecoder.forward)
        current_x = x_packed
        hiddens = []
        masks_unet = [mask.clone()]

        for resnet, transformer_blocks, downsample_layer in self.down_blocks:
            mask_down = masks_unet[-1]
            current_x = resnet(current_x, mask_down, t_emb)
            current_x_rearranged = rearrange(current_x, "b c t -> b t c").contiguous()

            attn_mask_val = add_optional_chunk_mask(current_x_rearranged, mask_down.bool(), False, False, 0,
                                                  self.static_chunk_size if streaming else 0, -1)
            if not streaming:
                attn_mask_val = attn_mask_val.repeat(1, current_x_rearranged.size(1), 1)
            attn_mask_val = mask_to_bias(attn_mask_val, current_x_rearranged.dtype)

            for transformer_block in transformer_blocks:
                current_x_rearranged = transformer_block(hidden_states=current_x_rearranged, attention_mask=attn_mask_val, timestep=t_emb)
            current_x = rearrange(current_x_rearranged, "b t c -> b c t").contiguous()
            hiddens.append(current_x)
            current_x = downsample_layer(current_x * mask_down)
            masks_unet.append(mask_down[:, :, ::2])

        masks_unet = masks_unet[:-1]
        mask_mid = masks_unet[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            current_x = resnet(current_x, mask_mid, t_emb)
            current_x_rearranged = rearrange(current_x, "b c t -> b t c").contiguous()

            attn_mask_val = add_optional_chunk_mask(current_x_rearranged, mask_mid.bool(), False, False, 0,
                                                  self.static_chunk_size if streaming else 0, -1)
            if not streaming:
                attn_mask_val = attn_mask_val.repeat(1, current_x_rearranged.size(1), 1)
            attn_mask_val = mask_to_bias(attn_mask_val, current_x_rearranged.dtype)

            for transformer_block in transformer_blocks:
                current_x_rearranged = transformer_block(hidden_states=current_x_rearranged, attention_mask=attn_mask_val, timestep=t_emb)
            current_x = rearrange(current_x_rearranged, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample_layer in self.up_blocks:
            mask_up = masks_unet.pop()
            skip = hiddens.pop()

            # Ensure skip connection matches current_x's length for packing
            # This part of skip connection handling might need adjustment based on actual layer outputs
            # Original ConditionalDecoder directly packs.
            if skip.shape[-1] != current_x.shape[-1]: # Adjust current_x if upsample changed length non-compatibly
                 current_x = F.interpolate(current_x, size=skip.shape[-1], mode='nearest')

            current_x = pack([current_x, skip], "b * t")[0]
            current_x = resnet(current_x, mask_up, t_emb)
            current_x_rearranged = rearrange(current_x, "b c t -> b t c").contiguous()

            attn_mask_val = add_optional_chunk_mask(current_x_rearranged, mask_up.bool(), False, False, 0,
                                                  self.static_chunk_size if streaming else 0, -1)
            if not streaming:
                 attn_mask_val = attn_mask_val.repeat(1, current_x_rearranged.size(1), 1)
            attn_mask_val = mask_to_bias(attn_mask_val, current_x_rearranged.dtype)

            for transformer_block in transformer_blocks:
                current_x_rearranged = transformer_block(hidden_states=current_x_rearranged, attention_mask=attn_mask_val, timestep=t_emb)
            current_x = rearrange(current_x_rearranged, "b t c -> b c t").contiguous()
            current_x = upsample_layer(current_x * mask_up)

        output = self.final_proj(self.final_block(current_x, mask_up) * mask_up)
        return output * mask
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
