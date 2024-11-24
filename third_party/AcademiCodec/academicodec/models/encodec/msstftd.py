# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""MS-STFT discriminator, provided here for reference."""
import typing as tp

import torch
import torchaudio
from einops import rearrange
from torch import nn

from academicodec.modules import NormConv2d

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(kernel_size: tp.Tuple[int, int],
                   dilation: tp.Tuple[int, int]=(1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, (
        (kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_fft: int=1024,
                 hop_length: int=256,
                 win_length: int=1024,
                 max_filters: int=1024,
                 filters_scale: int=1,
                 kernel_size: tp.Tuple[int, int]=(3, 9),
                 dilations: tp.List=[1, 2, 4],
                 stride: tp.Tuple[int, int]=(1, 2),
                 normalized: bool=True,
                 norm: str='weight_norm',
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm))
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        # print('x ', x.shape)
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        # print('z ', z.shape)
        z = torch.cat([z.real, z.imag], dim=1)
        # print('cat_z ', z.shape)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            # print('z i', i, z.shape)
            fmap.append(z)
        z = self.conv_post(z)
        # print('logit ', z.shape)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_ffts: tp.List[int]=[1024, 2048, 512, 256, 128],
                 hop_lengths: tp.List[int]=[256, 512, 128, 64, 32],
                 win_lengths: tp.List[int]=[1024, 2048, 512, 256, 128],
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_ffts[i],
                win_length=win_lengths[i],
                hop_length=hop_lengths[i],
                **kwargs) for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


def test():
    disc = MultiScaleSTFTDiscriminator(filters=32)
    y = torch.randn(1, 1, 24000)
    y_hat = torch.randn(1, 1, 24000)

    y_disc_r, fmap_r = disc(y)
    y_disc_gen, fmap_gen = disc(y_hat)
    assert len(y_disc_r) == len(y_disc_gen) == len(fmap_r) == len(
        fmap_gen) == disc.num_discriminators

    assert all([len(fm) == 5 for fm in fmap_r + fmap_gen])
    assert all(
        [list(f.shape)[:2] == [1, 32] for fm in fmap_r + fmap_gen for f in fm])
    assert all([len(logits.shape) == 4 for logits in y_disc_r + y_disc_gen])


if __name__ == '__main__':
    test()
