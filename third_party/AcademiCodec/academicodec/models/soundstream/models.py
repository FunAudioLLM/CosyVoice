import torch
import torch.nn as nn
import torch.nn.functional as F
from academicodec.modules import NormConv1d
from academicodec.modules import NormConv2d
from academicodec.utils import get_padding
from torch.nn import AvgPool1d
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self,
                 period,
                 kernel_size=5,
                 stride=3,
                 use_spectral_norm=False,
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList([
            NormConv2d(
                1,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(
                32,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(
                32,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(
                32,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(32, 32, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = NormConv2d(32, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self,
                 use_spectral_norm=False,
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super(DiscriminatorS, self).__init__()
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList([
            NormConv1d(1, 32, 15, 1, padding=7),
            NormConv1d(32, 32, 41, 2, groups=4, padding=20),
            NormConv1d(32, 32, 41, 2, groups=16, padding=20),
            NormConv1d(32, 32, 41, 4, groups=16, padding=20),
            NormConv1d(32, 32, 41, 4, groups=16, padding=20),
            NormConv1d(32, 32, 41, 1, groups=16, padding=20),
            NormConv1d(32, 32, 5, 1, padding=2),
        ])
        self.conv_post = NormConv1d(32, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
