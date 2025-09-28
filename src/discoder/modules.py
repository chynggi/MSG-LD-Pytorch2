import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F

from discoder.filter import LowPassFilter1d, kaiser_sinc_filter1d


@torch.jit.script
def snake(x, alpha):
    """
    Implementation from https://github.com/descriptinc/descript-audio-codec
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """
    Implementation from https://github.com/descriptinc/descript-audio-codec
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class Activation1d(nn.Module):
    """
    Implementation from https://github.com/NVIDIA/BigVGAN
    """
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class UpSample1d(nn.Module):
    """
    Implementation from https://github.com/NVIDIA/BigVGAN
    """

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x


class DownSample1d(nn.Module):
    """
    Implementation from https://github.com/NVIDIA/BigVGAN
    """

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


class ResBlock(nn.Module):
    """
    Implementation adapted from https://github.com/NVIDIA/BigVGAN
    """

    def __init__(self, channels, kernel_size, dilation: list):
        super(ResBlock, self).__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dil in dilation:
            self.convs1.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=dil, padding=get_padding(kernel_size, dil))
            ))
            self.convs2.append(weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
            ))

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.acts = nn.ModuleList([
            Activation1d(activation=Snake1d(channels))
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.acts[::2], self.acts[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


def get_padding(ks, dil):
    return int((ks * dil - dil) / 2)
