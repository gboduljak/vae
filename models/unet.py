from typing import List, Literal, Tuple, Union

import torch.nn as nn
from torch.nn.utils import spectral_norm


def get_norm(norm: str, channels: int, num_groups: int):
    match norm:
        case 'BatchNorm':
            return nn.BatchNorm2d(channels)
        case 'GroupNorm':
            return nn.GroupNorm(num_groups, channels)
        case 'SpectralNorm':
            return nn.Identity()
        case _:
            raise NotImplementedError()


class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int,
                 dropout: float,
                 norm: Literal['BatchNorm', 'GroupNorm', 'SpectralNorm'],
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0
                 ):
        super(ConvBlock, self).__init__(
            get_norm(norm, in_channels, num_groups),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout, inplace=True) if dropout else nn.Identity(),
            (
                spectral_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding
                    )
                )
                if norm == 'SpectralNorm' else
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding
                )
            ),
        )


class ResidualBlock(nn.Sequential):
    def __init__(self,
                 channels: int,
                 num_groups: int,
                 dropout: float,
                 norm: Literal['BatchNorm', 'GroupNorm', 'SpectralNorm']
                 ):
        super(ResidualBlock, self).__init__(
            ConvBlock(channels, channels, num_groups, dropout, norm, 3, 1, 1),
            ConvBlock(channels, channels, num_groups, dropout, norm, 3, 1, 1)
        )

    def forward(self, x):
        return (
            x + super(ResidualBlock, self).forward(x)
        )


class Downsample(ConvBlock):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        dropout: float,
        norm: Literal['BatchNorm', 'GroupNorm', 'SpectralNorm']
    ):
        super(Downsample, self).__init__(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            dropout=dropout,
            norm=norm,
            kernel_size=3,
            stride=2,
            padding=1
        )


class Upsample(nn.Sequential):
    def __init__(self,
                 channels,
                 num_groups: int,
                 dropout: float,
                 norm: Literal['BatchNorm', 'GroupNorm', 'SpectralNorm']
                 ):
        super(Upsample, self).__init__(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(
                channels,
                channels,
                num_groups,
                dropout,
                norm,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        num_channels: int = 64,
        num_groups: int = 32,
        channel_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
        dropout: float = 0,
        norm:  Literal['BatchNorm', 'GroupNorm', 'SpectralNorm'] = 'GroupNorm'
    ):
        super(UNet, self).__init__()

        self.in_proj = nn.Conv2d(
            image_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.out_proj = nn.Conv2d(
            num_channels,
            image_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        num_resolutions = len(channel_mults)

        down: List[nn.Module] = []
        left_middle: List[nn.Module] = []
        right_middle: List[nn.Module] = []
        up: List[nn.Module] = []

        out_channels = num_channels
        in_channels = num_channels

        for i in range(num_resolutions):
            out_channels = in_channels * channel_mults[i]
            down.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_groups=num_groups,
                    dropout=dropout,
                    norm=norm,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            in_channels = out_channels
            if i < num_resolutions - 1:
                down.append(
                    Downsample(
                        channels=in_channels,
                        num_groups=num_groups,
                        dropout=dropout,
                        norm=norm
                    )
                )

        in_channels = out_channels
        left_middle = [
            ResidualBlock(
                channels=out_channels,
                num_groups=num_groups,
                dropout=dropout,
                norm=norm
            ),
            ResidualBlock(
                channels=out_channels,
                num_groups=num_groups,
                dropout=dropout,
                norm=norm
            ),
        ]
        right_middle = [
            ResidualBlock(
                channels=out_channels,
                num_groups=num_groups,
                dropout=dropout,
                norm=norm
            ),
            ResidualBlock(
                channels=out_channels,
                num_groups=num_groups,
                dropout=dropout,
                norm=norm
            ),
        ]

        for i in reversed(range(num_resolutions)):
            out_channels = in_channels // channel_mults[i]
            up.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_groups=num_groups,
                    dropout=dropout,
                    norm=norm,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            in_channels = out_channels
            if i > 0:
                up.append(
                    Upsample(
                        channels=in_channels,
                        num_groups=num_groups,
                        dropout=dropout,
                        norm=norm
                    )
                )

        self.down = nn.Sequential(*down)
        self.left_middle = nn.Sequential(*left_middle)
        self.right_middle = nn.Sequential(*right_middle)
        self.up = nn.Sequential(*up)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.down(x)
        x = self.left_middle(x)
        x = self.right_middle(x)
        x = self.up(x)
        x = self.out_proj(x)
        return x
