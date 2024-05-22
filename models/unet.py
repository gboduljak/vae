from typing import List, Tuple, Union

import torch.nn as nn


class ResidualBlock(nn.Sequential):

    def __init__(self,
                 channels: int,
                 num_groups: int,
                 dropout: float
                 ):
        super(ResidualBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.silu2 = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        y = x
        y = self.norm1(y)
        y = self.silu1(y)
        y = self.conv1(y)
        y = self.norm2(y)
        y = self.silu2(y)
        y = self.dropout(y)
        y = self.conv2(y)
        return y + x


class Downsample(nn.Conv2d):
    def __init__(self, channels):
        super(Downsample, self).__init__(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1
        )


class Upsample(nn.Sequential):
    def __init__(self, channels):
        super(Upsample, self).__init__(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(
                channels, channels,
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
        dropout: int = 0.1,
    ):
        super(UNet, self).__init__()

        self.in_proj = nn.Conv2d(
            image_channels,
            num_channels,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.out_proj = nn.Conv2d(
            num_channels,
            image_channels,
            kernel_size=(3, 3),
            padding=(1, 1)
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
                nn.Sequential(
                    nn.GroupNorm(num_groups, in_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(3, 3),
                        padding=(1, 1)
                    )
                )
            )
            in_channels = out_channels
            if i < num_resolutions - 1:
                down.append(Downsample(in_channels))

        in_channels = out_channels

        left_middle = [
            ResidualBlock(out_channels, num_groups, dropout),
            ResidualBlock(out_channels, num_groups,  dropout)
        ]
        right_middle = [
            ResidualBlock(out_channels, num_groups,  dropout),
            ResidualBlock(out_channels, num_groups,  dropout)
        ]

        for i in reversed(range(num_resolutions)):
            out_channels = in_channels // channel_mults[i]
            up.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups, in_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(3, 3),
                        padding=(1, 1)
                    )
                )
            )
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

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
