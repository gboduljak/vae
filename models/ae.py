from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from distributions import PointMass

from .unet import UNet


class AE(UNet):
    def __init__(
        self,
        image_channels: int = 3,
        num_channels: int = 64,
        num_groups: int = 32,
        image_size: Tuple[int, int] = (32, 32),
        channel_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
        latent_dim: int = 32,
        dropout: int = 0.1,
        norm: Literal['BatchNorm', 'GroupNorm', 'SpectralNorm'] = 'GroupNorm',
        **kwargs
    ):
        super(AE, self).__init__(
            image_channels,
            num_channels,
            num_groups,
            channel_mults,
            dropout,
            norm
        )
        [H, W] = image_size
        self.latent_dim = latent_dim

        self.max_channels = num_channels * np.prod(channel_mults)
        self.min_H, self.min_W = (
            H // np.power(2, len(channel_mults) - 1),
            W // np.power(2, len(channel_mults) - 1)
        )
        self.to_latent = nn.Linear(
            self.max_channels,
            latent_dim
        )
        self.from_latent = nn.Linear(
            latent_dim,
            self.max_channels * self.min_H * self.min_W
        )

    def forward(self, x) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.encode(x)
        z = q.loc
        p = self.decode(z)
        loss = ((x - p.loc)**2).sum(dim=[1, 2, 3])

        return (p.loc, {"loss": loss})

    def encode(self, x) -> PointMass:
        x = self.in_proj(x)
        x = self.down(x)
        x = self.left_middle(x)  # [B, C, H, W]
        x = torch.mean(x, dim=[2, 3])  # [B, C] # GAP
        x = self.to_latent(x)  # [B, latent_dim]
        return PointMass(loc=x)

    def decode(self, z) -> PointMass:
        x = self.from_latent(z).view(
            (-1, self.max_channels, self.min_H, self.min_W)
        )
        x = self.right_middle(x)
        x = self.up(x)
        x = self.out_proj(x)
        x = F.sigmoid(x)
        return PointMass(loc=x)

    def sample(self, num_samples: int, device) -> torch.Tensor:
        prior = Normal(
            loc=torch.zeros((num_samples, self.latent_dim), device=device),
            scale=torch.ones((num_samples, self.latent_dim), device=device),
        )
        z = prior.sample()
        p = self.decode(z)
        x = p.loc
        return x
