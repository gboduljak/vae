from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

from .unet import ConvBlock, UNet


class BetaVAE(UNet):
    def __init__(
        self,
        beta: float = 1.0,
        image_channels: int = 3,
        num_channels: int = 64,
        num_groups: int = 32,
        image_size: Tuple[int, int] = (32, 32),
        channel_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
        latent_channels: int = 3,
        dropout: int = 0.1,
        norm: Literal['BatchNorm', 'GroupNorm', 'SpectralNorm'] = 'GroupNorm',
        **kwargs
    ):
        super(BetaVAE, self).__init__(
            image_channels,
            num_channels,
            num_groups,
            channel_mults,
            dropout,
            norm
        )
        [H, W] = image_size

        self.beta = beta
        self.max_channels = num_channels * np.prod(channel_mults)
        self.min_H, self.min_W = (
            H // np.power(2, len(channel_mults) - 1),
            W // np.power(2, len(channel_mults) - 1)
        )

        self.latent_dim = latent_channels * self.min_H * self.min_W
        self.latent_channels = latent_channels

        self.to_latent = ConvBlock(
            in_channels=self.max_channels,
            out_channels=2 * latent_channels,
            num_groups=1,
            dropout=dropout,
            norm=norm,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.from_latent = ConvBlock(
            in_channels=latent_channels,
            out_channels=self.max_channels,
            num_groups=1,
            dropout=dropout,
            norm=norm,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.encode(x)
        z = q.rsample()
        p = self.decode(z)

        approx_log_likelihood = -((x - p.loc)**2).sum(dim=[1, 2, 3])
        kl = kl_divergence(
            q,
            Normal(
                loc=torch.zeros_like(q.loc, device=q.loc.device),
                scale=torch.ones_like(q.scale, device=q.scale.device)
            )
        ).sum(dim=1)
        elbo = approx_log_likelihood - self.beta * kl

        return (
            p.loc,
            {
                "loss": -elbo,
                "elbo": elbo,
                "approx_log_likelihood": approx_log_likelihood,
                "kl": kl
            }
        )

    def encode(self, x) -> Normal:
        batch_size, *_ = x.shape
        x = self.in_proj(x)
        x = self.down(x)
        x = self.left_middle(x)  # [B, C, H, W]
        x = self.to_latent(x)  # [B, 2D, H, W]

        [mu, logvar] = x.chunk(2, dim=1)  # 2 x [B, D, H, W]
        mu = mu.view((batch_size, self.latent_dim))
        logvar = logvar.view((batch_size, self.latent_dim))

        std = torch.exp(0.5 * logvar)

        return Normal(mu, std)

    def decode(self, z) -> Normal:
        x = z.view(
            (-1, self.latent_channels, self.min_H, self.min_W)
        )
        x = self.from_latent(x)
        x = self.right_middle(x)
        x = self.up(x)
        x = self.out_proj(x)
        x = F.tanh(x)
        return Normal(x, torch.ones_like(x, device=x.device))

    def sample(self, num_samples: int, device) -> torch.Tensor:
        prior = Normal(
            loc=torch.zeros((num_samples, self.latent_dim), device=device),
            scale=torch.ones((num_samples, self.latent_dim), device=device),
        )
        z = prior.sample()
        p = self.decode(z)
        return p.loc
