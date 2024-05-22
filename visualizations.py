
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import to_numpy, unnorm


def reconstruct(vae, num_recons: int, num_per_row: int, num_channels: int, image_size: int, dataloader: DataLoader, device):
    vae.eval()
    vae = vae.to(device)

    x, _ = next(iter(dataloader))
    x = x.to(device)
    x = x[:num_recons, :, :, :]

    with torch.inference_mode():
        x_hat, *_ = vae(x)
        x_hat = unnorm(x_hat)
        x = unnorm(x)

    x_recons = torch.zeros(
        (2 * num_recons, num_channels, image_size, image_size),
        dtype=x.dtype,
        device=x.device
    )
    x_recons[0::2] = x
    x_recons[1::2] = x_hat

    sample_grid = make_grid(x_recons, nrow=num_per_row)
    sample_grid = to_numpy(sample_grid)
    return Image.fromarray(sample_grid)


def sample(vae, num_samples, device):
    vae.eval()
    vae = vae.to(device)

    with torch.inference_mode():
        x = vae.sample(num_samples, device=device)
        x = unnorm(x)

    sample_grid = make_grid(x, nrow=num_samples // 10)
    sample_grid = to_numpy(sample_grid)
    return Image.fromarray(sample_grid)


def interpolate(vae, num_interpolations, latent_dim, num_steps, test_dataset, device):
    vae.eval()
    vae = vae.to(device)
    # Obtain x
    dataloader = DataLoader(
        test_dataset,
        batch_size=2*num_interpolations,
        shuffle=True
    )
    x, _ = next(iter(dataloader))
    x = x.to(device)
    # Obtain latents
    with torch.inference_mode():
        q = vae.encode(x)
        z = q.rsample()
    # Obtain endpoints
    x_left, x_right = x.chunk(2, dim=0)
    z_left, z_right = z.chunk(2, dim=0)
    # Obtain interpolation endpoints
    alpha = torch.linspace(
        0, 1,
        num_steps,
        device=z.device
    ).view(-1, 1)
    alpha = alpha.repeat(1, latent_dim)
    alpha = alpha.view((num_steps, latent_dim))

    x_interps: List[np.array] = []
    for i in range(num_interpolations):
        z_interp = (1 - alpha) * z_left[i, :] + alpha * z_right[i, :]
        x_interp = vae.decode(z_interp).loc
        x_interp = unnorm(x_interp)
        x_interps.append(
            torch.cat(
                [
                    unnorm(x_left[i, ...].unsqueeze(0)),
                    x_interp,
                    unnorm(x_right[i, ...].unsqueeze(0))
                ],
                dim=0
            )
        )  # [B, C, H, W]

    x_interps = torch.cat(x_interps, dim=0)  # [B, C, H, W]
    interp_grid = make_grid(x_interps, nrow=num_steps + 2)
    interp_grid = to_numpy(interp_grid)  # [B, H, W, C]
    return Image.fromarray(interp_grid)
