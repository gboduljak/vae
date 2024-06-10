
from io import BytesIO
from math import ceil
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from scipy.stats import multivariate_normal, norm
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from utils import to_numpy, unnorm


def reconstruct(vae, num_recons: int, num_per_row: int, num_channels: int, image_size: int, dataset: Dataset, device):
    vae.eval()
    vae = vae.to(device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=num_recons,
        shuffle=False
    )
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


def plot_latent_space_distribution(
    model,
    dataloader: DataLoader,
    latent_dim: int,
    cols: int = 8,
    device=torch.device("cpu")
) -> Image:

    model = model.to(device)
    model.eval()

    z = []
    with torch.inference_mode():
        for (x, _) in iter(dataloader):
            x = x.to(device)
            z.append(model.encode(x).loc)
        z = torch.cat(z, dim=0)
        z = z.detach().cpu().numpy()

    sns.set_palette("pastel")
    sns.set_style("whitegrid")
    sns.set_context("paper")

    rows = ceil(latent_dim / cols)

    x = np.linspace(-3, 3, 128)
    norm_pdf = norm.pdf(x)

    fig = plt.figure(figsize=(30, 10))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(latent_dim):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(z[:, i], density=True, bins=20)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            f"latent_dim={i}",
            fontsize=10,
            ha="center",
            transform=ax.transAxes
        )
        ax.plot(x, norm_pdf)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def plot_latent_space_in_2d(
    embeddings: np.array,
    labels: np.array,
    hue: str
):
    sns.jointplot(
        x="z1",
        y="z2",
        hue=hue,
        palette="viridis",
        data=pd.DataFrame({
            "z1": embeddings[:, 0],
            "z2": embeddings[:, 1],
            hue: labels
        }),
        legend="full",
        alpha=0.6,
    )
    # Define the grid for the contour plot
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    x, y = np.meshgrid(x_range, y_range)
    # Define the mean and covariance matrix for the 2D Gaussian
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    # Contour plot unit Gaussian
    plt.contour(
        x,
        y,
        multivariate_normal(mean, cov).pdf(np.dstack((x, y))),
        levels=10,
        colors="black",
        alpha=0.5,
        linestyles="dashed",
    )
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)
