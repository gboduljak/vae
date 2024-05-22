from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from tqdm import tqdm

import wandb
from models import VAE
from save import save
from visualizations import interpolate, reconstruct, sample


def train_model(
    model: VAE,
    model_name: str,
    train_dataset,
    test_dataset,
    ckpt_dir: Path,
    num_epochs: int,
    batch_size: int,
    lr: float = 3e-4,
    image_size: int = 28,
    num_channels: int = 1,
    num_samples: int = 100,
    num_recons: int = 100,
    num_interps: int = 10,
    eval_steps: int = 100,
    device: torch.device = torch.device('cpu')
) -> VAE:

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    for epoch in range(num_epochs):
        model.train()

        running_losses = []

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for (batch_idx, batch) in enumerate(tepoch):
                x, y = batch

                optimizer.zero_grad()
                x = x.to(device)
                _, metrics = model(x)
                loss = metrics["loss"].mean()
                loss.backward()
                optimizer.step()

                running_losses.append(loss.item())
                tepoch.set_postfix(loss=np.mean(running_losses))

                if batch_idx % eval_steps == 0:
                    samples = sample(model, num_samples, device)
                    reconstructions = reconstruct(
                        model,
                        num_recons,
                        10,
                        num_channels,
                        image_size,
                        test_dataloader,
                        device
                    )
                    interpolations = interpolate(
                        vae,
                        num_interps,
                        latent_dim,
                        10,
                        test_dataset,
                        device
                    )
                    metrics_to_log = {
                        f"train_{metric}": value.mean().item()
                        for (metric, value) in metrics.items()
                    }
                    imgs_to_log = {
                        "samples": wandb.Image(samples),
                        "reconstructions": wandb.Image(reconstructions),
                        "interpolations": wandb.Image(interpolations)
                    }
                    wandb.log({
                        **metrics_to_log,
                        **imgs_to_log
                    })

            model.eval()
            running_metrics: Dict[str, float] = {}
            for batch in iter(test_dataloader):
                with torch.inference_mode():
                    x = x.to(device)
                    _, metrics = model(x)
                    for metric, value in metrics.items():
                        if metric not in running_metrics:
                            running_metrics[metric] = []
                        running_metrics[metric].append(value)

            metrics_to_log = {
                f"test_{metric}": torch.cat(value, dim=0).mean().item()
                for metric, value in running_metrics.items()
            }
            samples = sample(model, num_samples, device)
            reconstructions = reconstruct(
                model,
                num_recons,
                10,
                num_channels,
                image_size,
                test_dataloader,
                device
            )
            interpolations = interpolate(
                vae,
                num_interps,
                latent_dim,
                10,
                test_dataset,
                device
            )
            imgs_to_log = {
                "samples": wandb.Image(samples),
                "reconstructions": wandb.Image(reconstructions),
                "interpolations": wandb.Image(interpolations)
            }
            wandb.log({
                **metrics_to_log,
                **imgs_to_log
            })
            save(
                model,
                optimizer,
                epoch,
                ckpt_dir,
                f"{model_name}_epoch={epoch}"
            )


if __name__ == "__main__":

    num_epochs = 20
    num_channels = 32
    channel_mults = (1, 2, 2)
    num_groups = 16
    latent_dim = 16

    batch_size = 128
    lr = 3e-4
    device = torch.device('mps')
    wandb.init()

    scale = transforms.ToTensor()

    train_dataset = SVHN(
        root='datasets/SVHN',
        split='train',
        download=True,
        transform=scale
    )
    test_dataset = SVHN(
        root='datasets/SVHN',
        split='test',
        download=True,
        transform=scale
    )

    vae = VAE(
        image_channels=3,
        image_size=(32, 32),
        channel_mults=channel_mults,
        num_channels=num_channels,
        num_groups=num_groups,
        latent_dim=latent_dim
    )

    train_model(
        vae,
        "VAE-SVHN",
        train_dataset,
        test_dataset,
        "checkpoints",
        num_epochs,
        batch_size,
        lr,
        image_size=32,
        num_channels=3,
        device=device,
        num_recons=50
    )

    wandb.init(
        project="vae-svhn",
        config={
            "dataset": "SVHN",
            "epochs": num_epochs,
            "num_channels": num_channels,
            "channel_mults": channel_mults,
            "num_groups": num_groups,
            "latent_dim": latent_dim
        }
    )
