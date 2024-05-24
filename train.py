from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import numpy as np
import torch
import yaml
from lpips import LPIPS
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

import wandb
from amp import get_amp_utils
from dataset import get_dataset
from models import VAE
from save import save
from schedulers import LinearWarmupScheduler
from seed import get_seeded_generator, seed_everything, seeded_worker_init_fn
from utils import suppress_external_logs
from visualizations import (interpolate, plot_latent_space_distribution,
                            reconstruct, sample)


def train_model(
    model: VAE,
    config: Dict[str, float],
    train_dataset,
    test_dataset,
    device: torch.device = torch.device("cpu"),
) -> VAE:

    with suppress_external_logs():
        lpips = LPIPS(net="vgg")
        lpips = lpips.to(device)

    use_wandb = config["wandb"]["enabled"]

    optimizer = AdamW(
        params=model.parameters(),
        lr=config["training"]["lr"]
    )

    if "warmup_steps_percentage" in config["training"]:
        total_steps = (
            len(train_dataset) // config["training"]["batch_size"]
        ) * config["training"]["num_epochs"]
        warmup_steps = int(
            total_steps * config["training"]["warmup_steps_percentage"]
        )

        scheduler = LinearWarmupScheduler(optimizer, warmup_steps, total_steps)
    else:
        scheduler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        worker_init_fn=seeded_worker_init_fn,
        generator=get_seeded_generator(config["training"]["seed"]),
        shuffle=True,
        pin_memory=config["training"]["pin_memory"],
        num_workers=config["training"]["num_workers"],
        prefetch_factor=(
            config["training"]["prefetch_factor"] if config["training"]["prefetch_factor"] else None
        ),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        worker_init_fn=seeded_worker_init_fn,
        generator=get_seeded_generator(config["training"]["seed"]),
        num_workers=config["training"]["num_workers"],
        prefetch_factor=(
            config["training"]["prefetch_factor"] if config["training"]["prefetch_factor"] else None
        ),
    )

    if use_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            config=config
        )

    model = model.to(device)

    best_epoch = -1
    best_epoch_lpips = pow(10, 9)

    autocast_factory, grad_scaler = get_amp_utils(config)

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        running_losses = []

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(
                f"epoch {epoch+1}/{config['training']['num_epochs']}"
            )
            for (batch_idx, batch) in enumerate(tepoch):
                x, y = batch

                optimizer.zero_grad()
                if autocast_factory:
                    with autocast_factory():
                        x = x.to(device)
                        _, metrics = model(x)
                        loss = metrics["loss"].mean()
                else:
                    x = x.to(device)
                    _, metrics = model(x)
                    loss = metrics["loss"].mean()

                if grad_scaler:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if scheduler:
                    scheduler.step()

                running_losses.append(loss.item())
                tepoch.set_postfix(loss=np.mean(running_losses))

                if batch_idx % config["training"]["eval_steps"] == 0:
                    samples = sample(
                        model,
                        config["training"]["num_samples"],
                        device
                    )
                    reconstructions = reconstruct(
                        model,
                        config["training"]["num_reconstructions"],
                        10,
                        config["image"]["channels"],
                        config["image"]["size"],
                        test_dataloader,
                        device
                    )
                    interpolations = interpolate(
                        model,
                        config["training"]["num_interpolations"],
                        config["model"]["latent_dim"],
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
                        "interpolations": wandb.Image(interpolations),
                    }

                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']

                    if use_wandb:
                        wandb.log({
                            **metrics_to_log,
                            **imgs_to_log,
                            **{
                                "lr": lr
                            }
                        })

            model.eval()
            running_metrics: Dict[str, List[np.array]] = {"lpips": []}

            for batch in iter(test_dataloader):
                with torch.inference_mode():
                    x = x.to(device)
                    x_hat, metrics = model(x)
                    for metric, value in metrics.items():
                        if metric not in running_metrics:
                            running_metrics[metric] = []
                        running_metrics[metric].append(value)
                    running_metrics["lpips"].append(
                        (
                            lpips(
                                in0=x,
                                in1=x_hat,
                                normalize=True
                            ).view((-1, ))
                            .detach()
                            .cpu()
                        )
                    )

            metrics_to_log = {
                f"test_{metric}": torch.cat(value, dim=0).mean().item()
                for metric, value in running_metrics.items()
            }
            current_epoch_lpips = metrics_to_log["test_lpips"]

            if current_epoch_lpips < best_epoch_lpips:
                best_epoch_lpips = current_epoch_lpips
                best_epoch = epoch
                save(
                    model,
                    optimizer,
                    {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_epoch_lpips": best_epoch_lpips,
                        **metrics_to_log
                    },
                    Path(config["training"]["checkpoints_dir"]),
                    f"{config['dataset']['name']}/{config['model']['name']}/best",
                    use_wandb
                )

            samples = sample(
                model,
                config["training"]["num_samples"],
                device
            )
            reconstructions = reconstruct(
                model,
                config["training"]["num_reconstructions"],
                10,
                config["image"]["channels"],
                config["image"]["size"],
                test_dataloader,
                device
            )
            interpolations = interpolate(
                model,
                config["training"]["num_interpolations"],
                config["model"]["latent_dim"],
                10,
                test_dataset,
                device
            )
            latent_space_distribution = plot_latent_space_distribution(
                model,
                test_dataloader,
                config["model"]["latent_dim"],
                device=device
            )
            imgs_to_log = {
                "samples": wandb.Image(samples),
                "reconstructions": wandb.Image(reconstructions),
                "interpolations": wandb.Image(interpolations),
                "latent_space_distribution": wandb.Image(latent_space_distribution)
            }

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if use_wandb:
                wandb.log({
                    **metrics_to_log,
                    **imgs_to_log,
                    **{
                        "lr": lr
                    }
                })

            save(
                model,
                optimizer,
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_epoch_lpips": best_epoch_lpips,
                    **metrics_to_log
                },
                Path(config["training"]["checkpoints_dir"]),
                f"{config['dataset']['name']}/{config['model']['name']}/latest",
                use_wandb
            )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as fs:
        config = yaml.safe_load(fs)

    device = torch.device(config["training"]["device"])
    seed_everything(config["training"]["seed"])

    train_dataset, test_dataset = get_dataset(
        config["dataset"]["name"],
        Path(config["dataset"]["datasets_dir"]),
        config["image"]["size"]
    )

    model = VAE(
        image_channels=config["image"]["channels"],
        image_size=(
            config["image"]["size"],
            config["image"]["size"]
        ),
        **config["model"]
    )

    print("config:")
    pprint(config)

    print("model:")
    summary(
        model,
        (
            config["image"]["channels"],
            config["image"]["size"],
            config["image"]["size"]
        ),
        batch_size=-1,
        device="cpu"
    )

    train_model(
        model,
        config,
        train_dataset,
        test_dataset,
        device,
    )
