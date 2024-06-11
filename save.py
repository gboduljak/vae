from pathlib import Path

import torch

import wandb


def save_locally(model, optimizer, metadata, ckpt_dir: Path, ckpt_name: str) -> str:
    ckpt_path = ckpt_dir / f"{ckpt_name}.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata,
    }, str(ckpt_path))

    return str(ckpt_path)


def save_to_wandb(ckpt_path: str, use_wandb: bool):
    if use_wandb:
        wandb.save(ckpt_path)
