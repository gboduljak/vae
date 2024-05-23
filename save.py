from pathlib import Path

import torch

import wandb


def save(model, optimizer, metadata, ckpt_dir: Path, ckpt_name: str, use_wandb: bool):
    ckpt_path = ckpt_dir / f"{ckpt_name}.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata,
    }, str(ckpt_path))
    if use_wandb:
        wandb.save(str(ckpt_path))
