from pathlib import Path

import torch

import wandb


def save(model, optimizer, epoch, ckpt_dir: Path, ckpt_name: str):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / f"{ckpt_name}.ckpt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, ckpt_path)
    wandb.save(ckpt_path)
