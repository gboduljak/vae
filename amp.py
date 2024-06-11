from typing import Any, Dict

import torch
from torch import autocast
from torch.amp import GradScaler


def get_amp_utils(config: Dict[str, Any]):

    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    grad_scalers = {
        torch.float16: GradScaler()
    }

    amp_enabled = config["training"]["amp"]
    device = config["training"]["device"]
    dtype = dtypes[config["training"]["dtype"]]

    grad_scaler = grad_scalers.get(dtype, None)

    if amp_enabled and dtype != torch.float32:
        return (
            lambda: autocast(device_type=device, dtype=dtype),
            grad_scaler,
        )

    return None, None
