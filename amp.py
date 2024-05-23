from typing import Any, Dict

import torch


def get_amp_utils(config: Dict[str, Any]):

    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }

    if config["training"]["amp"] and config["training"]["dtype"] != "float32":
        if config["training"]["device"] == "cuda":
            from torch.cuda.amp import GradScaler, autocast
            return autocast, GradScaler(), dtypes[config["training"]["dtype"]]
        if config["training"]["device"] == "mps":
            from torch import GradScaler, autocast
            return (
                lambda dtype: autocast(device_type="mps", dtype=dtype),
                GradScaler(),
                dtypes[config["training"]["dtype"]]
            )

    return None, None, None
