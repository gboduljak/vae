import torch
from typing import Dict, Any

def get_amp_utils(config: Dict[str, Any]):

    dtypes = {
       "float32": torch.float32,
       "bfloat16": torch.bfloat16
    }

    if config["training"]["amp"] and config["training"]["dtype"] != "float32":
      if config["training"]["device"] == "cuda":
          from torch.cuda.amp import autocast, GradScaler
          return autocast, GradScaler(), dtypes[config["training"]["dtype"]]
        
    
    return None, None, None