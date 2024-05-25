from .ae import *
from .unet import *
from .vae import *


def get_model(model_name: str):
    if "VAE" in model_name:
        return VAE
    if "AE" in model_name:
        return AE
    raise NotImplementedError()
