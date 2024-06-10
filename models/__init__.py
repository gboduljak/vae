from .ae import *
from .beta_vae import *
from .unet import *
from .vae import *


def get_model(model_name: str):
    if "BetaVAE" in model_name:
        return BetaVAE
    if "VAE" in model_name:
        return VAE
    if "AE" in model_name:
        return AE
    raise NotImplementedError()
