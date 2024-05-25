import torch
from torch.distributions.distribution import Distribution


class PointMass(Distribution):
    def __init__(self, loc: torch.Tensor):
        self.loc = loc

    def rsample(self, sample_shape: torch.Size = ...) -> torch.Tensor:
        return self.loc
