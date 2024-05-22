import torch


def unnorm(x):
    x = x * 255
    x = torch.floor(x)
    x = x.to(torch.uint8)
    return x


def to_numpy(x):
    return (
        x.permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
