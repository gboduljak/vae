from pathlib import Path

from torchvision.datasets import CIFAR10, SVHN, CelebA
from torchvision.transforms import ToTensor, Resize, Compose


def get_dataset(dataset: str, datasets_dir: Path, image_size: int):
    transforms = Compose([
        Resize((image_size, image_size)),
        ToTensor()
    ])
    match dataset:
        case "SVHN":
            train_dataset = SVHN(
                root=datasets_dir / "SVHN",
                split="train",
                download=True,
                transform=transforms
            )
            test_dataset = SVHN(
                root=datasets_dir / "SVHN",
                split="test",
                download=True,
                transform=transforms
            )
            return train_dataset, test_dataset
        case "CIFAR10":
            train_dataset = CIFAR10(
                root=datasets_dir / "CIFAR10",
                train=True,
                download=True,
                transform=transforms
            )
            test_dataset = CIFAR10(
                root=datasets_dir / "CIFAR10",
                train=False,
                download=True,
                transform=transforms
            )
            return train_dataset, test_dataset
        case "CelebA":
            train_dataset = CelebA(
                root=datasets_dir / "CelebA",
                split="train",
                download=True,
                transform=transforms,
            )
            test_dataset = CelebA(
                root=datasets_dir / "CelebA",
                split="test",
                download=True,
                transform=transforms,
            )
            return train_dataset, test_dataset
