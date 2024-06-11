from pathlib import Path

from torchvision.datasets import CIFAR10, MNIST, SVHN, CelebA, FashionMNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_dataset(dataset: str, datasets_dir: Path, image_size: int):
    match dataset:
        case "CelebA":
            transforms = Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        case _:
            transforms = Compose([
                Resize((image_size, image_size)),
                ToTensor()
            ])
    match dataset:
        case "MNIST":
            train_dataset = MNIST(
                root=datasets_dir,
                train=True,
                download=True,
                transform=transforms
            )
            test_dataset = MNIST(
                root=datasets_dir,
                train=False,
                download=True,
                transform=transforms
            )
            return train_dataset, test_dataset
        case "FashionMNIST":
            train_dataset = FashionMNIST(
                root=datasets_dir,
                train=True,
                download=True,
                transform=transforms
            )
            test_dataset = FashionMNIST(
                root=datasets_dir,
                train=False,
                download=True,
                transform=transforms
            )
            return train_dataset, test_dataset
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
