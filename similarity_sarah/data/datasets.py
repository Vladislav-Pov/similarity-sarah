"""Dataset loading utilities.

Currently supports CIFAR-10.  To add a new dataset, create a new branch
inside ``load_dataset`` (keyed by ``cfg.name``) and return ``(train, test)``
dataset objects.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, TensorDataset, random_split


def _cifar10_transform() -> T.Compose:
    """Deterministic transform (no augmentation) for exact gradient computation."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def _synthetic_cifar10(n_train: int = 500, n_test: int = 100) -> tuple[Dataset, Dataset]:
    """Random tensors shaped like CIFAR-10 — useful for offline smoke tests."""
    train = TensorDataset(
        torch.randn(n_train, 3, 32, 32),
        torch.randint(0, 10, (n_train,)),
    )
    test = TensorDataset(
        torch.randn(n_test, 3, 32, 32),
        torch.randint(0, 10, (n_test,)),
    )
    return train, test


def load_dataset(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    """Load train and test datasets according to *cfg*.

    Returns:
        ``(train_dataset, test_dataset)``
    """
    if cfg.name == "cifar10":
        transform = _cifar10_transform()
        train = torchvision.datasets.CIFAR10(
            root=cfg.data_dir, train=True, download=True, transform=transform,
        )
        test = torchvision.datasets.CIFAR10(
            root=cfg.data_dir, train=False, download=True, transform=transform,
        )
        return train, test

    if cfg.name == "synthetic":
        return _synthetic_cifar10(
            n_train=cfg.get("n_train", 500),
            n_test=cfg.get("n_test", 100),
        )

    raise ValueError(f"Unknown dataset: {cfg.name}")


def split_train_val(
    dataset: Dataset,
    val_fraction: float,
) -> tuple[Dataset, Dataset]:
    """Split *dataset* into non-overlapping train and validation subsets."""
    n = len(dataset)  # type: ignore[arg-type]
    n_val = int(n * val_fraction)
    n_train = n - n_val
    return random_split(dataset, [n_train, n_val])
