"""Dataset loading utilities.

Currently supports CIFAR-10.  To add a new dataset, create a new branch
inside :func:`load_dataset` (keyed by ``cfg.name``) and return
``(train, test)`` ``Dataset`` objects.

Notes on data augmentation
--------------------------
Strict SARAH-style algorithms assume *deterministic* per-sample gradients
(otherwise ∇f_i(w) is ill-defined).  However:

* In our distributed simulation the only place where stochasticity hurts
  is the *client gradient computation*.  Those still use the
  deterministic transform.
* On the *server* it is perfectly safe (and practically necessary on
  CIFAR-10 to reach high accuracy) to apply random crops + horizontal
  flips inside the inexact prox solver.

To enable the standard CIFAR-10 augmentation pipeline on the server
loader, set ``data.augment_server: true`` in the config.  Both loaders
otherwise use the deterministic transform.
"""

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import Dataset, TensorDataset, random_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _cifar10_eval_transform() -> T.Compose:
    """Deterministic transform (no augmentation) for exact gradient computation."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _cifar10_train_transform() -> T.Compose:
    """Standard CIFAR-10 augmentation: random crop + horizontal flip."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
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
        transform = _cifar10_eval_transform()
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


def load_augmented_train(cfg: DictConfig) -> Dataset | None:
    """If supported, return an augmented copy of the training set.

    ``None`` means "no augmentation available for this dataset"; the
    caller should fall back to the deterministic loader.
    """
    if cfg.name != "cifar10":
        return None
    return torchvision.datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=_cifar10_train_transform(),
    )


def split_train_val(
    dataset: Dataset,
    val_fraction: float,
) -> tuple[Dataset, Dataset]:
    """Split *dataset* into non-overlapping train and validation subsets."""
    n = len(dataset)  # type: ignore[arg-type]
    n_val = int(n * val_fraction)
    n_train = n - n_val
    return random_split(dataset, [n_train, n_val])
