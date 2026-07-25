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
from omegaconf import DictConfig
import torchvision
import torchvision.transforms as T
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

    if cfg.name == "glue":
        return _load_glue(cfg)

    raise ValueError(f"Unknown dataset: {cfg.name}")


# GLUE task → (text field(s), num_labels, validation split name).  Single-
# sentence tasks have a ``None`` second field.  STS-B is regression and is
# intentionally excluded (needs an MSE task, not ClassificationTask).
_GLUE_TASKS: dict[str, tuple[str, str | None, int, str]] = {
    "cola": ("sentence", None, 2, "validation"),
    "sst2": ("sentence", None, 2, "validation"),
    "mrpc": ("sentence1", "sentence2", 2, "validation"),
    "rte": ("sentence1", "sentence2", 2, "validation"),
    "qnli": ("question", "sentence", 2, "validation"),
    "qqp": ("question1", "question2", 2, "validation"),
    "mnli": ("premise", "hypothesis", 3, "validation_matched"),
}


def glue_num_labels(task: str) -> int:
    """Number of classification labels for a GLUE *task* (see ``_GLUE_TASKS``)."""
    key = str(task).lower()
    if key not in _GLUE_TASKS:
        raise ValueError(
            f"Unsupported GLUE task {key!r}; choose one of {sorted(_GLUE_TASKS)}",
        )
    return _GLUE_TASKS[key][2]


def _load_glue(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    """Load a GLUE task as ``(train, eval)`` TensorDatasets of packed tensors.

    Each sample ``x`` is an integer tensor of shape ``(2, max_length)`` with
    ``x[0] = input_ids`` and ``x[1] = attention_mask`` — the packing the
    :class:`RobertaLoRA` wrapper expects, so the vision-shaped ``(x, y)``
    pipeline (loaders, SARAH recursion, prox solvers) needs no changes.

    GLUE test splits on the Hub are unlabeled, so the returned "test" set is
    the task's validation split (standard practice for reporting GLUE dev
    numbers).  ``runner`` further carves a val split out of ``train`` via
    ``data.val_fraction``.
    """
    from datasets import load_dataset as hf_load_dataset
    from transformers import RobertaTokenizer

    task = str(cfg.task).lower()
    if task not in _GLUE_TASKS:
        raise ValueError(
            f"Unsupported GLUE task {task!r}; choose one of {sorted(_GLUE_TASKS)}",
        )
    field_a, field_b, _, eval_split = _GLUE_TASKS[task]
    max_length = int(cfg.get("max_length", 128))
    tokenizer_path = str(cfg.get("tokenizer_path", "roberta-base"))

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    # The canonical "glue" id is a legacy *script* dataset and breaks with
    # recent datasets/huggingface_hub (invalid ``hf://datasets/glue@...`` URI).
    # Load the namespaced parquet mirror instead (overridable via cfg.hf_path).
    hf_path = str(cfg.get("hf_path", "nyu-mll/glue"))
    raw = hf_load_dataset(hf_path, task, cache_dir=cfg.get("data_dir", None))

    def _pack(split_name: str) -> TensorDataset:
        split = raw[split_name]
        texts_a = split[field_a]
        texts_b = split[field_b] if field_b is not None else None
        enc = tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        # (N, 2, L): stack [input_ids, attention_mask] on a new middle axis.
        x = torch.stack([enc["input_ids"], enc["attention_mask"]], dim=1)
        y = torch.tensor(split["label"], dtype=torch.long)
        return TensorDataset(x, y)

    return _pack("train"), _pack(eval_split)


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
