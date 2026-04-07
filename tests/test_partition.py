"""Tests for data partitioning."""

import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from similarity_sarah.data.partition import create_partition


def _make_dataset(n: int = 100) -> TensorDataset:
    return TensorDataset(torch.randn(n, 3), torch.randint(0, 10, (n,)))


def test_uniform_partition_count():
    ds = _make_dataset(100)
    cfg = OmegaConf.create({"name": "uniform"})
    parts = create_partition(ds, 5, cfg)
    assert len(parts) == 5


def test_uniform_partition_sizes_sum():
    ds = _make_dataset(100)
    cfg = OmegaConf.create({"name": "uniform"})
    parts = create_partition(ds, 5, cfg)
    total = sum(len(p) for p in parts)
    assert total == 100


def test_uniform_partition_no_overlap():
    ds = _make_dataset(100)
    cfg = OmegaConf.create({"name": "uniform"})
    parts = create_partition(ds, 5, cfg)
    all_indices: list[int] = []
    for p in parts:
        all_indices.extend(p.indices)
    assert len(set(all_indices)) == 100


def test_uneven_split():
    ds = _make_dataset(103)
    cfg = OmegaConf.create({"name": "uniform"})
    parts = create_partition(ds, 5, cfg)
    sizes = sorted(len(p) for p in parts)
    # 103 = 5*20 + 3 → three partitions get 21, two get 20
    assert sizes == [20, 20, 21, 21, 21]
