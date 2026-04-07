"""Tests for parameter utilities and gradient computation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from similarity_sarah.utils import (
    add_params_,
    clone_params,
    compute_full_gradient,
    get_params,
    set_params,
    zeros_like_params,
)


def _make_linear() -> nn.Linear:
    m = nn.Linear(4, 2, bias=False)
    nn.init.ones_(m.weight)
    return m


def test_get_set_params_roundtrip():
    m = _make_linear()
    p = get_params(m)
    nn.init.zeros_(m.weight)
    set_params(m, p)
    assert torch.allclose(m.weight, torch.ones_like(m.weight))


def test_zeros_like_params():
    m = _make_linear()
    z = zeros_like_params(m)
    assert len(z) == 1
    assert z[0].shape == m.weight.shape
    assert z[0].abs().sum().item() == 0.0


def test_add_params_():
    m = _make_linear()
    a = get_params(m)
    b = clone_params(a)
    add_params_(a, b, alpha=2.0)
    assert torch.allclose(a[0], torch.full_like(a[0], 3.0))


def test_compute_full_gradient():
    torch.manual_seed(0)
    m = nn.Linear(4, 2, bias=True)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)

    loader_full = DataLoader(ds, batch_size=8, shuffle=False)
    loader_split = DataLoader(ds, batch_size=4, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    g_full = compute_full_gradient(m, loader_full, loss_fn, torch.device("cpu"))
    g_split = compute_full_gradient(m, loader_split, loss_fn, torch.device("cpu"))

    for gf, gs in zip(g_full, g_split):
        assert torch.allclose(gf, gs, atol=1e-6)
