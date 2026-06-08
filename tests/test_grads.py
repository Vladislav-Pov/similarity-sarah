"""Unit tests for the core gradient utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from similarity_sarah.core import grads

_DEVICE = torch.device("cpu")


def _setup() -> tuple[nn.Module, TensorDataset]:
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    return model, dataset


def test_full_gradient_independent_of_batching():
    model, dataset = _setup()
    loss_fn = nn.CrossEntropyLoss()
    g_full = grads.compute_full_gradient(
        model, DataLoader(dataset, batch_size=8, shuffle=False), loss_fn, _DEVICE
    )
    g_split = grads.compute_full_gradient(
        model, DataLoader(dataset, batch_size=4, shuffle=False), loss_fn, _DEVICE
    )
    for a, b in zip(g_full, g_split):
        assert torch.allclose(a, b, atol=1e-6)


def test_batch_gradient_with_xy_matches_first_batch():
    model, dataset = _setup()
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    xy = next(iter(loader))
    g_xy = grads.compute_batch_gradient(model, loader, loss_fn, _DEVICE, xy=xy)
    g_drawn = grads.compute_batch_gradient(model, loader, loss_fn, _DEVICE)
    for a, b in zip(g_xy, g_drawn):
        assert torch.equal(a, b)


def test_batch_equals_full_for_single_minibatch():
    model, dataset = _setup()
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)  # one batch covers all
    g_batch = grads.compute_batch_gradient(model, loader, loss_fn, _DEVICE)
    g_full = grads.compute_full_gradient(model, loader, loss_fn, _DEVICE)
    for a, b in zip(g_batch, g_full):
        assert torch.allclose(a, b, atol=1e-6)
