"""``core.grads`` must be a bit-identical port of the ``utils`` equivalents."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from similarity_sarah import utils
from similarity_sarah.core import grads

_DEVICE = torch.device("cpu")


def _setup() -> tuple[nn.Module, TensorDataset]:
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    return model, dataset


def test_full_gradient_matches_utils():
    model, dataset = _setup()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    g1 = grads.compute_full_gradient(model, loader, loss_fn, _DEVICE)
    g2 = utils.compute_full_gradient(model, loader, loss_fn, _DEVICE)
    for a, b in zip(g1, g2):
        assert torch.equal(a, b)


def test_batch_gradient_matches_utils_with_shared_xy():
    model, dataset = _setup()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    xy = next(iter(loader))
    g1 = grads.compute_batch_gradient(model, loader, loss_fn, _DEVICE, xy=xy)
    g2 = utils.compute_batch_gradient(model, loader, loss_fn, _DEVICE, xy=xy)
    for a, b in zip(g1, g2):
        assert torch.equal(a, b)
