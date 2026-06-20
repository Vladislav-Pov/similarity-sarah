"""Loader helpers — index alignment for train augmentation."""

import torch
from torch.utils.data import Subset, TensorDataset

from similarity_sarah.data.loaders import _abs_indices


def test_abs_indices_identity_on_base():
    base = TensorDataset(torch.arange(10))
    assert _abs_indices(base) == list(range(10))


def test_abs_indices_resolves_nested_subset():
    # A node partition is Subset(Subset(full_train)); the augmented copy must
    # reference the SAME absolute indices into the base dataset.
    base = TensorDataset(torch.arange(10))
    sub1 = Subset(base, [9, 8, 7, 6, 5, 4])      # positions into base
    sub2 = Subset(sub1, [0, 2, 4])               # positions into sub1 → base 9,7,5
    assert _abs_indices(sub2) == [9, 7, 5]
