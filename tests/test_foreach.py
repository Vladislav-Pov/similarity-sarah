"""``core.foreach`` must be bit-identical to the per-tensor loops it replaces."""

import torch

from similarity_sarah.core import foreach


def _rand_list() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(3, 4), torch.randn(5), torch.randn(2, 2, 2)]


def test_add_matches_per_tensor():
    a = _rand_list()
    b = [t + 1.0 for t in _rand_list()]
    ref = [t.clone() for t in a]

    foreach.add_(a, b, alpha=0.37)
    for r, s in zip(ref, b):
        r.add_(s, alpha=0.37)

    for got, want in zip(a, ref):
        assert torch.equal(got, want)


def test_scale_matches_per_tensor():
    a = _rand_list()
    ref = [t.clone() for t in a]

    foreach.scale_(a, 0.123)
    for r in ref:
        r.mul_(0.123)

    for got, want in zip(a, ref):
        assert torch.equal(got, want)


def test_sub_matches_per_tensor():
    a = _rand_list()
    b = [t + 0.5 for t in _rand_list()]

    out = foreach.sub(a, b)
    for o, x, y in zip(out, a, b):
        assert torch.equal(o, x - y)


def test_empty_lists_are_noops():
    foreach.add_([], [])
    foreach.scale_([], 2.0)
    assert foreach.sub([], []) == []
