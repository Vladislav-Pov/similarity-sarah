"""Unit tests for the core parameter-list utilities."""

import torch
import torch.nn as nn

from similarity_sarah.core import params


def _model() -> nn.Module:
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_get_set_roundtrip():
    m = _model()
    snapshot = params.get_params(m)
    for q in m.parameters():
        nn.init.zeros_(q)
    params.set_params(m, snapshot)
    for orig, q in zip(snapshot, m.parameters()):
        assert torch.equal(orig, q.data)


def test_norms_against_analytic():
    p = [torch.tensor([3.0, 4.0]), torch.tensor([[12.0]])]  # 3,4,12 -> norm 13
    assert params.compute_param_norm_sq(p) == 169.0
    assert params.compute_param_norm(p) == 13.0


def test_diff_norm():
    a = [torch.tensor([1.0, 2.0])]
    b = [torch.tensor([1.0, 5.0])]
    assert params.diff_param_norm(a, b) == 3.0


def test_add_and_scale():
    a = [torch.ones(3)]
    params.add_params_(a, [torch.full((3,), 2.0)], alpha=0.5)  # 1 + 0.5*2 = 2
    assert torch.equal(a[0], torch.full((3,), 2.0))
    params.scale_params_(a, 0.5)  # 2 * 0.5 = 1
    assert torch.equal(a[0], torch.ones(3))


def test_zeros_like_params():
    m = _model()
    z = params.zeros_like_params(m)
    assert len(z) == len(list(m.parameters()))
    assert all(t.abs().sum().item() == 0.0 for t in z)
    assert params.zeros_like_params(z)[0].shape == z[0].shape
