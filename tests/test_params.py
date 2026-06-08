"""``core.params`` must be a bit-identical port of the ``utils`` equivalents."""

import torch
import torch.nn as nn

from similarity_sarah import utils
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


def test_norms_match_utils():
    p = params.get_params(_model())
    assert params.compute_param_norm_sq(p) == utils.compute_param_norm_sq(p)
    assert params.compute_param_norm(p) == utils.compute_param_norm(p)


def test_diff_norm_matches_utils():
    a = params.get_params(_model())
    b = [t + 0.3 for t in a]
    assert params.diff_param_norm(a, b) == utils.diff_param_norm(a, b)


def test_add_and_scale_match_utils():
    a1 = params.get_params(_model())
    a2 = params.clone_params(a1)
    src = [t * 2 for t in a1]

    params.add_params_(a1, src, alpha=0.5)
    utils.add_params_(a2, src, alpha=0.5)
    for x, y in zip(a1, a2):
        assert torch.equal(x, y)

    params.scale_params_(a1, 0.7)
    utils.scale_params_(a2, 0.7)
    for x, y in zip(a1, a2):
        assert torch.equal(x, y)


def test_zeros_like_params():
    m = _model()
    z = params.zeros_like_params(m)
    assert len(z) == len(list(m.parameters()))
    assert all(t.abs().sum().item() == 0.0 for t in z)
