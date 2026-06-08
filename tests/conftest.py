"""Shared pytest fixtures and test hygiene."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_determinism():
    """Undo the process-wide deterministic switch after each test.

    ``core.repro.set_seed(deterministic=True)`` flips a global PyTorch flag;
    reset it on teardown so it cannot leak into unrelated tests.
    """
    yield
    torch.use_deterministic_algorithms(False)
