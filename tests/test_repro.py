"""Tests for the seeding / determinism primitives in ``core.repro``."""

import random

import numpy as np
import torch

from similarity_sarah.core.repro import make_generator, set_seed


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_seeds_python_and_numpy():
    set_seed(7)
    r1, n1 = random.random(), float(np.random.rand())
    set_seed(7)
    r2, n2 = random.random(), float(np.random.rand())
    assert r1 == r2
    assert n1 == n2


def test_deterministic_flag_enables_torch_deterministic():
    set_seed(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()


def test_make_generator_is_independent_of_global_rng():
    g1 = make_generator(99)
    a = torch.randn(4, generator=g1)
    _ = torch.randn(10)  # perturb global RNG; must not affect g2
    g2 = make_generator(99)
    b = torch.randn(4, generator=g2)
    assert torch.equal(a, b)
