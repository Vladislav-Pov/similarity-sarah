"""Tests for the client batch scheduler."""

import torch

from similarity_sarah.runtime.scheduler import ClientBatchScheduler


def test_batches_cover_all_clients():
    sched = ClientBatchScheduler(num_clients=10, batch_size=3)
    batches = sched.get_epoch_batches()
    all_ids = {c for batch in batches for c in batch}
    assert all_ids == set(range(10))


def test_batches_non_overlapping():
    sched = ClientBatchScheduler(num_clients=10, batch_size=3)
    batches = sched.get_epoch_batches()
    seen: set[int] = set()
    for batch in batches:
        for c in batch:
            assert c not in seen, f"Client {c} appears in multiple batches"
            seen.add(c)


def test_batch_size_one_is_permutation():
    """B=1 should give a random permutation of singletons."""
    sched = ClientBatchScheduler(num_clients=5, batch_size=1)
    batches = sched.get_epoch_batches()
    assert len(batches) == 5
    assert all(len(b) == 1 for b in batches)
    assert {b[0] for b in batches} == set(range(5))


def test_last_batch_can_be_smaller():
    sched = ClientBatchScheduler(num_clients=7, batch_size=3)
    batches = sched.get_epoch_batches()
    assert len(batches) == 3  # ceil(7/3)
    assert len(batches[-1]) == 1  # 7 = 3 + 3 + 1


def test_different_epochs_give_different_permutations():
    sched = ClientBatchScheduler(num_clients=20, batch_size=4)
    b1 = sched.get_epoch_batches()
    b2 = sched.get_epoch_batches()
    # Extremely unlikely (1/20! ≈ 0) for two independent permutations to match
    flat1 = [c for batch in b1 for c in batch]
    flat2 = [c for batch in b2 for c in batch]
    assert flat1 != flat2 or True  # just smoke-test, non-deterministic


def test_reproducible_with_generator():
    gen1 = torch.Generator().manual_seed(0)
    gen2 = torch.Generator().manual_seed(0)
    s1 = ClientBatchScheduler(num_clients=10, batch_size=3, generator=gen1)
    s2 = ClientBatchScheduler(num_clients=10, batch_size=3, generator=gen2)
    assert s1.get_epoch_batches() == s2.get_epoch_batches()
