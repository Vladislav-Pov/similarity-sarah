"""The quarantined ablations must not drift from the main algorithm."""

from similarity_sarah.algorithms.base import AlgorithmCtx
from similarity_sarah.algorithms.nfg_ss import NFGSS
from similarity_sarah.experimental.nfg_ss_ablations import NfgSSAblations
from similarity_sarah.prox import AccvrsBatchSGD
from tests.golden_oracle import _weight_sha256, build_fast_setup


def _solver() -> AccvrsBatchSGD:
    return AccvrsBatchSGD(
        num_steps=4, lr=None, L1=200.0, lr_factor=0.1, weight_decay=0.1,
        momentum=0.9, grad_clip=0.0, inner_decay_factor=1.0, inner_decay_period=None,
        early_stop_ratio=1e-4, include_linear_term=False, eval_batches=2,
    )


def _hash(make_algo) -> str:
    model, sg, sp, clients, loss_fn, device = build_fast_setup()
    algo = make_algo(_solver())
    algo.bind(AlgorithmCtx(model, sg, sp, clients, loss_fn, device))
    algo.run_epoch(0)
    return _weight_sha256(model)


def test_ablations_default_path_matches_nfg_ss():
    base = _hash(lambda s: NFGSS(theta=0.2, batch_size_clients=1, prox_solver=s))
    ablations = _hash(
        lambda s: NfgSSAblations(theta=0.2, batch_size_clients=1, prox_solver=s)
    )
    assert base == ablations
