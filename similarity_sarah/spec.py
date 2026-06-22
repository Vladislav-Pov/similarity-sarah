"""Typed, frozen configuration specs parsed once from a Hydra ``DictConfig``.

The pre-rewrite runner read configuration lazily and repeatedly via
``OmegaConf.select(..., default=...)`` scattered across ``runner.py:259-370``.
This module reads it *once*, up front, into immutable dataclasses, so the rest
of the code consumes plain typed attributes instead of an untyped config.

Every default here mirrors the corresponding pre-rewrite ``OmegaConf.select``
default exactly; ``tests/test_spec.py`` pins this against the three
``docs/reference_runs/*.json`` configs.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ProxSpec:
    """Inexact proximal-solver configuration (``nfg_ss`` / ``svrs``)."""

    kind: str
    num_steps: int
    lr: float | None
    momentum: float
    weight_decay: float
    grad_clip: float
    eval_batches: int
    v_schedule: str
    # AccVRS-specific.
    L1: float
    lr_factor: float
    inner_decay_factor: float
    inner_decay_period: int | None
    early_stop_ratio: float
    include_linear_term: bool
    # Adam-specific.
    adam_betas: tuple[float, float]


@dataclass(frozen=True)
class ScheduleSpec:
    """Per-epoch ``prox_lr`` schedule. Unset numeric fields resolve at apply."""

    kind: str
    t_max: int | None
    min_factor: float
    step_every: int | None
    gamma: float


@dataclass(frozen=True)
class DataSpec:
    name: str
    val_fraction: float
    augment_server: bool
    augment_train: bool
    n_train: int | None
    n_test: int | None
    data_dir: str | None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    server_fraction: float | None


@dataclass(frozen=True)
class RuntimeSpec:
    device: str
    deterministic: bool
    eval_every: int
    num_workers: int
    batch_size: int
    large_batch_size: int
    batch_size_server_grad: int
    batch_size_server_prox: int
    batch_size_data_clients: int
    prox_lr_schedule: ScheduleSpec | None


@dataclass(frozen=True)
class RunSpec:
    """Fully resolved configuration for one experiment run."""

    seed: int
    algorithm_name: str
    num_epochs: int
    num_clients: int
    batch_size_clients: int
    theta: float | None
    lr: float | None
    weight_decay: float
    momentum: float
    lr_schedule: str
    lr_min_factor: float
    include_server: bool
    prox: ProxSpec | None
    data: DataSpec
    model: ModelSpec
    partition: PartitionSpec
    runtime: RuntimeSpec


def parse_prox_spec(algo_cfg: DictConfig) -> ProxSpec:
    """Parse the proximal-solver fields of an algorithm config."""
    lr_raw = OmegaConf.select(algo_cfg, "prox_lr", default=None)
    decay_period_raw = OmegaConf.select(algo_cfg, "prox_inner_decay_period", default=None)
    return ProxSpec(
        kind=str(OmegaConf.select(algo_cfg, "prox_solver", default="sgd")).lower(),
        num_steps=int(OmegaConf.select(algo_cfg, "prox_num_steps", default=0)),
        lr=None if lr_raw is None else float(lr_raw),
        momentum=float(OmegaConf.select(algo_cfg, "prox_momentum", default=0.0)),
        weight_decay=float(OmegaConf.select(algo_cfg, "prox_weight_decay", default=0.0)),
        grad_clip=float(OmegaConf.select(algo_cfg, "prox_grad_clip", default=0.0)),
        eval_batches=int(OmegaConf.select(algo_cfg, "prox_eval_batches", default=1)),
        v_schedule=str(
            OmegaConf.select(algo_cfg, "prox_v_schedule", default="constant")
        ).lower(),
        L1=float(OmegaConf.select(algo_cfg, "prox_L1", default=200.0)),
        lr_factor=float(OmegaConf.select(algo_cfg, "prox_lr_factor", default=1.0)),
        inner_decay_factor=float(
            OmegaConf.select(algo_cfg, "prox_inner_decay_factor", default=0.9)
        ),
        inner_decay_period=None if decay_period_raw is None else int(decay_period_raw),
        early_stop_ratio=float(
            OmegaConf.select(algo_cfg, "prox_inner_early_stop_ratio", default=1e-3)
        ),
        include_linear_term=bool(
            OmegaConf.select(algo_cfg, "prox_include_linear_term", default=False)
        ),
        adam_betas=(
            float(OmegaConf.select(algo_cfg, "prox_adam_beta1", default=0.9)),
            float(OmegaConf.select(algo_cfg, "prox_adam_beta2", default=0.999)),
        ),
    )


def _parse_schedule(rt: DictConfig) -> ScheduleSpec | None:
    sched = OmegaConf.select(rt, "prox_lr_schedule", default=None)
    if sched is None:
        return None
    t_max = OmegaConf.select(sched, "t_max", default=None)
    step_every = OmegaConf.select(sched, "step_every", default=None)
    return ScheduleSpec(
        kind=str(OmegaConf.select(sched, "kind", default="constant")).lower(),
        t_max=None if t_max is None else int(t_max),
        min_factor=float(OmegaConf.select(sched, "min_factor", default=0.0)),
        step_every=None if step_every is None else int(step_every),
        gamma=float(OmegaConf.select(sched, "gamma", default=0.1)),
    )


def _parse_runtime(rt: DictConfig) -> RuntimeSpec:
    base_bs = int(OmegaConf.select(rt, "batch_size", default=512))
    large_bs = int(OmegaConf.select(rt, "large_batch_size", default=base_bs))
    return RuntimeSpec(
        device=str(OmegaConf.select(rt, "device", default="auto")),
        deterministic=bool(OmegaConf.select(rt, "deterministic", default=False)),
        eval_every=int(OmegaConf.select(rt, "eval_every", default=1)),
        num_workers=int(OmegaConf.select(rt, "num_workers", default=0)),
        batch_size=base_bs,
        large_batch_size=large_bs,
        batch_size_server_grad=int(
            OmegaConf.select(rt, "batch_size_server_grad", default=large_bs)
        ),
        batch_size_server_prox=int(
            OmegaConf.select(rt, "batch_size_server_prox", default=base_bs)
        ),
        batch_size_data_clients=int(
            OmegaConf.select(rt, "batch_size_data_clients", default=large_bs)
        ),
        prox_lr_schedule=_parse_schedule(rt),
    )


def _parse_data(data: DictConfig) -> DataSpec:
    n_train = OmegaConf.select(data, "n_train", default=None)
    n_test = OmegaConf.select(data, "n_test", default=None)
    data_dir = OmegaConf.select(data, "data_dir", default=None)
    return DataSpec(
        name=str(data.name),
        val_fraction=float(OmegaConf.select(data, "val_fraction", default=0.0)),
        augment_server=bool(OmegaConf.select(data, "augment_server", default=False)),
        augment_train=bool(OmegaConf.select(data, "augment_train", default=False)),
        n_train=None if n_train is None else int(n_train),
        n_test=None if n_test is None else int(n_test),
        data_dir=None if data_dir is None else str(data_dir),
    )


def parse_run_spec(cfg: DictConfig) -> RunSpec:
    """Parse a fully-composed Hydra config into an immutable :class:`RunSpec`."""
    algo = cfg.algorithm
    has_prox = OmegaConf.select(algo, "prox_num_steps", default=None) is not None
    theta = OmegaConf.select(algo, "theta", default=None)
    lr = OmegaConf.select(algo, "lr", default=None)
    return RunSpec(
        seed=int(cfg.seed),
        algorithm_name=str(algo.name),
        num_epochs=int(OmegaConf.select(algo, "num_epochs", default=0)),
        num_clients=int(OmegaConf.select(algo, "num_clients", default=0)),
        batch_size_clients=int(OmegaConf.select(algo, "batch_size_clients", default=1)),
        theta=None if theta is None else float(theta),
        lr=None if lr is None else float(lr),
        weight_decay=float(OmegaConf.select(algo, "weight_decay", default=0.0)),
        momentum=float(OmegaConf.select(algo, "momentum", default=0.0)),
        lr_schedule=str(OmegaConf.select(algo, "lr_schedule", default="constant")).lower(),
        lr_min_factor=float(OmegaConf.select(algo, "lr_min_factor", default=0.0)),
        include_server=bool(OmegaConf.select(algo, "include_server", default=True)),
        prox=parse_prox_spec(algo) if has_prox else None,
        data=_parse_data(cfg.data),
        model=ModelSpec(
            name=str(cfg.model.name),
            num_classes=int(OmegaConf.select(cfg.model, "num_classes", default=10)),
        ),
        partition=PartitionSpec(
            name=str(cfg.partition.name),
            server_fraction=(
                None
                if OmegaConf.select(cfg.partition, "server_fraction", default=None) is None
                else float(OmegaConf.select(cfg.partition, "server_fraction"))
            ),
        ),
        runtime=_parse_runtime(cfg.runtime),
    )
