"""Training loop and single-run orchestration for the rewritten pipeline.

``run_experiment`` wires a :class:`RunSpec` into data, model, algorithm, and a
:class:`TrainLoop`. The loop owns only the epoch cadence: per-epoch prox-lr
schedule, ``algorithm.run_epoch``, periodic evaluation, and logging.
"""

from __future__ import annotations

import logging
import math
import time

import torch

from similarity_sarah.algorithms.base import ALGORITHMS, Algorithm, AlgorithmCtx
from similarity_sarah.core.repro import make_generator, set_seed
from similarity_sarah.data.loaders import FederatedData, build_federated_data
from similarity_sarah.models import build_model
from similarity_sarah.runtime.metrics import MetricsLogger, NullLogger
from similarity_sarah.spec import RunSpec, ScheduleSpec
from similarity_sarah.tasks.classification import ClassificationTask

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, else honour the literal name."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _scaled_prox_lr(
    base_lr: float, sched: ScheduleSpec, epoch: int, num_epochs: int
) -> float | None:
    """Per-epoch prox-lr factor; ``None`` means leave the lr unchanged."""
    if sched.kind == "constant":
        return None
    if sched.kind == "cosine":
        t_max = sched.t_max or num_epochs or 1
        t = min(epoch, t_max)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / t_max))
        return base_lr * (sched.min_factor + (1.0 - sched.min_factor) * cos)
    if sched.kind == "step":
        step_every = sched.step_every or max(num_epochs // 3, 1)
        return base_lr * (sched.gamma ** (epoch // max(step_every, 1)))
    logger.warning("Unknown prox_lr_schedule.kind=%s; ignoring.", sched.kind)
    return None


class TrainLoop:
    """Drives the outer epochs: schedule, run_epoch, evaluate, log."""

    def __init__(
        self,
        *,
        algorithm: Algorithm,
        model: torch.nn.Module,
        data: FederatedData,
        task: ClassificationTask,
        device: torch.device,
        spec: RunSpec,
        logger: MetricsLogger,
    ) -> None:
        self.algorithm = algorithm
        self.model = model
        self.data = data
        self.task = task
        self.device = device
        self.spec = spec
        self.logger = logger

    def _base_prox_lr(self) -> float | None:
        prox = getattr(self.algorithm, "prox_solver", None)
        lr = getattr(prox, "lr", None)
        return None if lr is None else float(lr)

    def run(self) -> dict[str, float]:
        spec = self.spec
        algo_name = spec.algorithm_name
        base_prox_lr = self._base_prox_lr()
        sched = spec.runtime.prox_lr_schedule

        best_val_acc = -float("inf")
        best_val_loss = float("inf")

        for epoch in range(spec.num_epochs):
            if base_prox_lr is not None and sched is not None:
                new_lr = _scaled_prox_lr(base_prox_lr, sched, epoch, spec.num_epochs)
                if new_lr is not None:
                    self.algorithm.prox_solver.lr = new_lr  # type: ignore[attr-defined]

            t0 = time.perf_counter()
            metrics = dict(self.algorithm.run_epoch(epoch))
            metrics["epoch_time_s"] = time.perf_counter() - t0
            step = epoch + 1
            self.logger.log(algo=algo_name, split="train", metrics=metrics, step=step)

            if step % spec.runtime.eval_every == 0:
                val = self.task.evaluate(self.model, self.data.val_loader, self.device)
                test = self.task.evaluate(self.model, self.data.test_loader, self.device)
                self.logger.log(algo=algo_name, split="val", metrics=val, step=step)
                self.logger.log(algo=algo_name, split="test", metrics=test, step=step)
                best_val_acc = max(best_val_acc, val.get("accuracy", -1.0))
                best_val_loss = min(best_val_loss, val.get("loss", float("inf")))

        summary = {"best_val_accuracy": best_val_acc, "best_val_loss": best_val_loss}
        self.logger.log_summary(algo=algo_name, metrics=summary)
        return summary


def run_experiment(spec: RunSpec, *, logger: MetricsLogger | None = None) -> dict[str, float]:
    """Build everything from ``spec`` and train; return the run summary."""
    set_seed(spec.seed, deterministic=spec.runtime.deterministic)
    generator = make_generator(spec.seed) if spec.runtime.deterministic else None

    data = build_federated_data(spec, generator=generator)
    device = resolve_device(spec.runtime.device)
    model = build_model(spec.model.name, spec.model.num_classes).to(device)
    task = ClassificationTask()

    algo = ALGORITHMS.get(spec.algorithm_name).from_spec(spec)
    algo.bind(
        AlgorithmCtx(
            model, data.server_grad_loader, data.server_prox_loader,
            data.client_loaders, task.loss_fn, device,
        )
    )

    loop = TrainLoop(
        algorithm=algo, model=model, data=data, task=task, device=device, spec=spec,
        logger=logger or NullLogger(),
    )
    return loop.run()
