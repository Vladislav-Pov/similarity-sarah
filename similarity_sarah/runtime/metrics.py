"""Metrics logging: a small protocol with W&B and no-op implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class MetricsLogger(Protocol):
    """Minimal interface for logging training metrics."""

    def log(
        self, *, algo: str, split: str, metrics: Mapping[str, float], step: int
    ) -> None: ...

    def log_summary(self, *, algo: str, metrics: Mapping[str, float]) -> None: ...

    def finish(self) -> None: ...


class NullLogger:
    """No-op logger — the default when W&B is disabled.

    Implementing the protocol lets call sites drop ``if logger is not None``
    guards entirely.
    """

    def log(
        self, *, algo: str, split: str, metrics: Mapping[str, float], step: int
    ) -> None:
        pass

    def log_summary(self, *, algo: str, metrics: Mapping[str, float]) -> None:
        pass

    def finish(self) -> None:
        pass


@dataclass
class WandbLogger:
    """Weights & Biases-backed metrics logger (``wandb`` imported lazily)."""

    run: Any

    @classmethod
    def start(
        cls,
        *,
        project: str,
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        job_type: str | None = None,
        name: str | None = None,
        notes: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> WandbLogger:
        import wandb

        run = wandb.init(
            project=project, entity=entity, group=group, tags=tags,
            job_type=job_type, name=name, notes=notes, config=config,
        )
        return cls(run=run)

    def log(
        self, *, algo: str, split: str, metrics: Mapping[str, float], step: int
    ) -> None:
        payload = {
            f"{algo}/{split}/{key}": value
            for key, value in metrics.items()
            if value is not None
        }
        if payload:
            self.run.log(payload, step=step)

    def log_summary(self, *, algo: str, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            if value is not None:
                self.run.summary[f"{algo}/{key}"] = value

    def finish(self) -> None:
        self.run.finish()
