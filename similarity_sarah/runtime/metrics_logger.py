"""Metric logging helpers (W&B and optional protocols)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class MetricsLogger(Protocol):
    """Minimal interface for logging training metrics."""

    def log(
        self,
        *,
        algo: str,
        split: str,
        metrics: Mapping[str, float],
        step: int,
    ) -> None:
        """Log metrics for a given algorithm/split at the step."""

    def log_summary(self, *, algo: str, metrics: Mapping[str, float]) -> None:
        """Write summary metrics (best/last) for the algorithm."""

    def log_raw(self, metrics: Mapping[str, float]) -> None:
        """Log raw metrics without namespacing."""

    def finish(self) -> None:
        """Close the logger and flush remaining data."""


@dataclass
class WandbLogger:
    """Weights & Biases-backed metrics logger."""

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
        config: Mapping[str, Any] | None = None,
    ) -> "WandbLogger":
        import wandb

        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            tags=tags,
            job_type=job_type,
            name=name,
            notes=notes,
            config=config,
        )
        return cls(run=run)

    def log(
        self,
        *,
        algo: str,
        split: str,
        metrics: Mapping[str, float],
        step: int,
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
            if value is None:
                continue
            self.run.summary[f"{algo}/{key}"] = value

    def log_raw(self, metrics: Mapping[str, float]) -> None:
        if metrics:
            self.run.log(metrics)

    def finish(self) -> None:
        self.run.finish()
