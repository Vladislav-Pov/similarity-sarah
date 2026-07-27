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

    def log_deviation_series(
        self,
        *,
        algo: str,
        step: int,
        rows: list[Mapping[str, float]],
        split: str = "train",
    ) -> None:
        """Log a within-epoch estimator-deviation curve at ``step``.

        ``rows`` is one dict per inner step (``inner_step`` 0 = start of
        epoch) carrying the deviation flavours computed by the algorithm
        (``dev_norm``, ``dev_rel``, ``cos_sim``, ``angle_deg``, ``est_norm``,
        ``ref_norm``).  Emitted as a W&B ``Table`` plus prebuilt line plots
        of the proportional (``dev_rel``) and angular (``angle_deg``)
        deviation vs. ``inner_step`` — a single ``run.log`` call so the
        global step stays monotonic with the per-epoch scalar logging.
        Failures (e.g. a W&B API mismatch) are swallowed so a diagnostic
        never crashes training.
        """
        if not rows:
            return
        try:
            import wandb

            cols = [
                "inner_step", "dev_norm", "dev_rel",
                "cos_sim", "angle_deg", "est_norm", "ref_norm",
            ]
            table = wandb.Table(columns=cols)
            for r in rows:
                table.add_data(*[float(r.get(c, float("nan"))) for c in cols])

            prefix = f"{algo}/{split}/inner_deviation"
            payload: dict[str, Any] = {f"{prefix}/table": table}
            for y, title in (
                ("dev_rel", "‖v-g‖/‖g‖ within epoch"),
                ("angle_deg", "∠(v, g) within epoch [deg]"),
                ("cos_sim", "cos∠(v, g) within epoch"),
            ):
                payload[f"{prefix}/{y}_vs_step"] = wandb.plot.line(
                    table, "inner_step", y, title=title,
                )
            self.run.log(payload, step=step)
        except Exception:  # pragma: no cover - diagnostic must never crash
            import logging

            logging.getLogger(__name__).warning(
                "log_deviation_series failed; skipping W&B curve.",
                exc_info=True,
            )

    def finish(self) -> None:
        self.run.finish()
