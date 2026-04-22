"""Experiment runner — wires data, model, task, and algorithm together."""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Mapping

from omegaconf import OmegaConf

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.algorithms.batched_nfg_sarah import BatchedNoFullGradSARAH
from similarity_sarah.algorithms.distributed_sarah import DistributedSARAH
from similarity_sarah.algorithms.svrs import SVRS
from similarity_sarah.data.datasets import (
    load_augmented_train,
    load_dataset,
    split_train_val,
)
from similarity_sarah.data.partition import create_partition
from similarity_sarah.models.simple_cnn import SimpleCNN
from similarity_sarah.models.resnet import ResNet18_32x32
from similarity_sarah.runtime.prox_solver import (
    InexactProxAdam,
    InexactProxSGD,
    ProxSolver,
)
from similarity_sarah.runtime.metrics_logger import MetricsLogger, WandbLogger
from similarity_sarah.tasks.classification import ClassificationTask
from similarity_sarah.utils import set_seed

logger = logging.getLogger(__name__)

# Short labels for W&B run names / tags (keys = Hydra search param paths).
_WANDB_PARAM_ABBREV: dict[str, str] = {
    "shared.batch_size_clients": "B",
    "shared.num_epochs": "E",
    "distributed_sarah.lr": "dlr",
    "batched_nfg_sarah.theta": "th",
    "batched_nfg_sarah.prox_lr": "plr",
    "batched_nfg_sarah.prox_num_steps": "pns",
    "batched_nfg_sarah.prox_momentum": "pmom",
    "batched_nfg_sarah.prox_weight_decay": "pwd",
    "svrs.theta": "svth",
}


def _wandb_abbrev_param_key(key: str) -> str:
    if key in _WANDB_PARAM_ABBREV:
        return _WANDB_PARAM_ABBREV[key]
    tail = key.split(".")[-1]
    return tail[:16] if len(tail) > 16 else tail


def _format_param_value_for_wandb(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _wandb_trial_run_name(trial_id: int, trial_params: Mapping[str, object]) -> str:
    """Build a readable W&B run name: trial index + compact hyperparameter summary."""
    parts: list[str] = [f"t{trial_id}"]
    for key in sorted(trial_params.keys()):
        abbrev = _wandb_abbrev_param_key(key)
        val = _format_param_value_for_wandb(trial_params[key])
        parts.append(f"{abbrev}={val}")
    raw = "__".join(parts)
    safe = "".join(c if c.isalnum() or c in "-_=." else "_" for c in raw)
    max_len = 128
    if len(safe) > max_len:
        safe = safe[: max_len - 3] + "..."
    return safe


def _wandb_trial_param_tags(
    trial_id: int,
    trial_params: Mapping[str, object],
) -> list[str]:
    """One tag per hyperparameter (and trial id) for W&B table filters."""
    tags: list[str] = [f"trial_id={trial_id}"]
    for key in sorted(trial_params.keys()):
        abbrev = _wandb_abbrev_param_key(key)
        val = _format_param_value_for_wandb(trial_params[key])
        tag = f"{abbrev}={val}"
        if len(tag) > 64:
            tag = tag[:61] + "..."
        tags.append(tag)
    return tags


class Runner:
    """Top-level object that sets up and executes an experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)

        self._setup_device()
        self._setup_data()
        self._setup_task()

        self._logger: MetricsLogger | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_device(self) -> None:
        device_str: str = self.cfg.runtime.device
        if device_str == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.device = torch.device(device_str)
        logger.info("Using device: %s", self.device)

    def _setup_data(self) -> None:
        from torch.utils.data import Subset

        train_dataset, test_dataset = load_dataset(self.cfg.data)
        train_dataset, val_dataset = split_train_val(
            train_dataset, self.cfg.data.val_fraction,
        )

        num_clients: int = self.cfg.algorithm.num_clients
        total_nodes = num_clients + 1
        partitions = create_partition(
            train_dataset, total_nodes, self.cfg.partition,
        )

        rt = self.cfg.runtime
        base_bs: int = int(rt.batch_size)
        bs_server = int(OmegaConf.select(rt, "batch_size_server", default=base_bs))
        bs_data_clients = int(
            OmegaConf.select(rt, "batch_size_data_clients", default=base_bs),
        )
        nw: int = self.cfg.runtime.num_workers
        eval_bs = max(bs_server, bs_data_clients, base_bs) * 2

        # Optional: server uses an *augmented* training set inside the
        # inexact prox solver.  Clients keep a deterministic transform so
        # that ∇f_i(w) is well-defined.
        augment_server = bool(
            OmegaConf.select(self.cfg.data, "augment_server", default=False),
        )
        server_loader_dataset = partitions[0]
        if augment_server:
            aug_train = load_augmented_train(self.cfg.data)
            if aug_train is not None:
                # The Subset's underlying dataset may be a random_split of the
                # full deterministic CIFAR-10; we can't trivially recover the
                # original indices.  Walk back through ``random_split`` and
                # ``create_partition`` to remap.
                aug_indices = self._original_indices(partitions[0])
                if aug_indices is not None:
                    server_loader_dataset = Subset(aug_train, aug_indices)
                    logger.info(
                        "Server loader uses augmented CIFAR-10 (%d samples).",
                        len(aug_indices),
                    )
                else:
                    logger.warning(
                        "augment_server=True but could not remap server indices; "
                        "falling back to deterministic transform.",
                    )

        self.server_loader = DataLoader(
            server_loader_dataset, batch_size=bs_server, shuffle=True,
            num_workers=nw,
        )
        self.client_loaders = [
            DataLoader(p, batch_size=bs_data_clients, shuffle=True, num_workers=nw)
            for p in partitions[1:]
        ]
        self.test_loader = DataLoader(
            test_dataset, batch_size=eval_bs, shuffle=False, num_workers=nw,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=eval_bs, shuffle=False, num_workers=nw,
        )

        logger.info(
            "Data: %d server samples, %d clients, %d test, %d val "
            "(minibatch server=%d, clients=%d, augment_server=%s)",
            len(partitions[0]),
            num_clients,
            len(test_dataset),
            len(val_dataset),
            bs_server,
            bs_data_clients,
            augment_server,
        )

    @staticmethod
    def _original_indices(dataset) -> list[int] | None:
        """Walk through ``Subset`` chains to recover indices into the *root* dataset."""
        from torch.utils.data import Subset

        indices: list[int] = []
        cur = dataset
        # Compose the index maps starting from the leaf Subset.
        chains: list[list[int]] = []
        while isinstance(cur, Subset):
            chains.append(list(cur.indices))
            cur = cur.dataset
        if not chains:
            return None
        # Innermost subset is the last appended; resolve outwards.
        chains.reverse()
        idx = chains[0]
        for nxt in chains[1:]:
            idx = [idx[i] for i in nxt]
        indices = idx
        return indices

    def _setup_model(self) -> None:
        name: str = self.cfg.model.name
        if name == "simple_cnn":
            self.model = SimpleCNN(
                num_classes=self.cfg.model.num_classes,
            ).to(self.device)
        elif name == "resnet18_32x32":
            self.model = ResNet18_32x32(
                num_classes=self.cfg.model.num_classes,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model: {name}")
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: %s  (%d parameters)", name, n_params)

    def _setup_task(self) -> None:
        self.task = ClassificationTask()

    def _build_prox_solver(self, algo_cfg: DictConfig) -> ProxSolver:
        """Build the inexact proximal solver from algorithm configuration."""
        kind = str(OmegaConf.select(algo_cfg, "prox_solver", default="sgd")).lower()
        num_steps = int(algo_cfg.prox_num_steps)
        lr = float(algo_cfg.prox_lr)
        weight_decay = float(
            OmegaConf.select(algo_cfg, "prox_weight_decay", default=0.0),
        )

        if kind == "sgd":
            momentum = float(
                OmegaConf.select(algo_cfg, "prox_momentum", default=0.0),
            )
            return InexactProxSGD(
                num_steps=num_steps,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        if kind == "adam":
            beta1 = float(OmegaConf.select(algo_cfg, "prox_adam_beta1", default=0.9))
            beta2 = float(OmegaConf.select(algo_cfg, "prox_adam_beta2", default=0.999))
            return InexactProxAdam(
                num_steps=num_steps,
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unknown prox_solver: {kind}")

    def _setup_algorithm(self, cfg: DictConfig) -> None:
        algo_cfg = cfg.algorithm
        algorithm: BaseAlgorithm

        if algo_cfg.name == "batched_nfg_sarah":
            algorithm = BatchedNoFullGradSARAH(
                theta=algo_cfg.theta,
                batch_size_clients=algo_cfg.batch_size_clients,
                prox_solver=self._build_prox_solver(algo_cfg),
            )
        elif algo_cfg.name == "svrs":
            algorithm = SVRS(
                theta=algo_cfg.theta,
                batch_size_clients=algo_cfg.batch_size_clients,
                prox_solver=self._build_prox_solver(algo_cfg),
            )
        elif algo_cfg.name == "distributed_sarah":
            algorithm = DistributedSARAH(
                lr=algo_cfg.lr,
                batch_size_clients=algo_cfg.batch_size_clients,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_cfg.name}")

        algorithm.initialize(
            model=self.model,
            server_loader=self.server_loader,
            client_loaders=self.client_loaders,
            loss_fn=self.task.loss_fn,
            device=self.device,
        )
        self.algorithm = algorithm
        logger.info("Algorithm: %s", algo_cfg.name)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        search_cfg = self.cfg.search
        if not getattr(search_cfg, "enabled", False):
            self._setup_model()
            self._setup_algorithm(self.cfg)
            self._logger = self._maybe_create_logger(self.cfg, run_name=None)
            try:
                self._run_training(
                    self.cfg,
                    algo_name=self.cfg.algorithm.name,
                    run_name=self.cfg.algorithm.name,
                    global_step_offset=0,
                )
            finally:
                if self._logger is not None:
                    self._logger.finish()
            return

        self._run_search(search_cfg)

    # ------------------------------------------------------------------
    # Search / orchestration
    # ------------------------------------------------------------------
    def _run_search(self, search_cfg: DictConfig) -> None:
        method = str(search_cfg.method).lower()
        if method not in {"grid", "optuna"}:
            raise ValueError(f"Unknown search method: {search_cfg.method}")

        if method == "grid":
            from similarity_sarah.runtime.search import GridSearch

            searcher = GridSearch(search_cfg, self.cfg)
        else:
            from similarity_sarah.runtime.search import OptunaSearch

            searcher = OptunaSearch(search_cfg, self.cfg)

        best = searcher.run(self._evaluate_trial)

        logger.info("Search done. Best summary: %s", best)

    def _evaluate_trial(
        self,
        *,
        trial_id: int,
        trial_cfgs: dict[str, DictConfig],
        trial_params: dict[str, object],
        report_intermediate: Callable[[int, float], bool] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {"trial_id": trial_id, "params": trial_params}
        wandb_run_name = _wandb_trial_run_name(trial_id, trial_params)
        trial_tags = _wandb_trial_param_tags(trial_id, trial_params)
        logger.info("Search trial %s — W&B run name: %s", trial_id, wandb_run_name)

        search_cfg = self.cfg.search
        self._logger = self._maybe_create_logger(
            self.cfg,
            run_name=wandb_run_name,
            project_override=search_cfg.get("wandb_project"),
            group_override=search_cfg.get("wandb_group"),
            config_override=trial_params,
            extra_tags=trial_tags,
        )

        try:
            summaries = {}
            wandb_step_offset = 0
            for algo_name, algo_cfg in trial_cfgs.items():
                self._setup_model()
                self._setup_algorithm(algo_cfg)
                summary = self._run_training(
                    algo_cfg,
                    algo_name=algo_name,
                    run_name=f"{wandb_run_name}/{algo_name}",
                    global_step_offset=wandb_step_offset,
                    report_intermediate=report_intermediate,
                )
                summaries[algo_name] = summary
                wandb_step_offset += int(algo_cfg.algorithm.num_epochs)
            result["summaries"] = summaries
            return result
        finally:
            if self._logger is not None:
                self._logger.finish()
            self._logger = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _run_training(
        self,
        cfg: DictConfig,
        *,
        algo_name: str,
        run_name: str,
        global_step_offset: int,
        report_intermediate: "Callable[[int, float], bool] | None" = None,
    ) -> dict[str, float]:
        num_epochs: int = cfg.algorithm.num_epochs
        eval_every: int = cfg.runtime.eval_every

        logger.info("Starting training for %d epochs", num_epochs)

        base_prox_lr = self._initial_prox_lr()
        sched_cfg = OmegaConf.select(cfg.runtime, "prox_lr_schedule", default=None)

        best_val_acc = -float("inf")
        best_val_loss = float("inf")
        last_val = None

        for epoch in range(num_epochs):
            self._apply_prox_lr_schedule(base_prox_lr, sched_cfg, epoch, num_epochs)

            t0 = time.perf_counter()
            metrics = dict(self.algorithm.run_epoch(epoch))
            elapsed = time.perf_counter() - t0
            metrics["epoch_time_s"] = float(elapsed)
            metrics["prox_lr"] = float(self._current_prox_lr() or 0.0)

            logger.info(
                "Epoch %d/%d  (%.1fs)  %s",
                epoch + 1, num_epochs, elapsed,
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
            )

            step = global_step_offset + epoch + 1
            if self._logger is not None:
                self._logger.log(
                    algo=algo_name,
                    split="train",
                    metrics={
                        k: v for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    },
                    step=step,
                )

            if (epoch + 1) % eval_every == 0:
                val_metrics = self.task.evaluate(
                    self.model, self.val_loader, self.device,
                )
                test_metrics = self.task.evaluate(
                    self.model, self.test_loader, self.device,
                )
                logger.info("  val:  %s", val_metrics)
                logger.info("  test: %s", test_metrics)

                if self._logger is not None:
                    self._logger.log(
                        algo=algo_name,
                        split="val",
                        metrics=val_metrics,
                        step=step,
                    )
                    self._logger.log(
                        algo=algo_name,
                        split="test",
                        metrics=test_metrics,
                        step=step,
                    )

                last_val = val_metrics
                best_val_acc = max(best_val_acc, val_metrics.get("accuracy", -1))
                best_val_loss = min(best_val_loss, val_metrics.get("loss", float("inf")))

                if report_intermediate is not None:
                    should_stop = report_intermediate(
                        epoch + 1, float(val_metrics.get("accuracy", -1.0)),
                    )
                    if should_stop:
                        logger.info(
                            "Early stop at epoch %d (val acc=%.4f)",
                            epoch + 1, val_metrics.get("accuracy", -1.0),
                        )
                        break

        summary = {
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        if last_val is not None:
            summary["last_val_accuracy"] = last_val.get("accuracy")
            summary["last_val_loss"] = last_val.get("loss")

        if self._logger is not None:
            self._logger.log_summary(algo=algo_name, metrics=summary)

        return summary

    # ------------------------------------------------------------------
    # Prox LR schedule
    # ------------------------------------------------------------------
    def _initial_prox_lr(self) -> float | None:
        prox = getattr(self.algorithm, "prox_solver", None)
        if prox is None or not hasattr(prox, "lr"):
            return None
        return float(prox.lr)

    def _current_prox_lr(self) -> float | None:
        prox = getattr(self.algorithm, "prox_solver", None)
        if prox is None or not hasattr(prox, "lr"):
            return None
        return float(prox.lr)

    def _apply_prox_lr_schedule(
        self,
        base_lr: float | None,
        sched_cfg: DictConfig | None,
        epoch: int,
        num_epochs: int,
    ) -> None:
        if base_lr is None or sched_cfg is None:
            return
        kind = str(sched_cfg.get("kind", "constant")).lower()
        prox = getattr(self.algorithm, "prox_solver", None)
        if prox is None or not hasattr(prox, "lr"):
            return

        if kind == "constant":
            return
        if kind == "cosine":
            t_max = int(sched_cfg.get("t_max", num_epochs)) or 1
            min_factor = float(sched_cfg.get("min_factor", 0.0))
            t = min(epoch, t_max)
            cos = 0.5 * (1.0 + math.cos(math.pi * t / t_max))
            factor = min_factor + (1.0 - min_factor) * cos
            prox.lr = base_lr * factor
            return
        if kind == "step":
            step_every = int(sched_cfg.get("step_every", max(num_epochs // 3, 1)))
            gamma = float(sched_cfg.get("gamma", 0.1))
            k = epoch // max(step_every, 1)
            prox.lr = base_lr * (gamma ** k)
            return
        logger.warning("Unknown prox_lr_schedule.kind=%s; ignoring.", kind)

    # ------------------------------------------------------------------
    # W&B logging helpers
    # ------------------------------------------------------------------
    def _maybe_create_logger(
        self,
        cfg: DictConfig,
        *,
        run_name: str | None = None,
        project_override: str | None = None,
        group_override: str | None = None,
        config_override: dict[str, object] | None = None,
        extra_tags: list[str] | None = None,
    ) -> MetricsLogger | None:
        wandb_cfg = cfg.runtime.get("wandb")
        if not wandb_cfg or not wandb_cfg.get("enabled", False):
            return None

        project = project_override or wandb_cfg.get("project", "similarity-sarah")
        entity = wandb_cfg.get("entity")
        group = group_override or wandb_cfg.get("group")
        tags = list(wandb_cfg.get("tags") or [])
        if extra_tags:
            tags.extend(extra_tags)
        job_type = wandb_cfg.get("job_type")
        name = run_name or wandb_cfg.get("name")
        run_config = OmegaConf.to_container(cfg, resolve=True)
        if config_override:
            run_config = {"base": run_config, "search_params": config_override}

        notes: str | None = None
        if config_override:
            lines = [
                f"{k}: {_format_param_value_for_wandb(v)}"
                for k, v in sorted(config_override.items())
            ]
            notes = "Search trial hyperparameters:\n" + "\n".join(lines)

        return WandbLogger.start(
            project=project,
            entity=entity,
            group=group,
            tags=tags,
            job_type=job_type,
            name=name,
            notes=notes,
            config=run_config,
        )
