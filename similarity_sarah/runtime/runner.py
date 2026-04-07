"""Experiment runner — wires data, model, task, and algorithm together."""

from __future__ import annotations

import logging
import time

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from similarity_sarah.algorithms.base import BaseAlgorithm
from similarity_sarah.algorithms.batched_nfg_sarah import BatchedNoFullGradSARAH
from similarity_sarah.algorithms.distributed_sarah import DistributedSARAH
from similarity_sarah.data.datasets import load_dataset, split_train_val
from similarity_sarah.data.partition import create_partition
from similarity_sarah.models.simple_cnn import SimpleCNN
from similarity_sarah.runtime.prox_solver import InexactProxSGD
from similarity_sarah.tasks.classification import ClassificationTask
from similarity_sarah.utils import set_seed

logger = logging.getLogger(__name__)


class Runner:
    """Top-level object that sets up and executes an experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)

        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_task()
        self._setup_algorithm()

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
        train_dataset, test_dataset = load_dataset(self.cfg.data)
        train_dataset, val_dataset = split_train_val(
            train_dataset, self.cfg.data.val_fraction,
        )

        num_clients: int = self.cfg.algorithm.num_clients
        total_nodes = num_clients + 1
        partitions = create_partition(
            train_dataset, total_nodes, self.cfg.partition,
        )

        bs: int = self.cfg.runtime.batch_size
        nw: int = self.cfg.runtime.num_workers

        self.server_loader = DataLoader(
            partitions[0], batch_size=bs, shuffle=True, num_workers=nw,
        )
        self.client_loaders = [
            DataLoader(p, batch_size=bs, shuffle=True, num_workers=nw)
            for p in partitions[1:]
        ]
        self.test_loader = DataLoader(
            test_dataset, batch_size=bs * 2, shuffle=False, num_workers=nw,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=bs * 2, shuffle=False, num_workers=nw,
        )

        logger.info(
            "Data: %d server samples, %d clients, %d test, %d val",
            len(partitions[0]),
            num_clients,
            len(test_dataset),
            len(val_dataset),
        )

    def _setup_model(self) -> None:
        name: str = self.cfg.model.name
        if name == "simple_cnn":
            self.model = SimpleCNN(
                num_classes=self.cfg.model.num_classes,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model: {name}")
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: %s  (%d parameters)", name, n_params)

    def _setup_task(self) -> None:
        self.task = ClassificationTask()

    def _setup_algorithm(self) -> None:
        algo_cfg = self.cfg.algorithm
        algorithm: BaseAlgorithm

        if algo_cfg.name == "batched_nfg_sarah":
            prox_solver = InexactProxSGD(
                num_steps=algo_cfg.prox_num_steps,
                lr=algo_cfg.prox_lr,
            )
            algorithm = BatchedNoFullGradSARAH(
                theta=algo_cfg.theta,
                batch_size_clients=algo_cfg.batch_size_clients,
                prox_solver=prox_solver,
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
        num_epochs: int = self.cfg.algorithm.num_epochs
        eval_every: int = self.cfg.runtime.eval_every

        logger.info("Starting training for %d epochs", num_epochs)

        for epoch in range(num_epochs):
            t0 = time.perf_counter()
            metrics = self.algorithm.run_epoch(epoch)
            elapsed = time.perf_counter() - t0

            logger.info(
                "Epoch %d/%d  (%.1fs)  %s",
                epoch + 1, num_epochs, elapsed, metrics,
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
