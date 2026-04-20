"""Entry point — run experiments via Hydra."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from similarity_sarah.runtime.runner import Runner

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("=== Experiment config ===\n%s", cfg)
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
