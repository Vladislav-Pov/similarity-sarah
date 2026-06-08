"""Entry point — run experiments via Hydra."""

from __future__ import annotations

import logging
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _maybe_wandb_logger(cfg: DictConfig):
    wandb_cfg = cfg.runtime.get("wandb")
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None
    from similarity_sarah.runtime.metrics import WandbLogger

    return WandbLogger.start(
        project=wandb_cfg.get("project", "similarity-sarah"),
        entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group"),
        tags=list(wandb_cfg.get("tags") or []),
        job_type=wandb_cfg.get("job_type"),
        name=wandb_cfg.get("name"),
        config=cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
    )


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("=== Experiment config ===\n%s", cfg)

    search = cfg.get("search")
    if search is not None and search.get("enabled", False):
        from similarity_sarah.runtime.search_runner import run_search

        run_search(search, cfg)
        return

    from similarity_sarah.runtime.loop import run_experiment
    from similarity_sarah.spec import parse_run_spec

    spec = parse_run_spec(cfg)
    mlogger = _maybe_wandb_logger(cfg)
    try:
        summary = run_experiment(spec, logger=mlogger)
        logger.info("Run summary: %s", summary)
    finally:
        if mlogger is not None:
            mlogger.finish()


if __name__ == "__main__":
    main()
