"""Hyperparameter-search entry point on the rewritten pipeline.

Reuses the backend-agnostic :class:`GridSearch` / :class:`OptunaSearch` from
``search.py`` and supplies a trial evaluator that builds each trial config into
a :class:`RunSpec` and trains it with :func:`run_experiment`. Optuna pruning is
wired through the loop's ``report_intermediate`` callback.

Trials log to a no-op logger (per-trial W&B is out of scope for the rewrite);
the search result carries each trial's ``best_val_accuracy``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from omegaconf import DictConfig

from similarity_sarah.runtime.loop import run_experiment
from similarity_sarah.runtime.search import GridSearch, OptunaSearch
from similarity_sarah.spec import parse_run_spec

logger = logging.getLogger(__name__)


def _evaluate_trial(
    *,
    trial_id: int,
    trial_cfgs: dict[str, DictConfig],
    trial_params: dict[str, object],
    report_intermediate: Callable[[int, float], bool] | None = None,
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for algo_name, full_cfg in trial_cfgs.items():
        spec = parse_run_spec(full_cfg)
        summaries[algo_name] = run_experiment(
            spec, report_intermediate=report_intermediate
        )
    return {"trial_id": trial_id, "params": trial_params, "summaries": summaries}


def run_search(search_cfg: DictConfig, base_cfg: DictConfig) -> dict[str, object]:
    """Run grid or Optuna search and return the best trial summary."""
    method = str(search_cfg.method).lower()
    if method == "grid":
        searcher: GridSearch | OptunaSearch = GridSearch(search_cfg, base_cfg)
    elif method == "optuna":
        searcher = OptunaSearch(search_cfg, base_cfg)
    else:
        raise ValueError(f"Unknown search method: {search_cfg.method}")

    best = searcher.run(_evaluate_trial)
    logger.info("Search done. Best summary: %s", best)
    return best
