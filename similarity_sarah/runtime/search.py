"""Hyper-parameter search utilities for SARAH experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, Iterator, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf


def _as_list(value: Iterable[object] | object) -> list[object]:
    """Turn a Hydra/OmegaConf node into a plain list of choices.

    ``ListConfig([2, 4])`` is not a ``list``, so a naive ``isinstance(..., list)``
    would wrap the whole node in ``[...]`` and Optuna would sample the list as
    one categorical value (then ``batch_size_clients`` became ``ListConfig``).
    """
    if isinstance(value, list):
        return list(value)
    if isinstance(value, ListConfig):
        return [
            OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
            for v in value
        ]
    return [OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value]


def _to_python(value: object) -> object:
    """Leaf values from trials may still be OmegaConf scalars; normalize for the runner."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _grid(values: Mapping[str, Iterable[object]]) -> Iterator[dict[str, object]]:
    keys = list(values.keys())
    lists = [list(values[key]) for key in keys]
    for combo in product(*lists):
        yield dict(zip(keys, combo))


def _coerce_dict(cfg: DictConfig | None) -> dict[str, object]:
    if cfg is None:
        return {}
    return {key: cfg[key] for key in cfg.keys()}


def _override_algorithm(cfg: DictConfig, algo_cfg: DictConfig) -> DictConfig:
    merged = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    merged.algorithm = algo_cfg
    return merged


def _join_params(prefix: str, params: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}.{key}": value for key, value in params.items()}


@dataclass
class SearchConfig:
    method: str
    max_epochs: int
    num_trials: int
    wandb_project: str | None
    wandb_group: str | None
    distributed: DictConfig
    batched: DictConfig


class GridSearch:
    """Deterministic grid search across parameter grids."""

    def __init__(self, cfg: DictConfig, base_cfg: DictConfig) -> None:
        self.cfg = cfg
        self.base_cfg = base_cfg

    def _iter_trials(self) -> Iterator[dict[str, object]]:
        shared_grid = _grid(_coerce_dict(getattr(self.cfg, "shared", None)))
        dist_grid = _grid(_coerce_dict(self.cfg.distributed_sarah))
        batched_grid = _grid(_coerce_dict(self.cfg.batched_nfg_sarah))

        for shared_params in shared_grid:
            for dist_params in dist_grid:
                for batched_params in batched_grid:
                    params = {}
                    params.update(_join_params("shared", shared_params))
                    params.update(_join_params("distributed_sarah", dist_params))
                    params.update(_join_params("batched_nfg_sarah", batched_params))
                    yield params

    def run(
        self,
        evaluator: Callable[
            [int, dict[str, DictConfig], dict[str, object]],
            dict[str, object],
        ],
    ) -> dict[str, object]:
        best = {"score": -float("inf")}
        for idx, params in enumerate(self._iter_trials()):
            if idx >= int(self.cfg.num_trials):
                break
            trial_cfgs = _build_trial_cfgs(self.base_cfg, self.cfg, params)
            result = evaluator(
                trial_id=idx,
                trial_cfgs=trial_cfgs,
                trial_params=params,
            )
            score = _score_trial(result)
            if score > best["score"]:
                best = {"score": score, "result": result}
        return best


class OptunaSearch:
    """Optuna-powered search across parameter ranges."""

    def __init__(self, cfg: DictConfig, base_cfg: DictConfig) -> None:
        self.cfg = cfg
        self.base_cfg = base_cfg

    def run(
        self,
        evaluator: Callable[
            [int, dict[str, DictConfig], dict[str, object]],
            dict[str, object],
        ],
    ) -> dict[str, object]:
        import optuna

        def objective(trial: "optuna.Trial") -> float:
            params = _sample_optuna_params(trial, self.cfg)
            trial_cfgs = _build_trial_cfgs(self.base_cfg, self.cfg, params)
            result = evaluator(
                trial_id=trial.number,
                trial_cfgs=trial_cfgs,
                trial_params=params,
            )
            score = _score_trial(result)
            trial.set_user_attr("result", result)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(self.cfg.num_trials))
        best = study.best_trial
        return {
            "score": best.value,
            "trial": best.number,
            "params": best.params,
            "result": best.user_attrs.get("result"),
        }


def _build_trial_cfgs(
    base_cfg: DictConfig,
    search_cfg: DictConfig,
    params: Mapping[str, object],
) -> dict[str, DictConfig]:
    dist_cfg = OmegaConf.create(
        OmegaConf.to_container(base_cfg.algorithm, resolve=True),
    )
    batched_cfg = OmegaConf.create(
        OmegaConf.to_container(base_cfg.algorithm, resolve=True),
    )

    dist_cfg.name = "distributed_sarah"
    batched_cfg.name = "batched_nfg_sarah"

    for key, value in params.items():
        v = _to_python(value)
        if key.startswith("shared."):
            shared_key = key.split(".", 1)[1]
            dist_cfg[shared_key] = v
            batched_cfg[shared_key] = v
        elif key.startswith("distributed_sarah."):
            dist_cfg[key.split(".", 1)[1]] = v
        elif key.startswith("batched_nfg_sarah."):
            batched_cfg[key.split(".", 1)[1]] = v

    max_epochs = int(search_cfg.max_epochs)
    dist_cfg.num_epochs = min(int(dist_cfg.num_epochs), max_epochs)
    batched_cfg.num_epochs = min(int(batched_cfg.num_epochs), max_epochs)

    return {
        "distributed_sarah": _override_algorithm(base_cfg, dist_cfg),
        "batched_nfg_sarah": _override_algorithm(base_cfg, batched_cfg),
    }


def _sample_optuna_params(
    trial: "optuna.Trial",
    search_cfg: DictConfig,
) -> dict[str, object]:
    params: dict[str, object] = {}

    for name, values in _coerce_dict(getattr(search_cfg, "shared", None)).items():
        values_list = _as_list(values)
        params[f"shared.{name}"] = trial.suggest_categorical(
            f"shared.{name}",
            values_list,
        )

    for name, values in _coerce_dict(search_cfg.distributed_sarah).items():
        values_list = _as_list(values)
        params[f"distributed_sarah.{name}"] = trial.suggest_categorical(
            f"distributed_sarah.{name}",
            values_list,
        )

    for name, values in _coerce_dict(search_cfg.batched_nfg_sarah).items():
        values_list = _as_list(values)
        params[f"batched_nfg_sarah.{name}"] = trial.suggest_categorical(
            f"batched_nfg_sarah.{name}",
            values_list,
        )

    return params


def _score_trial(result: dict[str, object]) -> float:
    summaries = result.get("summaries")
    if not isinstance(summaries, dict):
        return -float("inf")

    dist_summary = summaries.get("distributed_sarah", {})
    batched_summary = summaries.get("batched_nfg_sarah", {})

    dist_acc = dist_summary.get("best_val_accuracy", -float("inf"))
    batched_acc = batched_summary.get("best_val_accuracy", -float("inf"))

    if dist_acc == -float("inf") or batched_acc == -float("inf"):
        return -float("inf")

    return float(dist_acc + batched_acc) / 2.0
