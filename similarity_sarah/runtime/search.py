"""Hyper-parameter search utilities for SARAH experiments.

Two backends are provided:

* :class:`GridSearch` — deterministic Cartesian product over fixed lists.
* :class:`OptunaSearch` — TPE / random search over **continuous ranges**
  with intermediate-value reporting and Optuna's built-in pruners
  (Median / Hyperband).

The Optuna search expects each hyper-parameter spec to be a small
mapping of the form

    name:
      type: float | int | categorical
      low: <float|int>             # for float/int
      high: <float|int>            # for float/int
      log: true|false              # for float/int (default false)
      step: <float|int>            # optional, for float/int
      choices: [...]               # for categorical

This makes the search space explicit, validates it up-front, and keeps
the search machinery backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf

# configs/algorithm/*.yaml — used so search trials have all required keys even when
# ``defaults`` only load one algorithm (e.g. batched_nfg_sarah has no ``lr``).
_ALGORITHM_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "algorithm"


def _algorithm_yaml_defaults(name: str) -> DictConfig:
    path = _ALGORITHM_CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        return OmegaConf.create()
    return OmegaConf.load(path)


def _as_list(value: Iterable[object] | object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, ListConfig):
        return [
            OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
            for v in value
        ]
    return [
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    ]


def _to_python(value: object) -> object:
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


# ----------------------------------------------------------------------
# Optuna sampling helpers
# ----------------------------------------------------------------------
def _is_range_spec(value: object) -> bool:
    """Heuristic: Optuna range entries are dicts with a ``type`` field."""
    if isinstance(value, Mapping):
        return "type" in value
    if OmegaConf.is_config(value) and not isinstance(value, ListConfig):
        try:
            return "type" in value
        except Exception:
            return False
    return False


def _suggest_from_spec(
    trial: "optuna.Trial",
    name: str,
    spec: Mapping[str, Any] | DictConfig,
) -> object:
    """Translate a YAML spec into an ``optuna.Trial.suggest_*`` call."""
    if OmegaConf.is_config(spec):
        spec = OmegaConf.to_container(spec, resolve=True)  # type: ignore[assignment]
    assert isinstance(spec, Mapping)

    kind = str(spec["type"]).lower()
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            log=bool(spec.get("log", False)),
            step=int(spec.get("step", 1)),
        )
    if kind == "categorical":
        choices = list(spec["choices"])
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unknown range spec for {name!r}: {kind}")


# ----------------------------------------------------------------------
@dataclass
class SearchConfig:
    method: str
    max_epochs: int
    num_trials: int
    wandb_project: str | None
    wandb_group: str | None
    batched: DictConfig


class GridSearch:
    """Deterministic grid search across parameter grids."""

    def __init__(self, cfg: DictConfig, base_cfg: DictConfig) -> None:
        self.cfg = cfg
        self.base_cfg = base_cfg

    def _iter_trials(self) -> Iterator[dict[str, object]]:
        # Materialise grids — ``_grid`` returns one-shot iterators and nested
        # loops would exhaust inner iterators after the first outer step.
        shared_grid = list(_grid(_coerce_dict(getattr(self.cfg, "shared", None))))
        batched_grid = list(
            _grid(_coerce_dict(getattr(self.cfg, "batched_nfg_sarah", None))),
        )
        if "svrs" in self.cfg and self.cfg.svrs is not None:
            svrs_grid = list(_grid(_coerce_dict(self.cfg.svrs)))
        else:
            svrs_grid = [{}]

        for shared_params in shared_grid:
            for batched_params in batched_grid:
                for svrs_params in svrs_grid:
                    params: dict[str, object] = {}
                    params.update(_join_params("shared", shared_params))
                    params.update(_join_params("batched_nfg_sarah", batched_params))
                    params.update(_join_params("svrs", svrs_params))
                    yield params

    def run(
        self,
        evaluator: Callable[
            [int, dict[str, DictConfig], dict[str, object]],
            dict[str, object],
        ],
    ) -> dict[str, object]:
        best: dict[str, object] = {"score": -float("inf")}
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
            if score > float(best["score"]):
                best = {"score": score, "result": result}
        return best


class OptunaSearch:
    """Optuna-powered search across continuous parameter ranges with pruning."""

    def __init__(self, cfg: DictConfig, base_cfg: DictConfig) -> None:
        self.cfg = cfg
        self.base_cfg = base_cfg

    # ------------------------------------------------------------------
    def _build_sampler(self) -> "optuna.samplers.BaseSampler":
        import optuna

        sampler_cfg = OmegaConf.select(self.cfg, "sampler", default=None)
        kind = "tpe"
        seed: int | None = None
        if sampler_cfg is not None:
            kind = str(OmegaConf.select(sampler_cfg, "kind", default="tpe")).lower()
            seed_val = OmegaConf.select(sampler_cfg, "seed", default=None)
            if seed_val is not None:
                seed = int(seed_val)
        if kind == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        return optuna.samplers.TPESampler(seed=seed)

    def _build_pruner(self) -> "optuna.pruners.BasePruner":
        import optuna

        pruner_cfg = OmegaConf.select(self.cfg, "pruner", default=None)
        if pruner_cfg is None:
            return optuna.pruners.MedianPruner()
        kind = str(OmegaConf.select(pruner_cfg, "kind", default="median")).lower()
        if kind == "none":
            return optuna.pruners.NopPruner()
        if kind == "hyperband":
            return optuna.pruners.HyperbandPruner(
                min_resource=int(OmegaConf.select(pruner_cfg, "min_resource", default=1)),
                max_resource=int(
                    OmegaConf.select(
                        pruner_cfg, "max_resource", default=int(self.cfg.max_epochs),
                    )
                ),
            )
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(
                OmegaConf.select(pruner_cfg, "n_startup_trials", default=5),
            ),
            n_warmup_steps=int(
                OmegaConf.select(pruner_cfg, "n_warmup_epochs", default=5),
            ),
            interval_steps=int(
                OmegaConf.select(pruner_cfg, "interval_epochs", default=1),
            ),
        )

    # ------------------------------------------------------------------
    def run(
        self,
        evaluator: Callable[..., dict[str, object]],
    ) -> dict[str, object]:
        import optuna

        def objective(trial: "optuna.Trial") -> float:
            params = _sample_optuna_params(trial, self.cfg)
            trial_cfgs = _build_trial_cfgs(self.base_cfg, self.cfg, params)

            def _report(epoch: int, val_acc: float) -> bool:
                trial.report(val_acc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return False

            try:
                result = evaluator(
                    trial_id=trial.number,
                    trial_cfgs=trial_cfgs,
                    trial_params=params,
                    report_intermediate=_report,
                )
            except optuna.TrialPruned:
                raise

            score = _score_trial(result)
            trial.set_user_attr("result", result)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
        )
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
    max_epochs = int(search_cfg.max_epochs)
    base_algo = OmegaConf.create(
        OmegaConf.to_container(base_cfg.algorithm, resolve=True),
    )

    shared: dict[str, object] = {}
    batched_over: dict[str, object] = {}
    svrs_over: dict[str, object] = {}
    for key, value in params.items():
        v = _to_python(value)
        if key.startswith("shared."):
            shared[key.split(".", 1)[1]] = v
        elif key.startswith("batched_nfg_sarah."):
            batched_over[key.split(".", 1)[1]] = v
        elif key.startswith("svrs."):
            svrs_over[key.split(".", 1)[1]] = v

    batched_cfg = OmegaConf.merge(
        _algorithm_yaml_defaults("batched_nfg_sarah"), base_algo,
    )
    batched_cfg.name = "batched_nfg_sarah"
    for k, v in shared.items():
        batched_cfg[k] = v
    for k, v in batched_over.items():
        batched_cfg[k] = v
    batched_cfg.num_epochs = min(int(batched_cfg.num_epochs), max_epochs)

    out: dict[str, DictConfig] = {
        "batched_nfg_sarah": _override_algorithm(base_cfg, batched_cfg),
    }

    if "svrs" in search_cfg and search_cfg.svrs is not None:
        svrs_cfg = OmegaConf.merge(
            _algorithm_yaml_defaults("svrs"), OmegaConf.create(),
        )
        svrs_cfg.name = "svrs"
        for k, v in shared.items():
            svrs_cfg[k] = v
        for k, v in svrs_over.items():
            svrs_cfg[k] = v
        svrs_cfg.num_epochs = min(int(svrs_cfg.num_epochs), max_epochs)
        out["svrs"] = _override_algorithm(base_cfg, svrs_cfg)

    return out


def _sample_optuna_params(
    trial: "optuna.Trial",
    search_cfg: DictConfig,
) -> dict[str, object]:
    params: dict[str, object] = {}

    for name, value in _coerce_dict(getattr(search_cfg, "shared", None)).items():
        full_name = f"shared.{name}"
        if _is_range_spec(value):
            params[full_name] = _suggest_from_spec(trial, full_name, value)
        else:
            params[full_name] = trial.suggest_categorical(full_name, _as_list(value))

    for name, value in _coerce_dict(
        getattr(search_cfg, "batched_nfg_sarah", None),
    ).items():
        full_name = f"batched_nfg_sarah.{name}"
        if _is_range_spec(value):
            params[full_name] = _suggest_from_spec(trial, full_name, value)
        else:
            params[full_name] = trial.suggest_categorical(full_name, _as_list(value))

    for name, value in _coerce_dict(
        getattr(search_cfg, "svrs", None),
    ).items():
        full_name = f"svrs.{name}"
        if _is_range_spec(value):
            params[full_name] = _suggest_from_spec(trial, full_name, value)
        else:
            params[full_name] = trial.suggest_categorical(full_name, _as_list(value))

    return params


def _score_trial(result: dict[str, object]) -> float:
    summaries = result.get("summaries")
    if not isinstance(summaries, dict):
        return -float("inf")

    accs: list[float] = []
    for summary in summaries.values():
        if isinstance(summary, dict):
            acc = summary.get("best_val_accuracy", -float("inf"))
            accs.append(float(acc))

    if not accs or any(a == -float("inf") for a in accs):
        return -float("inf")

    return sum(accs) / len(accs)
