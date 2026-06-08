# similarity-sarah — NFG-SS distributed optimization

Research codebase for **NFG-SS** (*NoFullGrad SARAH Similarity*): distributed,
non-convex optimization under second-order similarity that **never computes a
full gradient**. All clients are simulated in one process (no real networking);
"communication" is a function call. See `docs/paper.pdf` for the algorithm and
theory.

This README is the entry point — read it top to bottom and you can run
everything.

---

## 1. Install

```bash
# 1) PyTorch (pick the build for your CUDA — see https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) The rest of the runtime deps
pip install -r requirements.txt

# 3) (optional) dev tools — lint / type-check / test
pip install -r requirements-dev.txt
```

> `requirements.txt` does **not** pin a CUDA build of torch (PyPI can't serve
> `+cuNN` wheels); install torch first from the PyTorch index, then the rest.

A quick check that everything imports and the algorithm is bit-correct:

```bash
python -m pytest          # 52 tests, ~1 min on CPU; no GPU/CIFAR needed
```

---

## 2. Quick start

**Fast smoke run** (synthetic data, CPU, no download, ~5 s) — proves the whole
stack works end to end:

```bash
python main.py algorithm=best data=synthetic model=simple_cnn \
  algorithm.num_epochs=1 algorithm.num_clients=4 \
  runtime.device=cpu runtime.num_workers=0 runtime.wandb.enabled=false
```

**Main NFG-SS run** (CIFAR-10 / ResNet-18 / 11 nodes — the paper setup):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py algorithm=best seed=0
```

`runtime.device` defaults to `auto` (CUDA if visible, else CPU), so on a GPU
server you only set `CUDA_VISIBLE_DEVICES`. The default config logs to W&B —
either `wandb login` first, or append `runtime.wandb.enabled=false`.

Hydra writes logs/outputs to `outputs/<date>/<time>/`.

---

## 3. The 30-second mental model

```
main.py (Hydra)
  └─ parse_run_spec(cfg)            configs/*.yaml  ->  frozen RunSpec  (spec.py)
       ├─ search enabled?  ── yes ─> run_search        (runtime/search_runner.py)
       └─ no ─> run_experiment(spec)                   (runtime/loop.py)
                  ├─ build_federated_data(spec)        (data/loaders.py)
                  ├─ build_model(...)                  (models/__init__.py)
                  ├─ ALGORITHMS.get(name).from_spec()  (registry -> algorithms/)
                  └─ TrainLoop(...).run()              epoch loop + eval + log
```

Every hyperparameter is a Hydra override on the command line:
`algorithm.theta=0.3`, `algorithm.prox_num_steps=5`, `runtime.batch_size=256`,
`seed=1`, …

---

## 4. Running experiments

```bash
# NFG-SS, best-known config, three seeds
CUDA_VISIBLE_DEVICES=0 python main.py algorithm=best seed=0
CUDA_VISIBLE_DEVICES=0 python main.py algorithm=best seed=1
CUDA_VISIBLE_DEVICES=0 python main.py algorithm=best seed=2

# Baselines (same data/partition)
CUDA_VISIBLE_DEVICES=0 python main.py algorithm=svrs   seed=0
CUDA_VISIBLE_DEVICES=0 python main.py algorithm=fedavg seed=0

# Override any hyperparameter
python main.py algorithm=best algorithm.theta=0.3 algorithm.prox_num_steps=5

# Turn W&B off / point it elsewhere
python main.py algorithm=best runtime.wandb.enabled=false
python main.py algorithm=best runtime.wandb.project=my-proj runtime.wandb.entity=me
```

---

## 5. Configuration (Hydra)

Configs live in `configs/`, composed by group. The top-level defaults are in
`configs/config.yaml`; override a whole group with `group=name` or a single key
with `group.key=value`.

| Group | Pick with | Options |
|-------|-----------|---------|
| `algorithm` | `algorithm=best` | `best` (winning NFG-SS), `batched_nfg_sarah`, `svrs`, `fedavg` |
| `data` | `data=cifar10` | `cifar10`, `synthetic` |
| `model` | `model=resnet18_32x32` | `resnet18_32x32`, `simple_cnn` |
| `partition` | `partition=uniform` | `uniform` (`server_fraction` knob) |
| `runtime` | `runtime=default` | batch sizes, device, workers, W&B, prox-lr schedule |
| `search` | `search=optuna_bnfg` | `none`, `grid*`, `optuna*` |

`algorithm=best` is the recommended NFG-SS config. `algorithm=batched_nfg_sarah`
is an alias that resolves to the same algorithm class (`NFGSS`).

---

## 6. Reproducing the reference runs

`docs/reference_runs/*.json` are the best-known config snapshots (A100,
CIFAR-10 / ResNet-18 / 11 nodes). `configs/algorithm/best.yaml` encodes the
winning formula. The three snapshots differ **only** in `prox_num_steps`
(`prox_v_schedule` is a no-op on the AccVRS path — see §8), so the distinct
behaviours are `num_steps ∈ {4, 5}`:

```bash
CUDA_VISIBLE_DEVICES=0 SEEDS="0 1 2" ./scripts/reproduce_reference.sh
```

To check **bit-reproducibility** against the pre-rewrite code, run the same
config on baseline commit `4ca4873` and on the current branch and compare the
final metrics/weights. The local CI gate already proves bit-identity on a fast
synthetic config (`tests/test_golden.py`, `tests/test_nfg_ss.py`).

---

## 7. Hyperparameter search (Optuna / grid)

```bash
python main.py search=optuna_bnfg   runtime.wandb.enabled=true   # NFG-SS, TPE + pruning
python main.py search=optuna_svrs   runtime.wandb.enabled=true
python main.py search=grid_bnfg     runtime.wandb.enabled=true
```

Search spaces are small range specs in `configs/search/*.yaml`:

```yaml
batched_nfg_sarah:
  theta:          {type: float, low: 1e-3, high: 1e-1, log: true}
  prox_num_steps: {type: int,   low: 4,    high: 6}
```

Trials report validation accuracy every `eval_every` epochs, so weak configs
are pruned early (`MedianPruner`/`HyperbandPruner`).

---

## 8. Algorithms & solvers

| `algorithm.name` | Class | Notes |
|------------------|-------|-------|
| `nfg_ss` (`batched_nfg_sarah`) | `NFGSS` | the method — no full gradient, SARAH telescope, prox on `f1` |
| `svrs` | `SVRS` | baseline; full-gradient anchor each epoch (Lin et al. 2023) |
| `fedavg` | `FedAvg` | communication-matched baseline (McMahan et al. 2017) |

**Inexact prox solvers** (`algorithm.prox_solver`): `accvrs_batch_sgd` (best,
the default in `best.yaml`), `sgd`, `adam`.

- **`accvrs_batch_sgd`** auto-derives the inner LR: with `prox_lr: null`,
  `gamma0 = (1/(2L))·prox_lr_factor`, `L = 1 + theta·prox_L1`. `prox_num_steps`
  counts **full passes over the server loader**, not iterations.
- **`prox_v_schedule` (`constant`/`linear`) is a no-op on the AccVRS path** —
  `v` enters only via the warm-start `z = w − theta·v`. It *does* take effect
  for the `sgd`/`adam` solvers.

### Add a new baseline = two files

1. `similarity_sarah/algorithms/<name>.py` — subclass `Algorithm`, decorate with
   `@ALGORITHMS.register("<name>")`, implement `bind`, `run_epoch`, `from_spec`.
2. `configs/algorithm/<name>.yaml` — its hyperparameters.

Then add one import line to `similarity_sarah/algorithms/__init__.py` so it
self-registers. No edits to the runner/loop/loaders.

---

## 9. Implementation invariants (must-know)

These are load-bearing — `tests/` guards each:

- **SARAH increment uses `1/(n·B)`** (`algorithms/nfg_ss.py`, `coeff_v`). This
  deliberately diverges from the paper's Algorithm 1 line 9 (`1/b`) so the
  published reference runs reproduce exactly. **Do not change it.**
- **Same minibatch at `w_t` and `w_{t-1}`** within an inner step — otherwise the
  SARAH telescope degenerates into noisy SGD (`docs/instructions.md` §11).
- **`server_grad_loader` is deterministic** (no augmentation); server-side
  augmentation is out of scope for the submission.
- **ResNet is BatchNorm-free** (GroupNorm) so `∇f_i(w)` is a deterministic
  function of `w`.
- **Bit-reproducibility**: a fixed `seed` ⇒ a fixed result; set
  `runtime.deterministic=true` to also pin PyTorch deterministic kernels and
  seed the data loaders.

---

## 10. Project layout

```
similarity_sarah/
├── spec.py             configs -> frozen RunSpec dataclasses
├── registry.py         name -> class registry (the extensibility seam)
├── core/               repro (seeding), params, grads, foreach (fused math)
├── algorithms/         base (Algorithm/AlgorithmCtx/ALGORITHMS), nfg_ss, svrs, fedavg
├── prox/               base (+ build_prox_solver), inexact (sgd/adam), accvrs, _diag
├── data/               datasets, partition, loaders (build_federated_data)
├── models/             resnet18_32x32 (GroupNorm), simple_cnn, build_model
├── runtime/            loop (TrainLoop, run_experiment), metrics, scheduler,
│                       search + search_runner
├── tasks/              classification (loss + eval)
└── experimental/       OUT OF SCOPE (ablations, AccVRS-Adam) — see its README
configs/   Hydra groups        main.py   entry point        tests/   pytest suite
docs/      paper, instructions, reference_runs/   scripts/  run.sh, tune.sh, reproduce_reference.sh
```

---

## 11. Tests & dev

```bash
python -m pytest                                   # 52 tests (bit-repro, invariants, pipeline)
python -m ruff check similarity_sarah main.py tests
python -m mypy                                     # type-checks the rewritten modules
```

CI (`.github/workflows/ci.yml`) runs all three on CPU, no training.

---

## 12. Logging

When W&B is enabled, per epoch the loop logs under `<algo>/{train,val,test}/<metric>`:
`v_norm`, `tilde_v_norm`, `param_norm`, `step_norm_mean`, the `prox_grad_norm_*`
diagnostics, `epoch_time_s`, and `val/test` `loss`/`accuracy`.
