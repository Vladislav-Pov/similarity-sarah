# Similarity-SARAH: Distributed Optimization Simulation

A research codebase for simulating server–client distributed optimisation
algorithms under Hessian-similarity assumptions.  All clients are
simulated in a single process (no real networking / DDP).

---

## Repository structure

```
similarity-sarah/
├── configs/                         # Hydra YAML configs
│   ├── config.yaml                  #   top-level defaults
│   ├── algorithm/                   #   algorithm hyper-parameters
│   │   ├── batched_nfg_sarah.yaml
│   │   ├── distributed_sarah.yaml
│   │   └── svrs.yaml
│   ├── data/
│   │   ├── cifar10.yaml
│   │   └── synthetic.yaml
│   ├── model/
│   │   ├── simple_cnn.yaml
│   │   └── resnet18_32x32.yaml
│   ├── partition/
│   │   └── uniform.yaml
│   ├── runtime/
│   │   └── default.yaml
│   ├── search/
│   │   ├── none.yaml
│   │   ├── grid.yaml
│   │   └── optuna.yaml
│   └── experiment/                  #   pre-built experiment presets
│       ├── debug.yaml
│       └── debug_distributed_sarah.yaml
├── similarity_sarah/                # Python package
│   ├── algorithms/                  #   optimisation algorithms
│   │   ├── base.py                  #     BaseAlgorithm ABC
│   │   ├── batched_nfg_sarah.py     #     main algorithm
│   │   ├── distributed_sarah.py     #     baseline (full grad at epoch start)
│   │   └── svrs.py                  #     baseline scaffold (Khaled & Jin 2023)
│   ├── data/                        #   dataset loading & partitioning
│   ├── models/                      #   neural-network models (BN-free ResNet)
│   ├── runtime/                     #   runner, scheduler, prox solver, search
│   ├── tasks/                       #   loss / metrics definitions
│   └── utils.py                     #   parameter & gradient helpers
├── tests/                           # pytest tests
├── main.py                          # Hydra entry point
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Smoke tests
pytest tests/ -v

# 2-epoch debug run on CPU
python main.py +experiment=debug

# Distributed SARAH baseline (debug)
python main.py +experiment=debug_distributed_sarah
```

## Running a full CIFAR-10 experiment

```bash
# Batched No Full Grad SARAH (default config)
python main.py algorithm.num_epochs=100

# Override individual hyper-parameters on the command line
python main.py algorithm=batched_nfg_sarah \
    algorithm.num_clients=20 \
    algorithm.batch_size_clients=4 \
    algorithm.theta=0.05

# Distributed SARAH baseline
python main.py algorithm=distributed_sarah algorithm.lr=0.005

# SVRS baseline scaffold
python main.py algorithm=svrs algorithm.theta=0.05
```

Hydra writes logs and outputs to `outputs/<date>/<time>/`.

## Algorithms

| Key in config         | Class                       | Description                                                                                                  |
|-----------------------|-----------------------------|--------------------------------------------------------------------------------------------------------------|
| `batched_nfg_sarah`   | `BatchedNoFullGradSARAH`    | Main algorithm — no full-gradient computation, recursive SARAH estimators, proximal server updates.          |
| `distributed_sarah`   | `DistributedSARAH`          | Baseline — full gradient at epoch start, standard SARAH correction, plain gradient step.                     |
| `svrs`                | `SVRS`                      | Baseline scaffold for SVRS (similarity-based variance reduction, Khaled & Jin 2023, arXiv:2304.07504).       |

Adding a new algorithm:

1. Create `similarity_sarah/algorithms/my_algo.py` subclassing `BaseAlgorithm`.
2. Implement `initialize(...)` and `run_epoch(...)` (optionally `run_step`).
3. Add `configs/algorithm/my_algo.yaml`.
4. Register the algorithm in `Runner._setup_algorithm`.

## Hyper-parameter search

```bash
# Grid search
python main.py search=grid runtime.wandb.enabled=true

# Optuna search over continuous ranges + median pruning
python main.py search=optuna runtime.wandb.enabled=true
```

`configs/search/optuna.yaml` describes the search space using small
range specs:

```yaml
batched_nfg_sarah:
  theta:        {type: float, low: 1e-3, high: 1e-1, log: true}
  prox_lr:      {type: float, low: 1e-3, high: 2e-1, log: true}
  prox_num_steps: {type: int, low: 20, high: 200, log: true}
```

Trials report intermediate validation accuracy after every `eval_every`
epoch, so under-performing configurations are pruned early
(`MedianPruner`/`HyperbandPruner`).

## Proximal solver

The proximal step `prox_{θ f₁}(z)` is approximated by `InexactProxSGD`
or `InexactProxAdam` (a few SGD/Adam steps on the proximal objective).
Both solvers accept a momentum / weight-decay parameter and return a
small dictionary of diagnostics (gradient norm at the first / last
inner step, mean prox-loss, etc.) that the runner forwards to W&B.

To plug in a custom solver:

1. Subclass `ProxSolver` in `similarity_sarah/runtime/prox_solver.py`.
2. Select it from a config:
   `algorithm.prox_solver: my_solver` in
   `configs/algorithm/my_algo.yaml`, then route through
   `Runner._build_prox_solver`.

## Logging

Per-epoch, the runner forwards to W&B:

* `train/v_norm`, `train/tilde_v_norm` — SARAH estimator norms.
* `train/param_norm`, `train/step_norm_mean`, `train/step_norm_last`.
* `train/prox_grad_norm_first_mean`, `train/prox_grad_norm_last_mean`,
  `train/prox_grad_norm_reduction`, `train/prox_loss_mean` —
  diagnostics of the inner prox subproblem.
* `train/prox_lr`, `train/epoch_time_s`.
* `val/loss`, `val/accuracy`, `test/loss`, `test/accuracy`.

## Design notes

* **No real distribution** — all clients run in one process; "communication"
  is a function call.
* **GroupNorm in ResNet** — BatchNorm in train mode breaks `∇f_i(w)`
  determinism (running statistics depend on the batch).  GroupNorm has
  no running buffers and gives sample-wise gradients.
* **Server-side augmentation** — `data.augment_server: true` enables
  RandomCrop+HorizontalFlip *only* inside the server's prox loader.
  Client gradients stay deterministic.
* **Prox LR schedule** — `runtime.prox_lr_schedule.kind: cosine`
  (or `step`) lets the inner-prox step-size decay across outer epochs.

## Common command lines

```bash
python main.py runtime.wandb.enabled=true                 # default run
python main.py search=optuna runtime.wandb.enabled=true   # Optuna sweep
```
