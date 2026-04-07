# Similarity-SARAH: Distributed Optimization Simulation

A research codebase for simulating server–client distributed optimisation
algorithms under Hessian-similarity assumptions.  All clients are simulated
in a single process (no real networking / DDP).

---

## Repository structure

```
similarity-sarah/
├── configs/                         # Hydra YAML configs
│   ├── config.yaml                  #   top-level defaults
│   ├── algorithm/                   #   algorithm hyper-parameters
│   │   ├── batched_nfg_sarah.yaml
│   │   └── distributed_sarah.yaml
│   ├── data/
│   │   └── cifar10.yaml
│   ├── model/
│   │   └── simple_cnn.yaml
│   ├── partition/
│   │   └── uniform.yaml
│   ├── runtime/
│   │   └── default.yaml
│   └── experiment/                  #   pre-built experiment presets
│       ├── debug.yaml
│       └── debug_distributed_sarah.yaml
├── similarity_sarah/                # Python package
│   ├── algorithms/                  #   optimisation algorithms
│   │   ├── base.py                  #     BaseAlgorithm ABC
│   │   ├── batched_nfg_sarah.py     #     main algorithm
│   │   └── distributed_sarah.py     #     baseline
│   ├── data/                        #   dataset loading & partitioning
│   │   ├── datasets.py
│   │   └── partition.py
│   ├── models/                      #   neural-network models
│   │   └── simple_cnn.py
│   ├── runtime/                     #   runner, scheduler, prox solver
│   │   ├── runner.py
│   │   ├── scheduler.py
│   │   └── prox_solver.py
│   ├── tasks/                       #   loss / metrics definitions
│   │   └── classification.py
│   └── utils.py                     #   parameter & gradient helpers
├── tests/                           # pytest tests
│   ├── test_scheduler.py
│   ├── test_partition.py
│   └── test_utils.py
├── main.py                          # Hydra entry point
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Run the debug experiment (small config, CPU, 2 epochs)
python main.py +experiment=debug

# 4. Run the Distributed SARAH baseline debug
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
    algorithm.theta=0.005

# Distributed SARAH baseline
python main.py algorithm=distributed_sarah algorithm.lr=0.005
```

Hydra writes logs and outputs to `outputs/<date>/<time>/`.

## Algorithms

| Key in config              | Class                              | Description |
|----------------------------|------------------------------------|-------------|
| `batched_nfg_sarah`       | `BatchedNoFullGradSARAH`          | Main algorithm — no full-gradient computation, recursive SARAH estimators, proximal server updates |
| `distributed_sarah`       | `DistributedSARAH`                | Baseline — full gradient at epoch start, standard SARAH correction, plain gradient step |

### Adding a new algorithm

1. Create `similarity_sarah/algorithms/my_algo.py` subclassing `BaseAlgorithm`.
2. Implement `initialize(...)` and `run_epoch(...)`.
3. Add a config file `configs/algorithm/my_algo.yaml`.
4. Register the algorithm in `Runner._setup_algorithm()` (`similarity_sarah/runtime/runner.py`).

## Datasets

Currently CIFAR-10.  To add a new dataset:

1. Add a loading branch in `similarity_sarah/data/datasets.py` → `load_dataset()`.
2. Create `configs/data/my_dataset.yaml`.
3. If the transform / image size changes, add a matching model config.

## Proximal solver

The proximal step `prox_{θf₁}(z)` is approximated by `InexactProxSGD`
(a few SGD steps on the proximal objective).  To swap in a different solver:

1. Subclass `ProxSolver` in `similarity_sarah/runtime/prox_solver.py`.
2. Instantiate it in `Runner._setup_algorithm()`.

## Design notes

- **No real distribution** — all clients run in one process; "communication"
  is a function call.
- **No random augmentation** during gradient computation — the exact empirical
  gradient is required for the SARAH estimators.
- **No BatchNorm** in the default CNN — avoids stochastic forward passes that
  would break deterministic gradient computation.
