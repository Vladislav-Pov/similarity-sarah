"""
Distributed SARAH training with Optuna tuning — analogous to main.py.

Usage:
    python main_distributed.py

Flags (same as main.py):
    RUN_TUNING = True   → Optuna hyperparameter search, then final train with best params
    RUN_TUNING = False  → single run with DEFAULT_PARAMS
"""

import random
import math

import optuna
import torch
import torchvision
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from main import (
    # Settings
    API_KEY, PROJECT_NAME, WORKSPACE, EXPERIMENT_NAME,
    USE_COMET, COMET_LOG_TUNING,
    DATA_ROOT, NUM_WORKERS, TEST_BATCH_SIZE, VAL_SPLIT, SEED,
    transform_train, transform_test,
    # Utility functions
    set_seed, maybe_create_experiment, log_metrics, finalize_params,
    build_model, build_criterions, build_testloader,
    get_lr, evaluate,
)
from distributed_sarah import (
    Server, create_clients, train_epoch_distributed, train_loop_distributed,
)

# ==========================================
# Distributed-specific settings
# ==========================================
RUN_TUNING = True
RUN_FINAL_TRAIN = True

TUNING_TRIALS = 100
TUNING_EPOCHS = 60
FINAL_EPOCHS = 200

NUM_CLIENTS = 10
NOFULLGRAD = False  # True = no-full-grad SARAH; False = standard (full grad each epoch)

DEFAULT_PARAMS = {
    "sarah_lr": 0.02,
    "weight_decay": 5e-4,
    "warmup_epochs": 10,
    "min_lr": 1e-4,
    "label_smoothing": 0.1,
    "max_grad_norm": 5.0,
    "batch_size": 256,
    "momentum": 0.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Data helpers
# ==========================================

def build_train_val_datasets(val_split, seed):
    """Split CIFAR-10 train into train/val Subsets (returns datasets, not loaders)."""
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train,
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_test,
    )
    num_train = len(train_dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator).tolist()
    split = int(num_train * (1 - val_split))

    train_subset = torch.utils.data.Subset(train_dataset, indices[:split])
    val_subset = torch.utils.data.Subset(val_dataset, indices[split:])

    return train_subset, val_subset


def build_full_train_dataset():
    """Full CIFAR-10 train dataset (no val split)."""
    return torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train,
    )


def build_val_loader(val_dataset, batch_size):
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


# ==========================================
# Optuna objective
# ==========================================

def objective(trial):
    params = {
        "sarah_lr": trial.suggest_float("sarah_lr", 1e-3, 0.2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 20),
        "min_lr_ratio": trial.suggest_float("min_lr_ratio", 1e-3, 1e-1, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 10.0),
        "momentum": trial.suggest_float("momentum", 0.0, 0.95),
    }
    params = finalize_params(params)
    params["batch_size"] = 256

    train_subset, val_subset = build_train_val_datasets(VAL_SPLIT, SEED)
    val_loader = build_val_loader(val_subset, params["batch_size"])

    model = build_model().to(device)
    criterion, criterion_sum = build_criterions(params["label_smoothing"])

    experiment = None
    if COMET_LOG_TUNING:
        experiment = maybe_create_experiment(
            f"{EXPERIMENT_NAME}-distributed-trial-{trial.number}"
        )

    best_val_acc = train_loop_distributed(
        train_dataset=train_subset,
        eval_loader=val_loader,
        model=model,
        criterion=criterion,
        criterion_sum=criterion_sum,
        params=params,
        total_epochs=TUNING_EPOCHS,
        num_clients=NUM_CLIENTS,
        device=device,
        eval_name="val",
        experiment=experiment,
        trial=trial,
        nofullgrad=NOFULLGRAD,
    )

    if experiment is not None:
        experiment.end()
    return best_val_acc


# ==========================================
# Final training with best params
# ==========================================

def train_with_params(params, total_epochs, run_name):
    train_dataset = build_full_train_dataset()
    test_loader = build_testloader()

    model = build_model().to(device)
    criterion, criterion_sum = build_criterions(params["label_smoothing"])

    experiment = maybe_create_experiment(run_name)

    best_test_acc = train_loop_distributed(
        train_dataset=train_dataset,
        eval_loader=test_loader,
        model=model,
        criterion=criterion,
        criterion_sum=criterion_sum,
        params=params,
        total_epochs=total_epochs,
        num_clients=NUM_CLIENTS,
        device=device,
        eval_name="test",
        experiment=experiment,
        nofullgrad=NOFULLGRAD,
    )

    if experiment is not None:
        experiment.end()
    return best_test_acc


# ==========================================
# Main
# ==========================================

def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"Num clients: {NUM_CLIENTS}")
    print(f"No-full-grad: {NOFULLGRAD}")

    if RUN_TUNING:
        sampler = TPESampler(seed=SEED)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=TUNING_TRIALS)

        best_params = finalize_params(study.best_trial.params)
        best_params["batch_size"] = 256
        print(f"Best val acc: {study.best_value:.2f}%")
        print(f"Best params: {best_params}")

        if RUN_FINAL_TRAIN:
            train_with_params(
                best_params, FINAL_EPOCHS, f"{EXPERIMENT_NAME}-distributed-best"
            )
    else:
        train_with_params(DEFAULT_PARAMS, FINAL_EPOCHS, f"{EXPERIMENT_NAME}-distributed")


if __name__ == "__main__":
    main()
