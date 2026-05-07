"""Compare NFG-SS / SVRS / FedAvg test accuracy on a single figure.

X-axis is the total number of transmitted client$\leftrightarrow$server
$\\mathbb{R}^d$ vectors (the cost measure used in the paper).  Per-epoch
vector budgets for our setup (1 server + 10 clients, B=1):

* NFG-SS  : 40 / epoch  (4 vectors per inner step, K=10 inner steps)
* SVRS    : 40 / epoch  (20 anchor refresh + 20 inner-loop expectation)
* FedAvg  : 20 / epoch  (2 vectors per round, 10 rounds)

Each run is fetched from W&B and clipped at the requested epoch count
so that all three end at the same total vector budget (here 2000).

Run:
    python scripts/plot_methods_comparison.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------------------------------------------------
# Configuration: one entry per method shown on the plot.
# ----------------------------------------------------------------------
@dataclass
class RunSpec:
    label: str
    url: str
    algo_namespace: str          # ``<algo>/test/accuracy`` key prefix
    vectors_per_epoch: int       # cost on x-axis
    max_epoch: int               # clip the run at this epoch (1-indexed)
    color: str
    marker: str


RUNS: list[RunSpec] = [
    RunSpec(
        label="NFG-SS (ours)",
        url="https://wandb-radfan.ru/chirkov/similarity-sarah/runs/93gffsfn",
        algo_namespace="batched_nfg_sarah",
        vectors_per_epoch=40,
        max_epoch=50,
        color="#d95f02",
        marker="s",
    ),
    RunSpec(
        label="SVRS",
        url="https://wandb-radfan.ru/chirkov/svrs/runs/apvzva47",
        algo_namespace="svrs",
        vectors_per_epoch=40,
        max_epoch=50,
        color="#1f78b4",
        marker="o",
    ),
    RunSpec(
        label="FedAvg",
        url="https://wandb-radfan.ru/chirkov/similarity-sarah/runs/pj2p1eyd",
        algo_namespace="fedavg",
        vectors_per_epoch=20,
        max_epoch=100,
        color="#7570b3",
        marker="^",
    ),
]

OUT_PNG = Path("figures/methods_comparison.png")
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_CSV = OUT_PNG.with_suffix(".csv")


# ----------------------------------------------------------------------
# W&B fetch helpers (same plumbing as scripts/plot_wandb_accuracy.py).
# ----------------------------------------------------------------------
def parse_run_url(url: str) -> tuple[str, str, str, str]:
    parsed = urlparse(url)
    host = f"{parsed.scheme}://{parsed.netloc}"
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 4 or parts[2] != "runs":
        raise ValueError(f"Unexpected W&B run URL shape: {url!r}")
    entity, project, _, run_id = parts[:4]
    return host, entity, project, run_id


def fetch_run_history(spec: RunSpec):
    host, entity, project, run_id = parse_run_url(spec.url)
    os.environ["WANDB_BASE_URL"] = host
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    hist = run.history(samples=100_000, pandas=True)
    print(
        f"[{spec.label}] {entity}/{project}/{run_id}: "
        f"{len(hist)} rows, columns sample: {list(hist.columns)[:6]}…"
    )
    return run, hist


def find_test_accuracy_column(hist: pd.DataFrame, algo_namespace: str) -> str:
    """Return the column holding the run's test accuracy series."""
    primary = f"{algo_namespace}/test/accuracy"
    if primary in hist.columns:
        return primary
    # Fallback: any column ending in /test/accuracy or just accuracy.
    pat = re.compile(rf"(^|/){re.escape(algo_namespace)}/test/accuracy$")
    for c in hist.columns:
        if pat.search(c):
            return c
    pat2 = re.compile(r"/test/accuracy$")
    for c in hist.columns:
        if pat2.search(c):
            return c
    raise SystemExit(
        f"No test/accuracy column found for {algo_namespace}. "
        f"Available: {sorted(hist.columns)}"
    )


def extract_curve(spec: RunSpec) -> pd.DataFrame:
    """Return a tidy DataFrame with columns: epoch, vectors, accuracy, label."""
    _, hist = fetch_run_history(spec)
    acc_col = find_test_accuracy_column(hist, spec.algo_namespace)

    # Use _step as 1-indexed epoch number (runner logs at step = epoch + 1).
    sub = hist[["_step", acc_col]].dropna().rename(
        columns={"_step": "epoch", acc_col: "accuracy"}
    )
    sub["epoch"] = sub["epoch"].astype(int)
    sub = sub.sort_values("epoch")

    # Clip to the requested epoch budget.
    sub = sub[sub["epoch"] <= spec.max_epoch].copy()

    sub["vectors"] = sub["epoch"] * spec.vectors_per_epoch
    sub["label"] = spec.label
    print(
        f"  → kept {len(sub)} pts (epoch ≤ {spec.max_epoch}); "
        f"x-range [0, {sub['vectors'].max()}], "
        f"acc range [{sub['accuracy'].min():.3f}, {sub['accuracy'].max():.3f}]"
    )
    return sub[["label", "epoch", "vectors", "accuracy"]]


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot(curves: dict[str, pd.DataFrame], specs: list[RunSpec], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    # ~10 evenly spaced markers per curve so the figure stays readable.
    target_markers = 10

    for spec in specs:
        df = curves[spec.label]
        if df.empty:
            print(f"[{spec.label}] empty after clipping; skipping.")
            continue
        markevery = max(1, len(df) // target_markers)
        ax.plot(
            df["vectors"], df["accuracy"],
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            markevery=markevery,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=1.8,
            alpha=0.95,
        )

    ax.set_xlabel("Transmitted vectors")
    ax.set_ylabel("Test accuracy")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.0, 0.85)

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.6",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"\nSaved figure → {out_png} (and {out_png.with_suffix('.pdf').name})")


def main() -> None:
    curves: dict[str, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []
    for spec in RUNS:
        df = extract_curve(spec)
        curves[spec.label] = df
        all_rows.append(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"Saved CSV    → {OUT_CSV}")

    plot(curves, RUNS, OUT_PNG)


if __name__ == "__main__":
    main()
