"""Fetch a single W&B run and plot test/val accuracy vs. epoch.

Usage:
    python scripts/plot_wandb_accuracy.py \
        --run-url https://wandb-radfan.ru/chirkov/similarity-sarah/runs/93gffsfn \
        --out figures/run_93gffsfn_accuracy.png

The script auto-detects the W&B host from the URL (custom hosts like
``wandb-radfan.ru`` need ``WANDB_BASE_URL`` set), pulls the run history,
discovers all ``*/test/accuracy`` and ``*/val/accuracy`` series, and
plots them on one figure.  Raw data is saved alongside the figure as a
CSV for later use (e.g. importing into the paper's TikZ plot).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlparse


def parse_run_url(url: str) -> tuple[str, str, str, str]:
    """Return (host_url, entity, project, run_id) from a W&B run URL.

    Accepts URLs of the form
        https://<host>/<entity>/<project>/runs/<run_id>[/...]
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Not a URL: {url!r}")
    host = f"{parsed.scheme}://{parsed.netloc}"
    parts = [p for p in parsed.path.split("/") if p]
    # Expect [entity, project, "runs", run_id, ...]
    if len(parts) < 4 or parts[2] != "runs":
        raise ValueError(
            f"Unexpected W&B run URL shape: {url!r} "
            "(expected /<entity>/<project>/runs/<run_id>)"
        )
    entity, project, _, run_id = parts[:4]
    return host, entity, project, run_id


def fetch_run_history(host: str, entity: str, project: str, run_id: str):
    """Pull the run history as a pandas DataFrame.

    Sets ``WANDB_BASE_URL`` so ``wandb.Api()`` talks to the right host
    (default https://api.wandb.ai is wrong for self-hosted servers).
    """
    os.environ["WANDB_BASE_URL"] = host
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"Run: {run.name}  (state={run.state}, created={run.created_at})")
    print(f"Tags: {run.tags}")
    # ``samples=100000`` defeats W&B's default 500-point downsampling.
    hist = run.history(samples=100_000, pandas=True)
    print(f"History: {len(hist)} rows, {len(hist.columns)} columns")
    return run, hist


def find_accuracy_columns(hist) -> list[str]:
    """Return columns that look like accuracy series.

    Matches both ``<algo>/<split>/accuracy`` (the runner's namespacing)
    and bare ``accuracy``.
    """
    pat = re.compile(r"(^|/)accuracy$", re.IGNORECASE)
    cols = [c for c in hist.columns if pat.search(c) and not c.startswith("_")]
    return sorted(cols)


def plot(hist, acc_cols, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # x-axis: prefer an explicit ``epoch`` column if logged; else _step.
    x_col = "epoch" if "epoch" in hist.columns else "_step"

    for col in acc_cols:
        sub = hist[[x_col, col]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col], sub[col], marker=".", linewidth=1.4, label=col)

    ax.set_xlabel("epoch" if x_col == "epoch" else "step")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved figure → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-url",
        required=True,
        help="Full W&B run URL, e.g. "
        "https://wandb-radfan.ru/chirkov/similarity-sarah/runs/93gffsfn",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to figures/<run_id>_accuracy.png.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional whitelist of accuracy columns to plot "
        "(e.g. batched_nfg_sarah/test/accuracy svrs/test/accuracy). "
        "By default all detected accuracy series are plotted.",
    )
    args = parser.parse_args()

    host, entity, project, run_id = parse_run_url(args.run_url)
    out_png = Path(args.out) if args.out else Path(f"figures/{run_id}_accuracy.png")

    run, hist = fetch_run_history(host, entity, project, run_id)

    acc_cols = find_accuracy_columns(hist)
    if not acc_cols:
        raise SystemExit(
            "No accuracy columns found in run history. "
            f"Available columns: {sorted(hist.columns)}"
        )
    print(f"Found accuracy columns: {acc_cols}")

    if args.include:
        keep = set(args.include)
        acc_cols = [c for c in acc_cols if c in keep]
        if not acc_cols:
            raise SystemExit(
                f"None of --include columns matched. Available: {sorted(hist.columns)}"
            )

    # Save the slice we plotted as CSV for downstream use (TikZ, paper).
    out_png.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_png.with_suffix(".csv")
    x_col = "epoch" if "epoch" in hist.columns else "_step"
    cols_to_save = [x_col] + acc_cols
    hist[cols_to_save].to_csv(csv_path, index=False)
    print(f"Saved CSV    → {csv_path}")

    title = f"{run.name}  ({entity}/{project}/{run_id})"
    plot(hist, acc_cols, out_png, title=title)


if __name__ == "__main__":
    main()
