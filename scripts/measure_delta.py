"""Measure the second-order similarity coefficient δ (Definition 1) on CIFAR-10.

    Definition 1 (Second-order similarity).  The local objectives satisfy the
    δ-second-order-similarity condition if, for all x and all i ∈ [n],
        ‖∇² f_i(x) − ∇² f(x)‖ ≤ δ,
    where f = (1/n) Σ_i f_i is the global objective.

Computing the Hessian of a ResNet w.r.t. *all* its parameters is intractable
(d ~ 10⁷).  Following the standard practical proxy, we **freeze the ResNet
backbone, treat the 512-d penultimate features φ as fixed inputs, and take the
Hessian only w.r.t. the final linear head**.  With the backbone frozen the model
is multinomial logistic regression on φ, whose Hessian is *exact and
closed-form* (cross-entropy is convex in the logits and linear in the head
weights ⇒ Hessian equals the Gauss–Newton matrix, no approximation):

    per-sample Hessian w.r.t. vec(W):   ∇²ℓ = A(φ) ⊗ (φ φᵀ),
        A(φ) = diag(s) − s sᵀ  ∈ ℝ^{K×K}   (softmax Hessian, label-independent),
        s = softmax(W φ),   W ∈ ℝ^{K×p}   (head weights, bias folded in as a
                                            constant φ-coordinate = 1).

Client Hessian  H_i = (1/N_i) Σ_{φ∈D_i} A(φ) ⊗ φφᵀ  (evaluated at a shared
point W = x), global  H = Σ_i w_i H_i, and

    δ = max_i ‖H_i − H‖₂   (largest-magnitude eigenvalue of a symmetric Kp×Kp
                            matrix).

Because A is label-independent, heterogeneity enters through the *feature
distribution* per client (and, at a trained W, through s(φ)).  Under a
Dirichlet-over-labels split the label skew induces feature skew, so δ tracks α.

The absolute δ depends on the backbone, the evaluation point W, and any feature
preprocessing (PCA / standardisation); those are all fit **once on the pooled
data** and held fixed across α, so the *comparison of δ across Dirichlet α* — the
thing this script is for — is apples-to-apples.

Example
-------
    python scripts/measure_delta.py \
        --alphas 0.5 1 10 --num-clients 10 --server-fraction 0.5 \
        --checkpoint checkpoints_base_0.8/batched_nfg_sarah_final.pt \
        --embed-dim 64 --eval-point global --wandb

Run ``python scripts/measure_delta.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running as a plain script (``python scripts/measure_delta.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# NB: torchvision / omegaconf / repo modules are imported *lazily* inside the
# functions that use them, so the pure-math core of this file can be imported
# (and unit-tested) without the full training stack installed.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("measure_delta")


# ----------------------------------------------------------------------
# Backbone + embedding extraction
# ----------------------------------------------------------------------
def build_backbone(checkpoint: str | None, device: torch.device):
    """Return an eval-mode ResNet18_32x32, optionally loading trained weights."""
    from similarity_sarah.models.resnet import ResNet18_32x32

    model = ResNet18_32x32(num_classes=10).to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        state = payload.get("model_state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded backbone from %s (missing=%d, unexpected=%d keys)",
            checkpoint, len(missing), len(unexpected),
        )
    else:
        logger.warning(
            "No --checkpoint given: using a randomly initialised backbone. "
            "δ is a much weaker heterogeneity proxy on random features; pass a "
            "trained checkpoint for meaningful absolute values.",
        )
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Penultimate-layer features φ = body(embed(x)) for every sample, in order.

    Returns a CPU float32 tensor of shape (N, 512).  Row ``i`` corresponds to
    ``dataset[i]`` so the partition ``Subset.indices`` index straight into it.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats: list[torch.Tensor] = []
    for x, _ in loader:
        x = x.to(device)
        h = model.body(model.embed(x))  # (B, 512)
        feats.append(h.detach().cpu())
    emb = torch.cat(feats, dim=0).float()
    logger.info("Extracted embeddings: %s", tuple(emb.shape))
    return emb


# ----------------------------------------------------------------------
# Feature preprocessing (fit once on pooled data → comparable across α)
# ----------------------------------------------------------------------
def fit_preprocess(
    emb: torch.Tensor, embed_dim: int | None, standardize: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fit centering (+ optional PCA / standardisation) on the pooled embeddings.

    Returns ``(mean, components, scale)`` where ``components`` (d_out × 512) is
    ``None`` when no PCA is applied.  Apply with :func:`apply_preprocess`.
    """
    mean = emb.mean(dim=0)
    centered = emb - mean
    components: torch.Tensor | None = None
    if embed_dim and embed_dim < emb.shape[1]:
        # Top-``embed_dim`` principal directions via SVD of the centered matrix.
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[:embed_dim]  # (embed_dim, 512)
        projected = centered @ components.T
    else:
        projected = centered
    scale: torch.Tensor | None = None
    if standardize:
        scale = projected.std(dim=0).clamp_min(1e-6)
    logger.info(
        "Preprocess: mean-centered, PCA→%s, standardize=%s",
        components.shape[0] if components is not None else "off", standardize,
    )
    return mean, components, scale


def apply_preprocess(
    emb: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    """Center → (optional PCA project) → (optional standardise)."""
    out = emb - mean
    if components is not None:
        out = out @ components.T
    if scale is not None:
        out = out / scale
    return out


def augment_bias(phi: torch.Tensor) -> torch.Tensor:
    """Append a constant 1 coordinate so the linear head's bias is folded in."""
    ones = torch.ones(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)
    return torch.cat([phi, ones], dim=1)


# ----------------------------------------------------------------------
# Evaluation point W = x
# ----------------------------------------------------------------------
def train_global_head(
    phi_aug: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    steps: int = 300,
    weight_decay: float = 1e-4,
) -> torch.Tensor:
    """Fit multinomial logistic regression on the pooled (augmented) features.

    Returns W of shape (K, d) — the shared evaluation point x at which every
    client Hessian is measured.  Convex problem; a few hundred LBFGS steps
    reach the optimum.  The pooled data is the same for every α (partitioning
    only relabels indices), so this W is identical across α → clean comparison.
    """
    phi_aug = phi_aug.to(device)
    labels = labels.to(device)
    W = torch.zeros(num_classes, phi_aug.shape[1], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = phi_aug @ W.T
        loss = F.cross_entropy(logits, labels) + 0.5 * weight_decay * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = (phi_aug @ W.T).argmax(1).eq(labels).float().mean().item()
    logger.info("Global head trained: pooled train accuracy = %.4f", acc)
    return W.detach()


# ----------------------------------------------------------------------
# Head Hessian (exact, closed form)
# ----------------------------------------------------------------------
def softmax_at(phi_aug: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """s = softmax(W φ) per sample; uniform when W = 0."""
    return torch.softmax(phi_aug @ W.T, dim=1)


def client_hessian(
    phi_aug: torch.Tensor,
    W: torch.Tensor,
    num_classes: int,
    chunk: int = 4096,
) -> torch.Tensor:
    """H_i = (1/N) Σ_s A(φ_s) ⊗ φ_s φ_sᵀ, assembled as a dense (Kd × Kd) matrix.

    Block (k, l) = δ_{kl} M_k − N_{kl}, with
        M_k   = (1/N) Σ_s s_{s,k} φ_s φ_sᵀ,
        N_{kl}= (1/N) Σ_s s_{s,k} s_{s,l} φ_s φ_sᵀ,
    which is exactly ``diag(s) − s sᵀ`` (= A) Kronecker-multiplied by φφᵀ and
    averaged over the client's samples.  Accumulated in sample-chunks to bound
    memory.
    """
    device = phi_aug.device
    d = phi_aug.shape[1]
    K = num_classes
    N = phi_aug.shape[0]
    M = torch.zeros(K, d, d, device=device)
    Nm = torch.zeros(K, K, d, d, device=device)

    for start in range(0, N, chunk):
        pc = phi_aug[start : start + chunk]            # (c, d)
        sc = softmax_at(pc, W)                         # (c, K)
        for k in range(K):
            wk = sc[:, k : k + 1]                       # (c, 1)
            M[k] += (pc * wk).T @ pc                    # Φᵀ diag(s_k) Φ
        for k in range(K):
            for l in range(K):
                wkl = (sc[:, k] * sc[:, l]).unsqueeze(1)  # (c, 1)
                Nm[k, l] += (pc * wkl).T @ pc

    M /= N
    Nm /= N

    H = torch.zeros(K * d, K * d, device=device)
    for k in range(K):
        for l in range(K):
            block = -Nm[k, l]
            if k == l:
                block = block + M[k]
            H[k * d : (k + 1) * d, l * d : (l + 1) * d] = block
    # Symmetrise away tiny numerical asymmetry before the eigensolver.
    return 0.5 * (H + H.T)


def spectral_norm_sym(mat: torch.Tensor) -> float:
    """Largest-magnitude eigenvalue of a symmetric matrix (= ‖mat‖₂)."""
    eig = torch.linalg.eigvalsh(mat)
    return float(eig.abs().max().item())


# ----------------------------------------------------------------------
# δ for one Dirichlet α
# ----------------------------------------------------------------------
def measure_delta_for_alpha(
    *,
    alpha: float,
    dataset,
    emb: torch.Tensor,
    prep: tuple,
    W: torch.Tensor,
    num_clients: int,
    server_fraction: float | None,
    seed: int,
    num_classes: int,
    include_server: bool,
    weighting: str,
    device: torch.device,
) -> dict[str, float]:
    """Partition with Dirichlet(α), build per-node head Hessians, return δ stats."""
    from omegaconf import OmegaConf
    from similarity_sarah.data.partition import create_partition

    cfg = OmegaConf.create({
        "name": "dirichlet",
        "alpha": float(alpha),
        "server_fraction": server_fraction,
        "seed": int(seed),
    })
    partitions = create_partition(dataset, num_clients + 1, cfg)
    # partitions[0] = server, partitions[1:] = clients.
    node_subsets = partitions if include_server else partitions[1:]

    # Per-node augmented features (in the shared, preprocessed head space).
    node_phi: list[torch.Tensor] = []
    for sub in node_subsets:
        idx = torch.as_tensor(sub.indices, dtype=torch.long)
        phi = apply_preprocess(emb[idx].to(device), *prep)
        node_phi.append(augment_bias(phi))
    sizes = torch.tensor([p.shape[0] for p in node_phi], dtype=torch.float64)

    if weighting == "samples":
        weights = (sizes / sizes.sum()).to(device)
    else:  # uniform — matches Definition 1's f = (1/n) Σ f_i
        weights = torch.full((len(node_phi),), 1.0 / len(node_phi), device=device)

    # Pass 1: global (weighted-average) Hessian H.
    Kd = num_classes * node_phi[0].shape[1]
    H_bar = torch.zeros(Kd, Kd, device=device)
    for phi_aug, w in zip(node_phi, weights):
        H_bar += float(w) * client_hessian(phi_aug, W, num_classes)

    # Pass 2: per-node deviation ‖H_i − H‖₂.
    devs: list[float] = []
    for phi_aug in node_phi:
        H_i = client_hessian(phi_aug, W, num_classes)
        devs.append(spectral_norm_sym(H_i - H_bar))

    devs_t = torch.tensor(devs)
    return {
        "alpha": float(alpha),
        "delta_max": float(devs_t.max()),           # δ in Definition 1
        "delta_mean": float(devs_t.mean()),
        "delta_median": float(devs_t.median()),
        "H_norm": spectral_norm_sym(H_bar),         # scale reference
        "delta_rel": float(devs_t.max()) / (spectral_norm_sym(H_bar) + 1e-12),
        "num_nodes": float(len(node_phi)),
        "min_node_size": float(sizes.min()),
        "max_node_size": float(sizes.max()),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 10.0],
                   help="Dirichlet concentrations to sweep.")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--server-fraction", type=float, default=0.5,
                   help="IID share to the server before the Dirichlet split "
                        "(null → equal 1/(clients+1) slice).")
    p.add_argument("--include-server", action="store_true",
                   help="Also count the server partition as one of the objectives "
                        "i∈[n] (default: clients only, matching f_1 = server).")
    p.add_argument("--weighting", choices=["uniform", "samples"], default="uniform",
                   help="How the global Hessian H averages the nodes.")
    p.add_argument("--eval-point", choices=["global", "zeros"], default="global",
                   help="x at which Hessians are measured: 'global' trains a "
                        "shared logistic head on pooled features; 'zeros' uses "
                        "W=0 (uniform softmax, instant, feature-covariance proxy).")
    p.add_argument("--embed-dim", type=int, default=64,
                   help="PCA-reduce features to this dim (0 = keep full 512; "
                        "raises the Hessian to 512·K per side).")
    p.add_argument("--standardize", action="store_true",
                   help="Divide features by their pooled per-dim std after PCA.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="ResNet18_32x32 checkpoint (.pt) for the frozen backbone.")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0,
                   help="Fixes the Dirichlet draw so α is the only variable.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--csv", type=str, default=None, help="Optional path to save results as CSV.")
    p.add_argument("--wandb", action="store_true", help="Log the δ-vs-α sweep to W&B.")
    p.add_argument("--wandb-project", type=str, default="similarity-sarah-delta")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    logger.info("Device: %s", device)

    import torchvision
    from similarity_sarah.data.datasets import _cifar10_eval_transform

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=_cifar10_eval_transform(),
    )
    labels = torch.as_tensor(np.asarray(dataset.targets), dtype=torch.long)
    num_classes = int(labels.max().item()) + 1

    model = build_backbone(args.checkpoint, device)
    emb = extract_embeddings(model, dataset, device, args.batch_size)

    embed_dim = None if not args.embed_dim else int(args.embed_dim)
    prep = fit_preprocess(emb, embed_dim, args.standardize)

    # Shared evaluation point W = x (identical across α).
    if args.eval_point == "global":
        phi_all = augment_bias(apply_preprocess(emb.to(device), *prep))
        W = train_global_head(phi_all, labels, num_classes, device)
        del phi_all
    else:
        d = (embed_dim or emb.shape[1]) + 1
        W = torch.zeros(num_classes, d, device=device)
        logger.info("Evaluation point: W = 0 (uniform softmax).")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    results: list[dict[str, float]] = []
    for alpha in args.alphas:
        res = measure_delta_for_alpha(
            alpha=alpha, dataset=dataset, emb=emb, prep=prep, W=W,
            num_clients=args.num_clients, server_fraction=args.server_fraction,
            seed=args.seed, num_classes=num_classes,
            include_server=args.include_server, weighting=args.weighting,
            device=device,
        )
        results.append(res)
        logger.info(
            "alpha=%-6.3g  delta(max)=%.4e  delta(mean)=%.4e  ‖H‖=%.4e  "
            "delta/‖H‖=%.4f",
            res["alpha"], res["delta_max"], res["delta_mean"],
            res["H_norm"], res["delta_rel"],
        )
        if wandb_run is not None:
            wandb_run.log(res)

    # ── Summary table ────────────────────────────────────────────────
    print("\n=== Second-order similarity δ vs Dirichlet α ===")
    print(f"{'alpha':>8} | {'delta_max':>12} | {'delta_mean':>12} | "
          f"{'||H||':>12} | {'delta/||H||':>11}")
    print("-" * 66)
    for r in results:
        print(f"{r['alpha']:>8.3g} | {r['delta_max']:>12.4e} | "
              f"{r['delta_mean']:>12.4e} | {r['H_norm']:>12.4e} | "
              f"{r['delta_rel']:>11.4f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved results → %s", args.csv)

    if wandb_run is not None:
        import wandb
        table = wandb.Table(columns=list(results[0].keys()),
                            data=[list(r.values()) for r in results])
        wandb_run.log({"delta_vs_alpha": table})
        wandb_run.finish()


if __name__ == "__main__":
    main()
