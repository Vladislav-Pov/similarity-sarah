# `experimental/` — not in the submission

Features that exist in the repository's history but are **out of scope** for the
NeurIPS submission per `docs/REWRITE_BRIEF.md`. They are kept here, off the hot
training path, so the main code stays focused on Algorithm 1.

Nothing in the main `similarity_sarah` package imports from `experimental/`.

| Feature | Why it is here | Migrated in |
|---------|----------------|-------------|
| `clip_number_of_clients_with_reshuffle` (NFG-SS ablation) | Ablation, not in paper | M5 |
| `update_v_tilde_in_the_end` (NFG-SS ablation) | Ablation, not in paper | M5 |
| `log_deviation` diagnostic | Diagnostic only, doubles epoch cost | M5 |
| `AccvrsBatchAdamProx` | Tried, no improvement | M3 |
| `distributed_sarah` | Not a comparison baseline | M5 |

> The table fills in as the rewrite milestones land; at M1 this package is an
> empty quarantine that establishes the boundary.
