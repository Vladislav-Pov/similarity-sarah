# `experimental/` — not in the submission

Features that exist in the repository's history but are **out of scope** for the
NeurIPS submission per `docs/REWRITE_BRIEF.md`. They are kept here, off the hot
training path, so the main code stays focused on Algorithm 1.

Nothing in the main `similarity_sarah` package imports from `experimental/`.
These modules are constructed manually (no registry / `from_spec`).

| File | What | Why it is out of scope |
|------|------|------------------------|
| `nfg_ss_ablations.py` | `NfgSSAblations(NFGSS)` with `update_v_tilde_in_the_end`, `clip_number_of_clients_with_reshuffle`, `log_deviation` | Ablations / diagnostic, not in the paper |
| `prox_accvrs_adam.py` | `AccvrsBatchAdam` prox solver (registers `accvrs_batch_adam` on import) | Tried during tuning, no improvement over `AccvrsBatchSGD` |

With all flags off, `NfgSSAblations` is bit-identical to `NFGSS`.

`distributed_sarah` was removed (not a comparison baseline; recoverable from
baseline commit `4ca4873` if needed).
