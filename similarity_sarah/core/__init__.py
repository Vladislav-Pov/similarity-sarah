"""Hardware-agnostic numerical and runtime primitives for the rewrite.

This package holds the small, dependency-light building blocks that the rest
of ``similarity_sarah`` composes: reproducibility/seeding utilities today,
parameter/gradient/foreach math in later milestones.
"""

from similarity_sarah.core.repro import make_generator, seed_worker, set_seed

__all__ = ["make_generator", "seed_worker", "set_seed"]
