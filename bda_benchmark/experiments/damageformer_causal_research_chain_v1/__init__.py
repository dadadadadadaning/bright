"""Causal research chain for BRIGHT DamageFormer region/tail/expert/memory validation.

This package is intentionally isolated from historical experiments.  It consumes
immutable checkpoints/caches and writes every experiment into a distinct output
folder so that causal ablations cannot silently share optimizer/model state.
"""

PROTOCOL = "DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1"

__all__ = ["PROTOCOL"]
