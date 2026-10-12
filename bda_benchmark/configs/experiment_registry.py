"""Canonical variant registries shared by models, datasets and launchers.

Keep this module free of torch/model imports.  Historically each layer kept
its own tuple, so adding a model variant could pass model tests but fail later
in the dataset or argparse contract.  New controlled families should define
their public variant names here and assert that implementation dictionaries
match these tuples.
"""
from __future__ import annotations


P4_NATIVE_SAR_VARIANTS = ("a0", "a1", "a2", "a3", "a4")

P3_SRAC_VARIANTS = (
    "sr0",
    "sr1",
    "sr1f",
    "sr1p",
    "sr1pa",
    "sr1pb",
    "sr1pai",
    "sr1pib",
    "sr1paa",
    "sr2",
)

HETEROGENEOUS_CORE_VARIANTS = ("h0", "h1", "h2", "h3", "h4")
EXTERNAL_CD_VARIANTS = ("x1", "x2")
OABH_VARIANTS = ("n0", "n1_shared", "n1_stage", "n1_cond", "n2")

HETEROGENEOUS_RGBSAR_VARIANTS = (
    *HETEROGENEOUS_CORE_VARIANTS,
    *EXTERNAL_CD_VARIANTS,
    *OABH_VARIANTS,
    *P3_SRAC_VARIANTS,
)


__all__ = [
    "EXTERNAL_CD_VARIANTS",
    "HETEROGENEOUS_CORE_VARIANTS",
    "HETEROGENEOUS_RGBSAR_VARIANTS",
    "OABH_VARIANTS",
    "P3_SRAC_VARIANTS",
    "P4_NATIVE_SAR_VARIANTS",
]

