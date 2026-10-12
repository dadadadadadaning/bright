"""Variant contracts for Building-Internal Change-Aware Partition V2.

The ablations separate *where regions come from* from *what evidence a region
contains*.  All variants keep the same predicted-building hard constraint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    use_structure_logits: bool = True
    use_change_refinement: bool = True
    use_soft_peak: bool = True
    use_std: bool = True
    use_bag_supervision: bool = True
    use_uncertainty_feedback: bool = True
    shuffle_sar: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


VARIANTS: Dict[str, VariantSpec] = {
    "p_s_struct_only": VariantSpec(
        name="p_s_struct_only",
        description=(
            "Predicted-building constrained adaptive regions with RGB structural "
            "association only. Cross-modal evidence is still encoded inside tokens, "
            "but it cannot alter region geometry."
        ),
        use_change_refinement=False,
    ),
    "p_c_change_only": VariantSpec(
        name="p_c_change_only",
        description=(
            "Predicted-building constrained change-driven association with the RGB "
            "structural logits removed. This is a risk/control analogue of V1."
        ),
        use_structure_logits=False,
        use_change_refinement=True,
    ),
    "p_sc_residual": VariantSpec(
        name="p_sc_residual",
        description=(
            "Main method: RGB structural base association plus zero-initialised "
            "RGB-SAR change residual refinement inside predicted buildings."
        ),
    ),
    "p_sc_no_peak": VariantSpec(
        name="p_sc_no_peak",
        description="Main geometry but region evidence token uses mean/std without soft-peak pooling.",
        use_soft_peak=False,
    ),
    "p_sc_mean_only": VariantSpec(
        name="p_sc_mean_only",
        description="Main geometry but region evidence token uses mean pooling only.",
        use_soft_peak=False,
        use_std=False,
    ),
    "p_sc_no_bag": VariantSpec(
        name="p_sc_no_bag",
        description="Main geometry/evidence representation without weak building-level task supervision.",
        use_bag_supervision=False,
    ),
    "p_sc_no_uncertainty": VariantSpec(
        name="p_sc_no_uncertainty",
        description="Main method but Stage-2 pixel feedback omits the frozen A1 uncertainty gate.",
        use_uncertainty_feedback=False,
    ),
    "p_sc_shuffled_sar": VariantSpec(
        name="p_sc_shuffled_sar",
        description="Main method with spatially shifted SAR evidence as a causal negative control.",
        shuffle_sar=True,
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(VARIANTS)}") from exc


__all__ = ["VariantSpec", "VARIANTS", "get_variant"]
