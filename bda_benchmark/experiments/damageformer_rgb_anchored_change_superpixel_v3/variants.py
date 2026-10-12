"""Causal variants for RGB-anchored, RGB-SAR-refined superpixels.

P0/P1 are audit-only fixed Grid/SLICO baselines.  P2 and P5-P8 share the
same trainable architecture so that every comparison changes one mechanism.
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
    use_alignment_confidence: bool = True
    use_damage_boundary: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


VARIANTS: Dict[str, VariantSpec] = {
    "p2_rgb_structure": VariantSpec(
        "p2_rgb_structure",
        "Learned RGB structural partition; cross-modal evidence cannot change geometry.",
        use_change_refinement=False,
        use_damage_boundary=False,
    ),
    "p5_anchored_change": VariantSpec(
        "p5_anchored_change",
        "Main method: RGB structural logits plus bounded aligned RGB-SAR change residual.",
    ),
    "p6_shuffled_sar": VariantSpec(
        "p6_shuffled_sar",
        "P5 with spatially shifted SAR; causal negative control.",
        shuffle_sar=True,
    ),
    "p7_no_confidence": VariantSpec(
        "p7_no_confidence",
        "P5 with alignment confidence/entropy replaced by uninformative constants.",
        use_alignment_confidence=False,
    ),
    "p8_no_damage_boundary": VariantSpec(
        "p8_no_damage_boundary",
        "P5 without occurrence-boundary and semantic reconstruction supervision.",
        use_damage_boundary=False,
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown V3 variant {name!r}; choose from {sorted(VARIANTS)}") from exc


__all__ = ["VariantSpec", "VARIANTS", "get_variant"]
