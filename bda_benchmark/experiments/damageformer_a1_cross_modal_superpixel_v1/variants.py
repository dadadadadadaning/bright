"""Ablation contracts for the isolated A1 cross-modal superpixel study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    use_lca: bool
    partition: str
    sar_source: str = "none"
    use_alignment_confidence: bool = False
    shuffle_sar: bool = False
    use_damage_hint: bool = False
    use_boundary_loss: bool = False

    @property
    def use_regions(self) -> bool:
        return self.partition != "none"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


VARIANTS: Dict[str, VariantSpec] = {
    "c0_a1": VariantSpec(
        "c0_a1", "Frozen architectural reproduction of the A1 soft-ROI parent.",
        False, "none",
    ),
    "c1_lca": VariantSpec(
        "c1_lca", "A1 plus 3x3 ROI-constrained local RGB-SAR alignment.",
        True, "none", "aligned", True,
    ),
    "c2_slico": VariantSpec(
        "c2_slico", "A1-LCA plus fixed pre-disaster RGB SLICO regions.",
        True, "slico", "aligned", True,
    ),
    "c3_rgb_assoc": VariantSpec(
        "c3_rgb_assoc", "Learnable regions from RGB structure only.",
        True, "learnable", "none", False,
    ),
    "c4_raw_sar_assoc": VariantSpec(
        "c4_raw_sar_assoc", "Learnable regions from RGB and unaligned SAR evidence.",
        True, "learnable", "raw", False,
    ),
    "c5_aligned_assoc": VariantSpec(
        "c5_aligned_assoc", "Learnable regions from RGB and confidence-gated aligned SAR.",
        True, "learnable", "aligned", True,
    ),
    "c6_no_confidence": VariantSpec(
        "c6_no_confidence", "C5 without alignment-confidence gating.",
        True, "learnable", "aligned", False,
    ),
    "c7_shuffled_sar": VariantSpec(
        "c7_shuffled_sar", "C5 with spatially shifted SAR as a causal negative control.",
        True, "learnable", "aligned", True, True,
    ),
    "c8_damage_hint": VariantSpec(
        "c8_damage_hint", "C5 but leaks detached A1 class probabilities into partitioning.",
        True, "learnable", "aligned", True, False, True,
    ),
    "c9_boundary": VariantSpec(
        "c9_boundary", "C5 plus weak building-boundary affinity supervision.",
        True, "learnable", "aligned", True, False, False, True,
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(VARIANTS)}") from exc

