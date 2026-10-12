from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    base_region_variant: str
    use_building_context: bool = True
    shuffle_building_context: bool = False
    freeze_cmsp: bool = False
    requires_c5_checkpoint: bool = False

    def to_dict(self):
        return asdict(self)


VARIANTS: Dict[str, VariantSpec] = {
    # Put the overall model first so the main direction is seen earliest.
    "full_joint": VariantSpec(
        "full_joint",
        "A1 + C5 aligned/confidence-gated learned regions + Building Context, jointly trained.",
        "c5_aligned_assoc",
    ),
    "bir_on_frozen_c5": VariantSpec(
        "bir_on_frozen_c5",
        "Load trained C5, freeze C5, train only Building Context.",
        "c5_aligned_assoc",
        freeze_cmsp=True,
        requires_c5_checkpoint=True,
    ),
    "bir_shuffle_on_frozen_c5": VariantSpec(
        "bir_shuffle_on_frozen_c5",
        "Frozen C5 + wrong-building context negative control.",
        "c5_aligned_assoc",
        shuffle_building_context=True,
        freeze_cmsp=True,
        requires_c5_checkpoint=True,
    ),
    "rgb_only_building": VariantSpec(
        "rgb_only_building",
        "C3 RGB-structure learned regions + Building Context.",
        "c3_rgb_assoc",
    ),
    "raw_sar_building": VariantSpec(
        "raw_sar_building",
        "C4 raw/unaligned-SAR regions + Building Context.",
        "c4_raw_sar_assoc",
    ),
    "no_conf_building": VariantSpec(
        "no_conf_building",
        "C6 aligned-SAR regions without confidence gate + Building Context.",
        "c6_no_confidence",
    ),
    "shifted_sar_building": VariantSpec(
        "shifted_sar_building",
        "C7 shifted-SAR negative control + Building Context.",
        "c7_shuffled_sar",
    ),
    "wrong_building_joint": VariantSpec(
        "wrong_building_joint",
        "Full C5 joint model but Building Context is deliberately assigned to the wrong building.",
        "c5_aligned_assoc",
        shuffle_building_context=True,
    ),
}


def get_variant(name: str) -> VariantSpec:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[name]
