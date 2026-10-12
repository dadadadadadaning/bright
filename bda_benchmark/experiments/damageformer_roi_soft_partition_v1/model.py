"""Single-method model: A1 soft RGB-building ROI + cross-modal soft partition.

No ablations, Building Reasoning, Graph, MIL, Memory, or Morphology are added.
The low-level partition implementation is reused from the already validated
``damageformer_a1_cross_modal_superpixel_v1`` experiment and locked to C5.
"""
from __future__ import annotations
from typing import Dict

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import (
    DamageFormerA1CrossModalSuperpixel,
)

FIXED_VARIANT = "c5_aligned_assoc"


class DamageFormerROISoftPartition(DamageFormerA1CrossModalSuperpixel):
    """A1 soft ROI + one fixed cross-modal damage-aware soft partition."""

    def __init__(self, region_stride: int = 8) -> None:
        super().__init__(variant=FIXED_VARIANT, region_stride=region_stride)

    def checkpoint_contract(self) -> Dict[str, object]:
        contract = dict(super().checkpoint_contract())
        contract.update({
            "protocol": "DAMAGEFORMER_ROI_SOFT_PARTITION_V1",
            "top_level_modules": [
                "A1_soft_RGB_building_ROI",
                "cross_modal_damage_aware_soft_partition",
            ],
            "fixed_variant": FIXED_VARIANT,
            "partition_is_color_slico": False,
            "partition_driver": [
                "RGB_encoder_structure",
                "RGB_structural_edge",
                "predicted_RGB_building_probability",
                "locally_aligned_SAR_relation",
                "alignment_confidence",
            ],
            "building_reasoning": False,
            "graph": False,
            "MIL": False,
            "memory": False,
            "morphology": False,
            "test_accessed": False,
        })
        return contract


__all__ = ["DamageFormerROISoftPartition", "FIXED_VARIANT"]
