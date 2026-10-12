"""Isolated V3 model built on the verified V2 region implementation."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import (
    DamageFormerBuildingInternalChangePartitionV2,
)

from .variants import get_variant


class _IgnoreAlignmentConfidence(nn.Module):
    """Ablation wrapper: preserve alignment values but remove quality signals."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    @property
    def gain(self):
        # Preserve V2's rule that the parent-feature injection gain remains
        # frozen: alignment supplies evidence only in this experiment.
        return self.base.gain

    def forward(self, *args, **kwargs):
        output = dict(self.base(*args, **kwargs))
        confidence = output["alignment_confidence"]
        output["alignment_confidence"] = torch.ones_like(confidence)
        if "alignment_entropy" in output:
            output["alignment_entropy"] = torch.zeros_like(output["alignment_entropy"])
        return output


class DamageFormerRGBAnchoredChangeSuperpixelV3(
    DamageFormerBuildingInternalChangePartitionV2
):
    """One shared soft partition: RGB anchor plus bounded change refinement."""

    def __init__(self, variant: str = "p5_anchored_change", **kwargs) -> None:
        spec = get_variant(variant)
        if not spec.use_change_refinement:
            base_variant = "p_s_struct_only"
        elif spec.shuffle_sar:
            base_variant = "p_sc_shuffled_sar"
        else:
            base_variant = "p_sc_residual"
        super().__init__(variant=base_variant, **kwargs)
        self.variant = variant
        self.variant_spec = spec
        self.biscp.spec = spec
        if not spec.use_alignment_confidence:
            self.biscp.lca = _IgnoreAlignmentConfidence(self.biscp.lca)

    def forward(self, *args, **kwargs):
        return_aux = bool(kwargs.get("return_aux", False))
        output = super().forward(*args, **kwargs)
        if not return_aux:
            return output
        loc, damage, aux = output
        aux["variant"] = self.variant
        aux["v3_use_damage_boundary"] = damage.new_tensor(
            float(self.variant_spec.use_damage_boundary)
        )
        aux["v3_use_alignment_confidence"] = damage.new_tensor(
            float(self.variant_spec.use_alignment_confidence)
        )
        return loc, damage, aux

    def checkpoint_contract(self) -> Dict[str, object]:
        contract = dict(super().checkpoint_contract())
        contract.update(
            {
                "protocol": "DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3",
                "variant": self.variant_spec.to_dict(),
                "partition_equation": "Q=softmax(L_rgb_structure+alpha*DeltaL_aligned_change)",
                "single_shared_partition": True,
                "independent_second_hard_partition": False,
                "gt_used_in_forward_partition": False,
                "test_accessed": False,
            }
        )
        return contract


__all__ = ["DamageFormerRGBAnchoredChangeSuperpixelV3"]
