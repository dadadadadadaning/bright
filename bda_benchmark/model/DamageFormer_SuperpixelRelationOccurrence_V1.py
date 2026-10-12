"""Frozen P3 with a *structural*, not semantic, superpixel relation adapter.

The adapter never assigns a class to a superpixel and never broadcasts a
region label.  It only pools already-computed pixel features inside regions
made from the augmented pre-disaster RGB crop.  The four-class P3 decoder
remains pixel-wise; the safe residual can only increase the joint affected
mass (Damaged and Destroyed equally).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .DamageFormer_AffectedOccurrence_V1 import _groups
from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from .DamageFormer_ModelCentric_V1 import checkpoint_state
from .safe_affected_plugins import SafeAffectedResidual
from bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.core import (
    make_region_ids, pooled_region_feature, region_boundary, split_regions_by_building,
)


@dataclass(frozen=True)
class SuperpixelVariant:
    region_mode: str
    consistency: bool


VARIANTS: Mapping[str, SuperpixelVariant] = {
    # Same adapter parameters, but all region information is suppressed.
    "sp1_parameter_matched": SuperpixelVariant("none", False),
    "sp2_slic_relation": SuperpixelVariant("slic", False),
    "sp3_slic_consistency": SuperpixelVariant("slic", True),
    "sp4_shuffled_slic": SuperpixelVariant("shuffled", True),
    "sp5_regular_grid": SuperpixelVariant("grid", True),
}


class SuperpixelRelationEvidence(nn.Module):
    """Returns an occurrence logit plus non-semantic local relation features."""

    def __init__(self, hidden: int, region_mode: str, segments: int, compactness: float) -> None:
        super().__init__()
        self.region_mode, self.segments, self.compactness = region_mode, int(segments), float(compactness)
        self.rgb = nn.Sequential(nn.Conv2d(192, hidden, 1, bias=False), nn.GroupNorm(_groups(hidden), hidden), nn.GELU())
        self.sar = nn.Sequential(nn.Conv2d(192, hidden, 1, bias=False), nn.GroupNorm(_groups(hidden), hidden), nn.GELU())
        # This module is retained in SP1 as a parameter-matched control.
        self.relation = nn.Sequential(
            # joint, pooled(joint), and their difference each have 2*hidden channels.
            nn.Conv2d(hidden * 6 + 1, hidden, 1, bias=False), nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden * 3, 1, 1)

    def forward(self, pre_rgb: torch.Tensor, rgb_stage2: torch.Tensor, sar_stage2: torch.Tensor, p_building: torch.Tensor, size: tuple[int, int]):
        rgb = F.interpolate(self.rgb(rgb_stage2), size=size, mode="bilinear", align_corners=False)
        sar = F.interpolate(self.sar(sar_stage2), size=size, mode="bilinear", align_corners=False)
        ids = make_region_ids(pre_rgb, size, mode=self.region_mode, n_segments=self.segments, compactness=self.compactness)
        inside_bool = p_building.detach() >= .5
        if self.region_mode != "none":
            # Pool building and non-building parts separately even when one
            # raw SLIC region crosses the predicted building boundary.
            ids = split_regions_by_building(ids, inside_bool)
        boundary = region_boundary(ids)
        joint = torch.cat((rgb, sar), 1)
        pooled = pooled_region_feature(joint, ids)
        if self.region_mode == "none":
            # Keep identical parameter count but prevent region context entering SP1.
            pooled = joint.detach() * 0.0
        relation_input = torch.cat((joint, pooled, joint - pooled, boundary.to(joint.dtype)), 1)
        relation = self.relation(relation_input)
        inside = inside_bool.to(relation.dtype)
        # Do not spread across a region border or from low-confidence localization.
        safe_gate = inside * (1.0 - boundary.to(relation.dtype))
        logits = self.head(torch.cat((rgb, sar, relation * safe_gate), 1))
        return logits, {"region_ids": ids, "region_boundary": boundary, "relation_feature": relation, "safe_gate": safe_gate}


class DamageFormerSuperpixelRelationOccurrenceV1(DamageFormer_S8_MICB_DSMIL):
    plugin_prefixes = ("superpixel_evidence.", "occurrence_residual.")

    def __init__(self, variant: str, hidden: int = 64, segments: int = 192, compactness: float = 8.0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose {tuple(VARIANTS)}")
        super().__init__(mil_residual_scale=0.0)
        self.variant, self.variant_config = variant, VARIANTS[variant]
        self.superpixel_evidence = SuperpixelRelationEvidence(hidden, self.variant_config.region_mode, segments, compactness)
        self.occurrence_residual = SafeAffectedResidual(maximum_shift=4.5)

    def load_parent_checkpoint(self, path: str) -> None:
        state = checkpoint_state(torch.load(path, map_location="cpu", weights_only=False))
        result = self.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith(self.plugin_prefixes)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"P3 checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")

    def freeze_parent(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.plugin_prefixes))

    def set_plugin_training(self) -> None:
        self.eval(); self.superpixel_evidence.train(); self.occurrence_residual.train()

    def project_safe_parameters(self) -> None:
        self.occurrence_residual.project_parameters()

    def _encode(self, pre: torch.Tensor, post: torch.Tensor):
        _, pre1, pre2, pre3, pre4 = self.encoder_1(pre)
        _, post1, post2, post3, post4 = self.encoder_2(post)
        pre1, pre2, pre3, pre4 = map(self._to_nchw, (pre1, pre2, pre3, pre4))
        post1, post2, post3, post4 = map(self._to_nchw, (post1, post2, post3, post4))
        rgb = self.fusion_layer_1(self._multi_scale_fuse(pre1, pre2, pre3, pre4))
        sar = self.fusion_layer_2(self._multi_scale_fuse(post1, post2, post3, post4))
        return rgb, sar, pre2, post2

    def forward(self, pre: torch.Tensor, post: torch.Tensor, return_aux: bool = False):
        rgb, sar, rgb_stage2, sar_stage2 = self._encode(pre, post)
        loc_feature = self.clf_1(rgb); p_building = loc_feature.softmax(1)[:, 1:2]
        roi = self.roi_alpha + (1.0 - self.roi_alpha) * p_building.detach()
        parent_feature = self.clf_2(self.fusion_layer_3(torch.cat((rgb, sar), 1)) * roi)
        occurrence, diagnostic = self.superpixel_evidence(pre, rgb_stage2, sar_stage2, p_building, parent_feature.shape[-2:])
        with torch.no_grad():
            probability = parent_feature.float().softmax(1)
            support = (0.10 + 0.90 * p_building.detach()) * probability[:, :2].sum(1, keepdim=True)
        damage_feature, delta = self.occurrence_residual(parent_feature, occurrence, support, diagnostic["safe_gate"])
        loc = F.interpolate(loc_feature, pre.shape[-2:], mode="bilinear", align_corners=False)
        damage = F.interpolate(damage_feature, post.shape[-2:], mode="bilinear", align_corners=False)
        if not return_aux:
            return loc, damage
        return loc, damage, {"parent_logits_feature": parent_feature, "occurrence_logits": occurrence,
                             "affected_delta": delta, "p_building": p_building,
                             "affected_gain": self.occurrence_residual.gain, **diagnostic}


__all__ = ["DamageFormerSuperpixelRelationOccurrenceV1", "SuperpixelRelationEvidence", "VARIANTS"]
