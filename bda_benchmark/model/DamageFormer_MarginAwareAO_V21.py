from __future__ import annotations

import torch
import torch.nn.functional as F

from .DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from .DamageFormer_ModelCentric_V1 import checkpoint_state
from .margin_aware_affected import ExactMarginAffectedResidual, MarginAwareAffectedEvidence


class DamageFormerMarginAwareAOV21(DamageFormer_S8_MICB_DSMIL):
    """Frozen P3 with candidate-only, state-conditioned affected correction."""

    plugin_prefixes = ("margin_evidence.", "margin_residual.")

    def __init__(self, hidden: int = 32, threshold: float = 0.5) -> None:
        super().__init__(mil_residual_scale=0.0)
        self.margin_evidence = MarginAwareAffectedEvidence(hidden)
        self.margin_residual = ExactMarginAffectedResidual(threshold=threshold)

    def load_parent_checkpoint(self, path: str) -> None:
        state = checkpoint_state(torch.load(path, map_location="cpu", weights_only=False))
        incompatible = self.load_state_dict(state, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith(self.plugin_prefixes)]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f"parent checkpoint mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}")

    def freeze_parent(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.plugin_prefixes))

    def set_plugin_training(self) -> None:
        self.eval()
        self.margin_evidence.train()
        self.margin_residual.train()

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)
        loc_feature = self.clf_1(rgb_feature)
        p_building = loc_feature.softmax(1)[:, 1:2]
        prior = p_building.detach() if self.detach_building_prior else p_building
        fused = self.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        parent_feature = self.clf_2(fused * (self.roi_alpha + (1.0 - self.roi_alpha) * prior))
        evidence = self.margin_evidence(pre_data, post_data, parent_feature, p_building)
        output_feature, shift, accepted = self.margin_residual(
            parent_feature,
            evidence.score_logits,
            evidence.required_shift,
            evidence.candidate_mask,
        )
        output_loc = F.interpolate(loc_feature, pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output = F.interpolate(output_feature, post_data.shape[-2:], mode="bilinear", align_corners=False)
        if not return_aux:
            return output_loc, output
        return output_loc, output, {
            "parent_logits_feature": parent_feature,
            "score_logits": evidence.score_logits,
            "required_shift": evidence.required_shift,
            "candidate_mask": evidence.candidate_mask,
            "accepted_mask": accepted,
            "affected_shift": shift,
            "match_confidence": evidence.match_confidence,
            "match_entropy": evidence.match_entropy,
        }


__all__ = ["DamageFormerMarginAwareAOV21"]
