"""Decision-level use of frozen V2 Stage1 region evidence.

The A1 parent and all V2 partition/evidence modules remain frozen.  New
parameters use the isolated ``readout.`` prefix and are identity-initialised.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.association import (
    soft_unpool_tokens,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import (
    DamageFormerBuildingInternalChangePartitionV2,
)

from .variants import get_variant


def _head(in_channels: int, out_channels: int, *, final_bias: float = 0.0) -> nn.Sequential:
    module = nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, bias=False),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.constant_(module[-1].bias, final_bias)
    return module


class EvidenceReadout(nn.Module):
    SAFE_CHANNELS = 64 + 128 + 4 + 1 + 1 + 1

    def __init__(self, variant: str, delta_max: float = 0.50, expert_max: float = 0.75) -> None:
        super().__init__()
        self.variant = variant
        self.spec = get_variant(variant)
        self.delta_max = float(delta_max)
        self.expert_max = float(expert_max)
        self.log_scale = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(2))
        self.id_gate_head = _head(self.SAFE_CHANNELS, 1, final_bias=-2.0)
        self.id_delta_head = _head(self.SAFE_CHANNELS, 1)
        self.id_expert_residual_head = _head(self.SAFE_CHANNELS, 2)
        self.logit4_residual_head = _head(self.SAFE_CHANNELS, 4)

    @staticmethod
    def _threshold_buildings(
        score: torch.Tensor,
        building_ids: torch.Tensor,
        base_intact: torch.Tensor,
        threshold: Optional[float],
    ) -> torch.Tensor:
        if threshold is None:
            return torch.ones_like(score)
        output = torch.zeros_like(score)
        for batch_index in range(building_ids.shape[0]):
            for building_id in torch.unique(building_ids[batch_index]).tolist():
                if int(building_id) <= 0:
                    continue
                mask = (building_ids[batch_index] == int(building_id)).unsqueeze(0)
                candidate = mask & base_intact[batch_index]
                if not bool(candidate.any().item()):
                    continue
                mean_score = score[batch_index][candidate].float().mean()
                if float(mean_score) >= float(threshold):
                    output[batch_index][mask] = 1.0
        return output

    def forward(
        self,
        parent_logits: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        *,
        selective_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        base_probability = torch.softmax(parent_logits.detach().float(), dim=1).to(parent_logits.dtype)
        if self.variant == "w1_id_calibration":
            output = parent_logits.clone()
            scale = torch.exp(self.log_scale).to(output.dtype)
            output[:, 1:2] = scale[0] * parent_logits[:, 1:2] + self.bias[0].to(output.dtype)
            output[:, 2:3] = scale[1] * parent_logits[:, 2:3] + self.bias[1].to(output.dtype)
            zero = parent_logits[:, :1] * 0.0
            return {
                "damage_logits": output,
                "id_gate": zero,
                "id_gate_logits": zero,
                "id_gate_score": zero,
                "id_logit_delta": zero,
                "id_expert_residual": parent_logits[:, 1:3] * 0.0,
                "delta4": parent_logits * 0.0,
                "candidate_gate": zero,
                "selective_gate": torch.ones_like(zero),
            }

        h, w = parent_logits.shape[-2:]
        token_margin = (aux["region_logits"][..., 1] - aux["region_logits"][..., 0]).unsqueeze(-1)
        margin_map = soft_unpool_tokens(
            token_margin,
            aux["association_probabilities"],
            aux["candidate_ids"],
            h,
            w,
        )
        safe_input = torch.cat(
            (
                aux["region_relation"].to(parent_logits.dtype),
                aux["region_context"].to(parent_logits.dtype),
                base_probability,
                aux["region_confidence"].to(parent_logits.dtype),
                aux["region_alignment_entropy"].to(parent_logits.dtype),
                margin_map.to(parent_logits.dtype),
            ),
            dim=1,
        )
        gate_logits = self.id_gate_head(safe_input)
        gate_score = torch.sigmoid(gate_logits)
        hard_building = (aux["building_ids"] > 0).unsqueeze(1).to(parent_logits.dtype)
        base_pred = base_probability.argmax(dim=1, keepdim=True)
        base_intact = (base_pred == 1).to(parent_logits.dtype)
        candidate_gate = hard_building * base_intact
        selective_gate = self._threshold_buildings(
            gate_score, aux["building_ids"], base_intact.bool(), selective_threshold
        )
        gate = candidate_gate * selective_gate * gate_score

        id_delta = self.delta_max * torch.tanh(self.id_delta_head(safe_input)) * gate
        expert_residual = self.expert_max * torch.tanh(self.id_expert_residual_head(safe_input))
        delta4 = 0.25 * torch.tanh(self.logit4_residual_head(safe_input)) * hard_building

        if self.variant == "w2_id_logit":
            output = parent_logits.clone()
            output[:, 1:2] = parent_logits[:, 1:2] - id_delta
            output[:, 2:3] = parent_logits[:, 2:3] + id_delta
        elif self.variant == "w3_id_mass":
            base_id_logits = parent_logits[:, 1:3]
            q_base = torch.softmax(base_id_logits.detach().float(), dim=1).to(parent_logits.dtype)
            q_expert = torch.softmax((base_id_logits + expert_residual).float(), dim=1).to(parent_logits.dtype)
            # Residual form is important: when the zero-initialised expert is
            # identical to q_base, q_expert-q_base is exactly zero and the
            # original parent logits are returned bit-for-bit.  Adding the
            # conditional log-ratio also preserves exp(z_I)+exp(z_D), hence
            # Background, Destroyed and total I/D probability remain fixed.
            q_mix = q_base + gate * (q_expert - q_base)
            log_ratio = (
                torch.log(q_mix.float().clamp_min(1.0e-8))
                - torch.log(q_base.float().clamp_min(1.0e-8))
            ).to(parent_logits.dtype)
            output = parent_logits.clone()
            output[:, 1:3] = base_id_logits + log_ratio
        elif self.variant in ("w0_full_residual", "w5_logit4_kd"):
            output = parent_logits + delta4
        else:
            raise ValueError(f"Unsupported readout variant: {self.variant}")
        return {
            "damage_logits": output,
            "safe_input": safe_input,
            "region_id_margin_map": margin_map,
            "id_gate": gate,
            "id_gate_logits": gate_logits,
            "id_gate_score": gate_score,
            "id_logit_delta": id_delta,
            "id_expert_residual": expert_residual,
            "delta4": delta4,
            "candidate_gate": candidate_gate,
            "selective_gate": selective_gate,
        }


class DamageFormerRegionEvidenceReadout(DamageFormerBuildingInternalChangePartitionV2):
    readout_prefix = "readout."

    def __init__(self, readout_variant: str, **kwargs) -> None:
        super().__init__(variant="p_sc_residual", **kwargs)
        self.readout_variant = readout_variant
        self.readout_spec = get_variant(readout_variant)
        self.readout = EvidenceReadout(readout_variant)
        self.selective_threshold: Optional[float] = None

    def configure_readout_training(self) -> None:
        # Stage1 Region formation/evidence and the A1 parent are immutable.
        self.configure_stage("change")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith(self.readout_prefix):
                parameter.requires_grad_(True)
        # Only parameters used by the selected causal branch should enter AdamW.
        used = {
            "w0_full_residual": ("readout.logit4_residual_head.",),
            "w1_id_calibration": ("readout.log_scale", "readout.bias"),
            "w2_id_logit": ("readout.id_gate_head.", "readout.id_delta_head."),
            "w3_id_mass": ("readout.id_gate_head.", "readout.id_expert_residual_head."),
            "w5_logit4_kd": ("readout.logit4_residual_head.",),
        }[self.readout_variant]
        for name, parameter in self.named_parameters():
            if name.startswith(self.readout_prefix) and not any(name.startswith(prefix) for prefix in used):
                parameter.requires_grad_(False)

    def set_readout_training(self) -> None:
        self.eval()
        self.readout.train()

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        output_loc, _, aux = super().forward(pre_data, post_data, return_aux=True)
        parent_logits = aux["parent_damage_logits_feature"]
        readout = self.readout(
            parent_logits,
            aux,
            selective_threshold=self.selective_threshold,
        )
        output_damage = F.interpolate(
            readout["damage_logits"], post_data.shape[-2:], mode="bilinear", align_corners=False
        )
        if not return_aux:
            return output_loc, output_damage
        aux.update(readout)
        aux["readout_variant"] = self.readout_variant
        return output_loc, output_damage, aux

    def checkpoint_contract(self) -> Dict[str, object]:
        return {
            "protocol": "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1",
            "readout": self.readout_spec.to_dict(),
            "a1_frozen": True,
            "v2_stage1_frozen": True,
            "step0_identity": True,
            "test_accessed": False,
        }


__all__ = ["DamageFormerRegionEvidenceReadout", "EvidenceReadout"]
