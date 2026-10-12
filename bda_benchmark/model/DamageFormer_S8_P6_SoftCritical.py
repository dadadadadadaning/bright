# -*- coding: utf-8 -*-
"""S8-P6: isolated A4 native-SAR + differentiable soft-critical MICB.

This experiment keeps P3's RGB localisation, ROI prior, pixel classifier,
checkpoint namespaces and zero inference MIL residual.  Its only architectural
change is the occurrence-critical aggregation used by the training-only MICB
auxiliary head.  N1 shared cross-attention is intentionally excluded because
its validation result increased damaged gate-miss and reduced event worst.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from model.DamageFormer_S8_MICB_DSMIL import MICBDSMILHead
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
)


class SoftCriticalMICBDSMILHead(MICBDSMILHead):
    """Replace hard argmax occurrence pooling with a soft multi-token pool.

    The normalised log-sum-exp scalar is invariant to duplicating identical
    tokens, avoiding a hidden preference for large connected components.
    This class introduces no trainable parameter, so accepted P3 checkpoints
    remain strict-load compatible.
    """

    def __init__(self, *args, occurrence_temperature: float = 0.5, **kwargs):
        if float(occurrence_temperature) <= 0.0:
            raise ValueError("occurrence_temperature must be positive")
        super().__init__(*args, **kwargs)
        self.occurrence_temperature = float(occurrence_temperature)

    def _aggregate_occurrence_critical(
        self,
        tokens: torch.Tensor,
        local_exist_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or local_exist_logits.ndim != 1:
            raise ValueError("tokens/logits must be [N,C] and [N]")
        if tokens.shape[0] != local_exist_logits.shape[0] or tokens.shape[0] == 0:
            raise ValueError("tokens and logits must contain the same non-empty bag")
        temperature = self.occurrence_temperature
        weights = torch.softmax(local_exist_logits / temperature, dim=0)
        critical = (weights[:, None] * tokens).sum(dim=0)
        evidence = temperature * (
            torch.logsumexp(local_exist_logits / temperature, dim=0)
            - math.log(float(local_exist_logits.numel()))
        )
        return critical, evidence[None]


class DamageFormer_S8_P6_SoftCritical(DamageFormer_S8_P4_NativeSAR_MICB):
    """Function-preserving A4 input plus a training-only soft-critical head."""

    def __init__(
        self,
        variant: str = "a4",
        occurrence_temperature: float = 0.5,
        **kwargs,
    ) -> None:
        if str(variant).lower() != "a4":
            raise ValueError("S8-P6 is locked to the function-preserving A4 variant")
        super().__init__(variant="a4", **kwargs)
        old_head = self.micb_dsmil
        new_head = SoftCriticalMICBDSMILHead(
            in_channels=256,
            token_dim=old_head.token_dim,
            topk=old_head.topk,
            max_tokens=old_head.max_tokens,
            attention_temperature=old_head.attention_temperature,
            mask_threshold=old_head.mask_threshold,
            ring_kernel=old_head.ring_kernel,
            use_asymmetric_regions=old_head.use_asymmetric_regions,
            occurrence_temperature=occurrence_temperature,
        )
        # Keep the exact P3 parameter initialisation before the external P3
        # checkpoint is strict-loaded.  This also makes direct unit tests
        # deterministic and documents that no namespace was added.
        new_head.load_state_dict(old_head.state_dict(), strict=True)
        self.micb_dsmil = new_head
        self.occurrence_temperature = float(occurrence_temperature)

    def experiment_metadata(self) -> Dict[str, object]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "family": "S8-P6 NativeSAR SoftCritical MICB",
                "soft_critical_enabled": True,
                "occurrence_temperature": self.occurrence_temperature,
                "shared_cross_attention_enabled": False,
                "pixel_occurrence_head_enabled": False,
                "mil_residual_scale": 0.0,
            }
        )
        return metadata


__all__ = [
    "DamageFormer_S8_P6_SoftCritical",
    "SoftCriticalMICBDSMILHead",
]
