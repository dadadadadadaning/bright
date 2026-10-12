from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import ModalitySpecificMoELite, SafeDifferenceResidual
from .heads import ConditionalSeverityResidual, HierarchicalSeverityResidual
from .sar_sources.base import freeze_module
from .sar_sources.p3_pseudo_sar import P3PseudoSARSource


VALID_P3_RESET_VARIANTS = ("c0", "c1", "c1s", "c2", "d1", "d2")


class P3ResetModel(nn.Module):
    """Controlled P3 reset chain with an immutable P3 prediction anchor.

    C0 is exact P3. C1/C2 alter the P3 fused representation through a safe
    zero-output residual. C1S uses the C1 evidence only for the conditional
    Damaged-vs-Destroyed odds while preserving P3's Intact-vs-Affected odds.
    D1/D2 change only building damage logits and keep P3 localisation untouched.
    The supplied SAR feature source is frozen here;
    feature-source fine tuning is a separate experiment, not an implicit side
    effect of fusion training.
    """

    def __init__(
        self,
        p3_model: nn.Module,
        variant: str = "c0",
        sar_source: Optional[nn.Module] = None,
        sar_key: str = "stage4",
        sar_channels: int = 768,
        interaction: str = "c1",
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in VALID_P3_RESET_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_P3_RESET_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.p3 = freeze_module(p3_model)
        self.sar_source = freeze_module(sar_source) if sar_source is not None else None
        self.sar_key = sar_key
        active_sar_channels = 256 if sar_source is None else int(sar_channels)
        interaction = interaction.lower()
        if interaction not in ("c1", "c2"):
            raise ValueError("interaction must be c1 or c2")
        self.interaction = interaction
        if variant in ("c1", "c1s") or (variant in ("d1", "d2") and interaction == "c1"):
            self.fusion_residual = SafeDifferenceResidual(256, active_sar_channels, out_channels=256)
        elif variant == "c2" or (variant in ("d1", "d2") and interaction == "c2"):
            self.fusion_residual = ModalitySpecificMoELite(256, active_sar_channels, out_channels=256)
        else:
            self.fusion_residual = None
        self.damage_head = None
        if variant == "d1":
            self.damage_head = ConditionalSeverityResidual(256)
        elif variant == "d2":
            self.damage_head = HierarchicalSeverityResidual(256)

    def train(self, mode: bool = True):
        super().train(mode)
        self.p3.eval()
        if self.sar_source is not None:
            self.sar_source.eval()
        return self

    def _external_sar(self, post_sar: torch.Tensor, p3_sar: torch.Tensor) -> torch.Tensor:
        if self.sar_source is None:
            return p3_sar
        features = self.sar_source(post_sar)
        if self.sar_key not in features:
            raise KeyError(f"SAR source has {sorted(features)}, requested {self.sar_key!r}")
        return features[self.sar_key]

    @staticmethod
    def _severity_only_update(
        base_logits: torch.Tensor,
        severity_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Change D-vs-X odds without changing total Affected probability.

        The subtraction of two identically reconstructed base states makes the
        update exactly zero when ``severity_delta`` is zero.  Background and
        Intact logits are never changed, and logsumexp(Damaged, Destroyed) is
        preserved up to floating-point round-off.
        """
        if base_logits.ndim != 4 or base_logits.shape[1] != 4:
            raise ValueError("severity-only update expects [B,4,H,W] logits")
        if severity_delta.shape != base_logits[:, :1].shape:
            raise ValueError("severity_delta must have shape [B,1,H,W]")
        base_severity = base_logits[:, 3:4] - base_logits[:, 2:3]
        affected_mass = torch.logsumexp(base_logits[:, 2:4], dim=1, keepdim=True)

        def split(severity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            damaged = affected_mass - F.softplus(severity)
            destroyed = affected_mass - F.softplus(-severity)
            return damaged, destroyed

        base_damaged, base_destroyed = split(base_severity)
        new_damaged, new_destroyed = split(base_severity + severity_delta)
        update = torch.zeros_like(base_logits)
        update[:, 2:3] = new_damaged - base_damaged
        update[:, 3:4] = new_destroyed - base_destroyed
        return base_logits + update

    def forward(self, pre_rgb: torch.Tensor, post_sar: torch.Tensor, return_dict: bool = False):
        p3_post = P3PseudoSARSource.to_p3_input(post_sar)
        with torch.no_grad():
            output_loc, base_logits, aux = self.p3(pre_rgb, p3_post, return_aux=True)
        if self.variant == "c0":
            if return_dict:
                return self._output_dict(output_loc, base_logits, {"variant": self.variant, "exact_p3": True})
            return output_loc, base_logits

        rgb = aux["rgb_feature"]
        sar = self._external_sar(post_sar, aux["sar_feature"])
        residual, diagnostics = self.fusion_residual(rgb, sar)
        base_fused = aux["fused_roi"]
        # Difference-of-heads preserves the exact P3 logit path at zero residual.
        candidate_low = self.p3.clf_2(base_fused + residual)
        base_low = self.p3.clf_2(base_fused)
        feature_delta = F.interpolate(
            candidate_low - base_low,
            base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        support = F.interpolate(
            aux["p_building"].detach(),
            base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature_delta = feature_delta * support
        feature_delta = feature_delta.clone()
        feature_delta[:, 0] = 0.0
        affected_override = None
        if self.variant == "c1s":
            # C1 showed useful D-vs-X evidence but damaged the occurrence gate.
            # Retain only the conditional severity component of its proposal.
            severity_delta = feature_delta[:, 3:4] - feature_delta[:, 2:3]
            logits = self._severity_only_update(base_logits, severity_delta)
            # The evaluator consumes this log-odds for occurrence metrics.
            # Reuse P3's tensor directly so the occurrence decision is not
            # vulnerable to a near-zero floating-point reconstruction error.
            affected_override = (
                torch.logsumexp(base_logits[:, 2:4], dim=1, keepdim=True)
                - base_logits[:, 1:2]
            )
            diagnostics.update({
                "severity_delta_abs": severity_delta.detach().abs().mean().reshape(1),
                "occurrence_delta_abs": (
                    (torch.logsumexp(logits[:, 2:4], dim=1, keepdim=True) - logits[:, 1:2])
                    - (torch.logsumexp(base_logits[:, 2:4], dim=1, keepdim=True) - base_logits[:, 1:2])
                ).detach().abs().mean().reshape(1),
            })
        else:
            logits = base_logits + feature_delta
        feature = base_fused + residual
        if self.damage_head is not None:
            logits, head_diagnostics = self.damage_head(
                p3_logits=logits,
                feature=feature,
                building_probability=aux["p_building"],
            )
            diagnostics.update(head_diagnostics)
        diagnostics.update(
            {
                "variant": self.variant,
                "exact_p3": False,
                "feature_delta_abs": feature_delta.detach().abs().mean().reshape(1),
            }
        )
        if return_dict:
            return self._output_dict(output_loc, logits, diagnostics, affected_override)
        return output_loc, logits

    @staticmethod
    def _output_dict(
        output_loc: torch.Tensor,
        logits: torch.Tensor,
        diagnostics: Dict[str, object],
        affected_override: Optional[torch.Tensor] = None,
    ):
        affected = affected_override
        if affected is None:
            affected = torch.logsumexp(logits[:, 2:4], dim=1, keepdim=True) - logits[:, 1:2]
        severity = logits[:, 3:4] - logits[:, 2:3]
        output: Dict[str, object] = {
            "output_loc": output_loc,
            "output_dam": logits,
            "h10_affected_logits": affected,
            "h10_severity_logits": severity,
        }
        for key, value in diagnostics.items():
            if torch.is_tensor(value):
                output[f"p3_reset_{key}"] = value
        return output

    def trainable_parameter_names(self) -> Tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
