from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .safe_affected_plugins import (
    RGBAnchoredLocalEvidence,
    RGBInputStructureStem,
    SARInputTextureStem,
    _groups,
)


@dataclass
class MarginAwareOutput:
    score_logits: torch.Tensor
    required_shift: torch.Tensor
    candidate_mask: torch.Tensor
    match_confidence: torch.Tensor
    match_entropy: torch.Tensor


def parent_decision_state(parent_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P3 state channels and the exact shift needed to reach Affected."""
    probability = parent_logits.float().softmax(1)
    entropy = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(1, keepdim=True)
    entropy = entropy / torch.tensor(4.0, device=entropy.device).log()
    top2 = probability.topk(2, dim=1).values
    top_margin = top2[:, :1] - top2[:, 1:2]
    affected_probability = probability[:, 2:4].sum(1, keepdim=True)
    nonaffected_logit = parent_logits[:, 0:2].amax(1, keepdim=True)
    affected_logit = parent_logits[:, 2:4].amax(1, keepdim=True)
    required = (nonaffected_logit - affected_logit).clamp_min(0.0)
    source = parent_logits.argmax(1)
    source_one_hot = torch.stack((source == 0, source == 1), dim=1).to(probability.dtype)
    state = torch.cat(
        (probability, affected_probability, entropy, top_margin, required, source_one_hot), dim=1
    )
    return state.to(parent_logits.dtype), required


class MarginAwareAffectedEvidence(nn.Module):
    """Shallow RGB/SAR evidence conditioned on the current P3 decision state."""

    state_channels = 10

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.rgb = RGBInputStructureStem(hidden)
        self.sar = SARInputTextureStem(hidden)
        self.matcher = RGBAnchoredLocalEvidence(hidden, max(hidden // 2, 4), 3)
        # RGB, SAR, direct relation, locally aligned relation, matching diagnostics,
        # P3 state and building probability are concatenated explicitly.
        channels = hidden * 4 + 4 + self.state_channels + 1
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(
        self,
        rgb_input: torch.Tensor,
        sar_input: torch.Tensor,
        parent_logits: torch.Tensor,
        p_building: torch.Tensor,
    ) -> MarginAwareOutput:
        size = parent_logits.shape[-2:]
        rgb = self.rgb(rgb_input, size)
        sar = self.sar(sar_input, size)
        local = self.matcher(rgb, sar)
        state, required = parent_decision_state(parent_logits)
        diagnostics = torch.cat((local.confidence, local.entropy, local.offset), dim=1)
        feature = torch.cat(
            (
                rgb,
                sar,
                (rgb - sar).abs(),
                (rgb - local.matched).abs(),
                diagnostics,
                state,
                p_building.detach(),
            ),
            dim=1,
        )
        source = parent_logits.argmax(1, keepdim=True)
        candidate = (source <= 1) & (p_building.detach() >= 0.5)
        return MarginAwareOutput(
            score_logits=self.head(feature),
            required_shift=required.detach(),
            candidate_mask=candidate,
            match_confidence=local.confidence,
            match_entropy=local.entropy,
        )


class ExactMarginAffectedResidual(nn.Module):
    """Promote only accepted candidates by their exact P3 decision margin."""

    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-3, maximum_shift: float = 8.0):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.epsilon = float(epsilon)
        self.maximum_shift = float(maximum_shift)

    def set_threshold(self, value: float) -> None:
        self.threshold.fill_(float(value))

    def forward(
        self,
        parent_logits: torch.Tensor,
        score_logits: torch.Tensor,
        required_shift: torch.Tensor,
        candidate_mask: torch.Tensor,
        hard: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(score_logits)
        if hard is None:
            hard = not self.training
        accepted = (probability >= self.threshold).to(probability.dtype)
        if not hard:
            # Straight-through gate: hard forward decision, useful score gradient.
            accepted = accepted + probability - probability.detach()
        accepted = accepted * candidate_mask.to(probability.dtype)
        shift = (required_shift + self.epsilon).clamp_max(self.maximum_shift) * accepted
        output = parent_logits.clone()
        output[:, 2:4] = output[:, 2:4] + shift
        return output, shift, accepted


__all__ = [
    "ExactMarginAffectedResidual",
    "MarginAwareAffectedEvidence",
    "MarginAwareOutput",
    "parent_decision_state",
]
