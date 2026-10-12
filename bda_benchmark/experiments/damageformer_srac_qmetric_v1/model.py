"""Q-sharpening-only adaptation of the audited Transformer-SRAC front.

The parent architecture is intentionally unchanged.  The only computational
difference is a scalar beta applied to the *complete* nine-candidate score
tensor immediately before softmax at every center-refinement iteration.  A
beta of one therefore reproduces the parent partition exactly.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map,
)
from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import (
    grouped_centers,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    SRACTransformerFront,
    SwinLocalSuperpixelAttention,
)


FRONT_FORMAT = "SRAC_QMETRIC_FRONT_V1"
VARIANTS = ("control", "q_anneal")


def q_beta_schedule(
    step: int,
    start: float = 1.0,
    end: float = 2.0,
    hold: int = 1000,
    ramp_end: int = 10000,
) -> float:
    """Piecewise-linear beta schedule used by the isolated Q experiment."""
    if step < 0 or start <= 0 or end <= 0 or hold < 0 or ramp_end <= hold:
        raise ValueError("Invalid Q-beta schedule")
    if step <= hold:
        return float(start)
    if step >= ramp_end:
        return float(end)
    progress = (float(step) - float(hold)) / float(ramp_end - hold)
    return float(start + progress * (end - start))


def _gather_centers(centers: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return centers.gather(1, index[..., None].expand(-1, -1, centers.shape[-1]))


@torch.no_grad()
def q_metrics(
    q: torch.Tensor,
    ids: torch.Tensor,
    count: int,
    valid_candidates: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Descriptive routing statistics; none of these terms enters the loss."""
    if q.ndim != 3 or ids.shape != q.shape[1:]:
        raise ValueError("Expected q[B,S,N] and ids[S,N]")
    batch, supports, pixels = q.shape
    if count < 1 or ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= count:
        raise ValueError("Invalid region IDs")
    probability = q.detach().float()
    if valid_candidates is None:
        valid_candidates = torch.ones_like(ids, dtype=torch.bool)
    if valid_candidates.shape != ids.shape:
        raise ValueError("Candidate-validity shape mismatch")

    raw_entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
    candidate_count = valid_candidates.sum(0).to(probability.dtype)
    denominator = candidate_count.clamp_min(2).log()
    normalized_entropy = torch.where(
        candidate_count > 1,
        raw_entropy / denominator,
        torch.zeros_like(raw_entropy),
    )

    mass = probability.new_zeros(batch, count)
    expanded_ids = ids[None].expand(batch, -1, -1)
    for support_index in range(supports):
        mass.scatter_add_(1, expanded_ids[:, support_index], probability[:, support_index])
    soft_occupied = (mass > 1e-6).sum(1).float()

    winner = probability.argmax(1)
    hard_ids = ids.transpose(0, 1).gather(1, winner.transpose(0, 1)).transpose(0, 1)
    hard_mass = probability.new_zeros(batch, count)
    hard_mass.scatter_add_(1, hard_ids, torch.ones_like(hard_ids, dtype=probability.dtype))
    hard_occupied = (hard_mass > 0).sum(1).float()

    return {
        "q_top1_mean": probability.max(1).values.mean(),
        "q_entropy_valid_mean": normalized_entropy.mean(),
        "q_valid_candidates_mean": candidate_count.mean(),
        "q_soft_occupied_mean": soft_occupied.mean(),
        "q_hard_occupied_mean": hard_occupied.mean(),
        "q_region_slots": probability.new_tensor(float(count)),
    }


class AnnealedSwinLocalSuperpixelAttention(SwinLocalSuperpixelAttention):
    """Parent local SSN attention with one parameter-free beta buffer."""

    def set_beta(self, beta: float) -> None:
        if beta <= 0 or not math.isfinite(beta):
            raise ValueError("Q beta must be finite and positive")
        self.q_beta.fill_(float(beta))

    @property
    def current_beta(self) -> float:
        return float(self.q_beta.item())

    def forward(self, embedding: torch.Tensor, building: torch.Tensor) -> Dict[str, torch.Tensor]:
        # This body mirrors the audited parent implementation.  The only model
        # change is the beta multiplication immediately before every softmax.
        batch, channels, height, width = embedding.shape
        ids, valid, grid_h, grid_w = candidate_map(height, width, self.stride, embedding.device)
        count = grid_h * grid_w
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            pixels = embedding.float().flatten(2).transpose(1, 2)
            pixels = self.norm(pixels)
            query = F.normalize(self.query(pixels), dim=-1)
            values = F.normalize(pixels, dim=-1)

            yy, xx = torch.meshgrid(
                torch.arange(height, device=embedding.device),
                torch.arange(width, device=embedding.device),
                indexing="ij",
            )
            coordinates = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2).float()
            coordinates = coordinates.div(float(self.stride)).expand(batch, -1, -1)
            building_grid = F.interpolate(building.detach().float(), (height, width), mode="bilinear", align_corners=False)
            building_flat = building_grid.flatten(2).transpose(1, 2)

            index = ids[None].expand(batch, -1, -1)
            assignment = embedding.new_zeros(batch, 9, height * width, dtype=torch.float32)
            assignment[:, 4] = 1.0
            scale = self.logit_scale.exp().clamp(1e-3, 100.0)

            for _ in range(self.iterations):
                center_value, mass = grouped_centers(values, assignment, index, count)
                center_xy, _ = grouped_centers(coordinates, assignment, index, count)
                center_building, _ = grouped_centers(building_flat, assignment, index, count)
                scores = []
                for candidate in range(9):
                    candidate_index = index[:, candidate]
                    center_key = F.normalize(self.key(_gather_centers(center_value, candidate_index)), dim=-1)
                    similarity = (query * center_key).sum(-1) * scale
                    distance = (coordinates - _gather_centers(center_xy, candidate_index)).square().sum(-1)
                    center_b = _gather_centers(center_building, candidate_index).squeeze(-1).clamp(1e-4, 1 - 1e-4)
                    pixel_b = building_flat.squeeze(-1)
                    compatibility = (pixel_b * center_b + (1 - pixel_b) * (1 - center_b)).clamp_min(1e-4)
                    score = similarity / self.temperature
                    score = score - self.spatial_weight * distance
                    score = score + self.building_weight * compatibility.log()
                    occupied = mass.gather(1, candidate_index) > 1e-6
                    score = score.masked_fill(~(valid[candidate][None] & occupied), -torch.inf)
                    scores.append(score)
                candidate_logits = torch.stack(scores, dim=1)
                # Avoid even an identity multiply in the beta-one control.  This
                # preserves the parent's values and gradients bit-for-bit.
                if self.current_beta != 1.0:
                    candidate_logits = candidate_logits * self.q_beta
                assignment = candidate_logits.softmax(dim=1)

        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(1)
        entropy_map = (entropy / math.log(9.0)).reshape(batch, 1, height, width)
        statistics = q_metrics(assignment, ids, count, valid)
        return {
            "q": assignment,
            "ids": ids,
            "shape": (height, width),
            "grid_shape": (grid_h, grid_w),
            "count": count,
            # Preserve the exact parent entropy definition because it is an
            # existing structural input to the unchanged adapter.
            "entropy_map": entropy_map,
            "q_beta": self.q_beta.detach().clone(),
            "q_logit_scale_exp": scale.detach().clone(),
            "q_similarity_coefficient": (scale * self.q_beta / self.temperature).detach().clone(),
            **statistics,
        }


def _upgrade_partition_without_initialization(partition: SwinLocalSuperpixelAttention, beta: float) -> None:
    """Change behavior without creating parameters or consuming RNG state."""
    if type(partition) is not SwinLocalSuperpixelAttention:
        raise TypeError("Expected the exact parent partition class")
    partition.__class__ = AnnealedSwinLocalSuperpixelAttention
    partition.register_buffer("q_beta", torch.tensor(float(beta), dtype=torch.float32), persistent=True)


class AnnealedFront(SRACTransformerFront):
    """Unchanged parent front with a scheduled Q-softmax beta."""

    def __init__(
        self,
        dim: int = 128,
        decoder_width: int = 64,
        stride: int = 8,
        iterations: int = 3,
        temperature: float = 0.2,
        spatial_weight: float = 0.1,
        building_weight: float = 0.5,
        region_context: bool = False,
        variant: str = "control",
        q_beta_start: float = 1.0,
        q_beta_end: float = 1.0,
        q_beta_hold: int = 1000,
        q_beta_ramp_end: int = 10000,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Q variant: {variant}")
        # Validate before constructing the parent so invalid requests do not
        # consume model-initialization RNG.
        q_beta_schedule(0, q_beta_start, q_beta_end, q_beta_hold, q_beta_ramp_end)
        if variant == "control" and (q_beta_start != 1.0 or q_beta_end != 1.0):
            raise ValueError("The control must keep beta exactly one")
        super().__init__(
            dim=dim,
            decoder_width=decoder_width,
            stride=stride,
            iterations=iterations,
            temperature=temperature,
            spatial_weight=spatial_weight,
            building_weight=building_weight,
            region_context=region_context,
        )
        _upgrade_partition_without_initialization(self.partition, q_beta_start)
        self.variant = variant
        self.q_beta_start = float(q_beta_start)
        self.q_beta_end = float(q_beta_end)
        self.q_beta_hold = int(q_beta_hold)
        self.q_beta_ramp_end = int(q_beta_ramp_end)
        self.config.update(
            variant=variant,
            q_beta_start=self.q_beta_start,
            q_beta_end=self.q_beta_end,
            q_beta_hold=self.q_beta_hold,
            q_beta_ramp_end=self.q_beta_ramp_end,
        )

    @property
    def current_beta(self) -> float:
        return self.partition.current_beta

    def set_step(self, step: int) -> float:
        beta = q_beta_schedule(
            step,
            self.q_beta_start,
            self.q_beta_end,
            self.q_beta_hold,
            self.q_beta_ramp_end,
        )
        self.partition.set_beta(beta)
        return beta

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = super().forward(pre, post)
        delta = (result["adapted_content"].float() - result["content_only"].float()).flatten(1)
        base = result["content_only"].float().flatten(1)
        delta_norm = delta.norm(dim=1).mean()
        base_norm = base.norm(dim=1).mean()
        result.update(
            adapter_delta_norm=delta_norm,
            adapter_base_norm=base_norm,
            adapter_delta_base_ratio=delta_norm / base_norm.clamp_min(1e-8),
            current_beta=result["q_beta"],
        )
        return result


__all__ = [
    "AnnealedFront",
    "AnnealedSwinLocalSuperpixelAttention",
    "FRONT_FORMAT",
    "VARIANTS",
    "q_beta_schedule",
    "q_metrics",
]
