"""Transformer-aware SRAC-style front end on an unchanged DamageFormer/A1.

The A1 backbone supplies features but its logits are never added to the final
prediction.  The final prediction is made directly from superpixel tokens, and
later from superpixel tokens jointly attended to Train-only memory supports.

This is an auditable approximation of the public SRAC diagram.  The original
paper/code does not currently expose enough implementation detail for a strict
reproduction.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map,
    soft_pool_unpool,
)
from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import grouped_centers
from bda_benchmark.experiments.damageformer_joint_region_v1.model import JointModel


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    groups = min(8, cout)
    while cout % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.GELU(),
    )


class NestedRGBDecoder(nn.Module):
    """Light UNet++-style decoder over the four RGB Swin feature levels."""

    def __init__(self, channels=(96, 192, 384, 768), width: int = 64):
        super().__init__()
        self.projections = nn.ModuleList([_conv_block(c, width) for c in channels])
        self.nodes = nn.ModuleDict(
            {
                f"{level}_{depth}": _conv_block((depth + 1) * width, width)
                for depth in range(1, 4)
                for level in range(4 - depth)
            }
        )

    def forward(self, pyramid):
        if len(pyramid) != 4:
            raise ValueError("Expected four RGB Swin feature levels")
        nodes = {(i, 0): projection(feature) for i, (projection, feature) in enumerate(zip(self.projections, pyramid))}
        for depth in range(1, 4):
            for level in range(4 - depth):
                up = F.interpolate(
                    nodes[level + 1, depth - 1],
                    size=nodes[level, 0].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                inputs = [nodes[level, previous] for previous in range(depth)] + [up]
                nodes[level, depth] = self.nodes[f"{level}_{depth}"](torch.cat(inputs, dim=1))
        return nodes[0, 3]


def _gather_centers(centers: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return centers.gather(1, index[..., None].expand(-1, -1, centers.shape[-1]))


class SwinLocalSuperpixelAttention(nn.Module):
    """Local pixel-to-seed cross-attention with iterative center refinement.

    Each Swin-grid location only considers its 3x3 neighboring seed cells.  The
    query/key affinity is combined with coordinate distance and a *soft*
    predicted-building compatibility prior.  GT never enters this module.
    """

    def __init__(
        self,
        in_dim: int = 64,
        attn_dim: int = 64,
        stride: int = 8,
        iterations: int = 3,
        temperature: float = 0.2,
        spatial_weight: float = 0.1,
        building_weight: float = 0.5,
    ):
        super().__init__()
        if stride < 1 or iterations < 1 or temperature <= 0:
            raise ValueError("Invalid superpixel configuration")
        if spatial_weight < 0 or building_weight < 0:
            raise ValueError("Prior weights must be non-negative")
        self.stride = int(stride)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.spatial_weight = float(spatial_weight)
        self.building_weight = float(building_weight)
        self.norm = nn.LayerNorm(in_dim)
        self.query = nn.Linear(in_dim, attn_dim, bias=False)
        self.key = nn.Linear(in_dim, attn_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(attn_dim))))

    def forward(self, embedding: torch.Tensor, building: torch.Tensor) -> Dict[str, torch.Tensor]:
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
                assignment = torch.stack(scores, dim=1).softmax(dim=1)

        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(1)
        entropy = (entropy / math.log(9.0)).reshape(batch, 1, height, width)
        return {
            "q": assignment,
            "ids": ids,
            "shape": (height, width),
            "grid_shape": (grid_h, grid_w),
            "count": count,
            "entropy_map": entropy,
        }


class StructureTokenAdapter(nn.Module):
    """Non-destructive feature-level structural injection with LayerScale."""

    def __init__(self, content_dim: int = 256, dim: int = 128):
        super().__init__()
        self.content = nn.Conv2d(content_dim, dim, 1)
        self.structure = nn.Conv2d(3, dim, 1)
        self.norm = nn.LayerNorm(2 * dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(
        self,
        feature: torch.Tensor,
        building: torch.Tensor,
        boundary: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        size = feature.shape[-2:]
        building = F.interpolate(building.detach().float(), size=size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary.float(), size=size, mode="bilinear", align_corners=False)
        entropy = F.interpolate(entropy.float(), size=size, mode="bilinear", align_corners=False)
        base = self.content(feature)
        structural = self.structure(torch.cat((building, boundary, entropy), dim=1))
        joined = torch.cat((base, structural), dim=1).flatten(2).transpose(1, 2)
        delta = self.mlp(self.norm(joined)) * self.layer_scale
        adapted = base.flatten(2).transpose(1, 2) + delta
        adapted = adapted.transpose(1, 2).reshape_as(base)
        return adapted, base


def soft_assignment_boundary(q: torch.Tensor, ids: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Differentiable probability that adjacent locations use different seeds."""
    batch, supports, pixels = q.shape
    height, width = shape
    if pixels != height * width or ids.shape != (supports, pixels):
        raise ValueError("Sparse association shape mismatch")
    probabilities = q.reshape(batch, supports, height, width)
    seed_ids = ids.reshape(supports, height, width)
    # Build every directional contribution out-of-place.  Updating overlapping
    # views of ``boundary`` in-place breaks autograd because the second max
    # reads storage already modified by the first assignment.
    contributions = [q.new_zeros(batch, 1, height, width)]
    if width > 1:
        horizontal_match = (seed_ids[:, None, :, :-1] == seed_ids[None, :, :, 1:]).to(q.dtype)
        horizontal_same = torch.einsum(
            "bshw,bthw,sthw->bhw",
            probabilities[:, :, :, :-1],
            probabilities[:, :, :, 1:],
            horizontal_match,
        )
        horizontal = (1 - horizontal_same).clamp(0, 1)
        horizontal = horizontal.unsqueeze(1)
        contributions.append(F.pad(horizontal, (0, 1, 0, 0)))
        contributions.append(F.pad(horizontal, (1, 0, 0, 0)))
    if height > 1:
        vertical_match = (seed_ids[:, None, :-1, :] == seed_ids[None, :, 1:, :]).to(q.dtype)
        vertical_same = torch.einsum(
            "bshw,bthw,sthw->bhw",
            probabilities[:, :, :-1, :],
            probabilities[:, :, 1:, :],
            vertical_match,
        )
        vertical = (1 - vertical_same).clamp(0, 1)
        vertical = vertical.unsqueeze(1)
        contributions.append(F.pad(vertical, (0, 0, 0, 1)))
        contributions.append(F.pad(vertical, (0, 0, 1, 0)))
    return torch.stack(contributions, dim=0).amax(dim=0)


def region_geometry(q: torch.Tensor, ids: torch.Tensor, shape: Tuple[int, int], count: int) -> torch.Tensor:
    """Predicted soft geometry: mass, centroid y/x and spread y/x."""
    height, width = shape
    batch = q.shape[0]
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height, device=q.device),
        torch.linspace(0, 1, width, device=q.device),
        indexing="ij",
    )
    values = torch.stack((yy, xx, yy.square(), xx.square()))[None].expand(batch, -1, -1, -1)
    pooled = soft_pool_unpool(values.float(), q, ids, count)
    mean = pooled["tokens"][..., :2]
    spread = (pooled["tokens"][..., 2:] - mean.square()).clamp_min(0).sqrt()
    return torch.cat((pooled["mass"][..., None] / float(height * width), mean, spread), dim=-1)


class RegionContextBlock(nn.Module):
    """Pre-LN MHSA over the compact superpixel token sequence."""

    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.scale1 = nn.Parameter(torch.full((dim,), 1e-3))
        self.scale2 = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.norm1(tokens)
        context, _ = self.attn(x, x, x, key_padding_mask=~valid, need_weights=False)
        tokens = tokens + self.scale1 * context
        tokens = tokens + self.scale2 * self.ffn(self.norm2(tokens))
        return tokens.masked_fill(~valid[..., None], 0)


class SuperpixelEmbedding(nn.Module):
    """Spatial attention pooling plus raw mean/std skip and token context."""

    def __init__(self, content_dim: int = 256, adapted_dim: int = 128, dim: int = 128, region_context: bool = False):
        super().__init__()
        self.score = nn.Conv2d(adapted_dim, 1, 1)
        self.main = nn.Sequential(nn.LayerNorm(adapted_dim), nn.Linear(adapted_dim, dim), nn.GELU())
        self.stats = nn.Sequential(nn.LayerNorm(2 * content_dim), nn.Linear(2 * content_dim, dim))
        self.geometry = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.skip_scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(dim)
        # Off in the primary experiment: enabling this is a separate ablation.
        self.context = RegionContextBlock(dim) if region_context else None

    def forward(self, adapted, raw, q, ids, shape, count):
        spatial_weight = self.score(adapted).float().sigmoid().flatten(2)
        main = soft_pool_unpool(adapted.float(), q * spatial_weight, ids, count)
        stats = soft_pool_unpool(torch.cat((raw.float(), raw.float().square()), dim=1), q, ids, count)
        mean, second = stats["tokens"].chunk(2, dim=-1)
        std = (second - mean.square()).clamp_min(0).add(1e-6).sqrt()
        geometry = region_geometry(q, ids, shape, count)
        tokens = self.main(main["tokens"])
        tokens = tokens + self.skip_scale.tanh() * self.stats(torch.cat((mean, std), dim=-1))
        tokens = self.norm(tokens + self.geometry(geometry))
        valid = stats["mass"] > 1e-6
        if self.context is not None:
            tokens = self.context(tokens, valid)
        else:
            tokens = tokens.masked_fill(~valid[..., None], 0)
        return tokens, stats["mass"], geometry, valid


def project_region_probabilities(logits, q, ids, shape, output_size):
    """Project normalized region probabilities through the same association Q."""
    batch, _, classes = logits.shape
    probability = logits.float().softmax(dim=-1)
    pixels = q.new_zeros(batch, q.shape[-1], classes)
    if ids.shape[0] != q.shape[1]:
        raise ValueError("Q and candidate IDs disagree")
    for candidate in range(q.shape[1]):
        gathered = probability.gather(1, ids[candidate][None, :, None].expand(batch, -1, classes))
        pixels = pixels + gathered * q[:, candidate, :, None]
    pixels = pixels.transpose(1, 2).reshape(batch, classes, *shape)
    pixels = F.interpolate(pixels, size=output_size, mode="bilinear", align_corners=False)
    return pixels / pixels.sum(1, keepdim=True).clamp_min(1e-8)


class SRACTransformerFront(nn.Module):
    """Four-stage front with a frozen, unchanged DamageFormer/A1 backbone."""

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
    ):
        super().__init__()
        self.config = dict(
            dim=dim,
            decoder_width=decoder_width,
            stride=stride,
            iterations=iterations,
            temperature=temperature,
            spatial_weight=spatial_weight,
            building_weight=building_weight,
            region_context=region_context,
        )
        self.base = JointModel("t0").base
        self.base.requires_grad_(False)
        self.decoder = NestedRGBDecoder(width=decoder_width)
        self.partition = SwinLocalSuperpixelAttention(
            decoder_width,
            decoder_width,
            stride,
            iterations,
            temperature,
            spatial_weight,
            building_weight,
        )
        self.adapter = StructureTokenAdapter(256, dim)
        self.embedding = SuperpixelEmbedding(256, dim, dim, region_context=region_context)
        self.region_classifier = nn.Linear(dim, 4)
        self._captured_rgb_pyramid = None
        self.base.encoder_1.register_forward_hook(self._capture_rgb_pyramid)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen weights are not enough: keep inherited running buffers frozen too.
        self.base.eval()
        return self

    def adapter_state_dict(self):
        return {name: value for name, value in self.state_dict().items() if not name.startswith("base.")}

    def load_adapter_state_dict(self, state):
        expected = {name for name in self.state_dict() if not name.startswith("base.")}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise ValueError(f"Adapter checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
        result = self.load_state_dict(state, strict=False)
        # PyTorch intentionally does not report missing BatchNorm
        # ``num_batches_tracked`` buffers for backward compatibility.  The
        # exact adapter-key equality above is the strict contract we need;
        # here only reject missing *non-base* keys or unexpected keys.
        missing_adapter = [name for name in result.missing_keys if not name.startswith("base.")]
        if result.unexpected_keys or missing_adapter:
            raise ValueError(
                "Adapter-only strict load failed; "
                f"missing_adapter={missing_adapter[:5]}, unexpected={result.unexpected_keys[:5]}"
            )

    def _capture_rgb_pyramid(self, _module, _inputs, output):
        self._captured_rgb_pyramid = output

    def _a1_features(self, pre: torch.Tensor, post: torch.Tensor):
        # One exact A1 forward; the hook exposes its already-computed RGB pyramid.
        self._captured_rgb_pyramid = None
        loc, pixel, auxiliary = self.base(pre, post, return_aux=True)
        if self._captured_rgb_pyramid is None:
            raise RuntimeError("RGB Swin feature hook did not run")
        rgb_pyramid = [self.base._to_nchw(feature) for feature in self._captured_rgb_pyramid[1:]]
        self._captured_rgb_pyramid = None
        return rgb_pyramid, loc, pixel, auxiliary["p_building"], auxiliary["fused_roi"]

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb_pyramid, loc, old_pixel, building, fused_roi = self._a1_features(pre, post)
        structural = self.decoder(rgb_pyramid)
        partition = self.partition(structural, building)
        boundary = soft_assignment_boundary(partition["q"], partition["ids"], partition["shape"])
        adapted, content_only = self.adapter(fused_roi, building, boundary, partition["entropy_map"])
        tokens, mass, geometry, valid = self.embedding(
            adapted,
            fused_roi,
            partition["q"],
            partition["ids"],
            partition["shape"],
            partition["count"],
        )
        region_logits = self.region_classifier(tokens)
        probability = project_region_probabilities(
            region_logits,
            partition["q"],
            partition["ids"],
            partition["shape"],
            pre.shape[-2:],
        )
        return {
            **partition,
            "tokens": tokens,
            "mass": mass,
            "geometry": geometry,
            "valid_regions": valid,
            "region_logits": region_logits,
            "prob": probability,
            "loc": loc,
            "old_pixel": old_pixel,
            "soft_boundary": boundary,
            "content_only": content_only,
            "adapted_content": adapted,
            "assignment_entropy": partition["entropy_map"].mean(),
        }

