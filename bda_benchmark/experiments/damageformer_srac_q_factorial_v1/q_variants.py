"""Isolated, matched Q variants on the archived, frozen SRAC A1 front.

Only the three cue arms change the partition module.  The other arms call the
original front forward unchanged; their intervention is a Train-only loss.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map
from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import grouped_centers
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    digest, load_a1_strict, source_hash as inherited_source_hash,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    SRACTransformerFront, SwinLocalSuperpixelAttention, _gather_centers,
    project_region_probabilities, soft_assignment_boundary,
)


FORMAT = "SRAC_Q_FACTORIAL_V1"
CUE_ARMS = frozenset(("sar_cue", "rgb_capacity", "shifted_sar"))
ARMS = frozenset(("baseline", "building_reconstruction", "semantic_reconstruction", *CUE_ARMS))


def q_source_hash() -> str:
    """Bind new implementation and all inherited original-front dependencies."""
    checksum = hashlib.sha256(inherited_source_hash().encode("ascii"))
    root = Path(__file__).parent
    for name in ("q_variants.py", "q_losses.py", "train_q.py"):
        checksum.update(name.encode("ascii"))
        checksum.update((root / name).read_bytes())
    return checksum.hexdigest()


class CuePartition(SwinLocalSuperpixelAttention):
    """Original local Q with one bounded, matched-parameter cue score.

    The 512->D projection has identical capacity in aligned SAR, RGB-only and
    shifted-SAR controls.  Every inherited score term and candidate set is
    copied from the archived partition; GT never enters the score.
    """

    def __init__(self, *args, cue_gain: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(cue_gain) or not 0 < cue_gain <= 0.25:
            raise ValueError("cue_gain must be finite in (0, 0.25]")
        self.cue_gain = float(cue_gain)
        self.cue_proj = nn.Conv2d(512, self.query.out_features, 1, bias=False)

    def forward(self, embedding: torch.Tensor, building: torch.Tensor,
                cue: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _channels, height, width = embedding.shape
        if cue.ndim != 4 or cue.shape[:2] != (batch, 512):
            raise ValueError("Expected aligned RGB/SAR cue with 512 channels")
        ids, valid, grid_h, grid_w = candidate_map(height, width, self.stride, embedding.device)
        count = grid_h * grid_w
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            pixels = embedding.float().flatten(2).transpose(1, 2)
            pixels = self.norm(pixels)
            query = F.normalize(self.query(pixels), dim=-1)
            values = F.normalize(pixels, dim=-1)

            cue = F.interpolate(cue.detach().float(), size=(height, width), mode="bilinear", align_corners=False)
            cue_values = F.normalize(self.cue_proj(cue).flatten(2).transpose(1, 2), dim=-1)

            yy, xx = torch.meshgrid(
                torch.arange(height, device=embedding.device),
                torch.arange(width, device=embedding.device), indexing="ij",
            )
            coordinates = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2).float()
            coordinates = coordinates.div(float(self.stride)).expand(batch, -1, -1)
            building_grid = F.interpolate(building.detach().float(), (height, width),
                                          mode="bilinear", align_corners=False)
            building_flat = building_grid.flatten(2).transpose(1, 2)

            index = ids[None].expand(batch, -1, -1)
            assignment = embedding.new_zeros(batch, 9, height * width, dtype=torch.float32)
            assignment[:, 4] = 1.0
            scale = self.logit_scale.exp().clamp(1e-3, 100.0)

            for _ in range(self.iterations):
                center_value, mass = grouped_centers(values, assignment, index, count)
                center_xy, _ = grouped_centers(coordinates, assignment, index, count)
                center_building, _ = grouped_centers(building_flat, assignment, index, count)
                center_cue, _ = grouped_centers(cue_values, assignment, index, count)
                scores = []
                base_magnitudes = []
                cue_magnitudes = []
                for candidate in range(9):
                    candidate_index = index[:, candidate]
                    center_key = F.normalize(self.key(_gather_centers(center_value, candidate_index)), dim=-1)
                    similarity = (query * center_key).sum(-1) * scale
                    distance = (coordinates - _gather_centers(center_xy, candidate_index)).square().sum(-1)
                    center_b = _gather_centers(center_building, candidate_index).squeeze(-1).clamp(1e-4, 1 - 1e-4)
                    pixel_b = building_flat.squeeze(-1)
                    compatibility = (pixel_b * center_b + (1 - pixel_b) * (1 - center_b)).clamp_min(1e-4)
                    base_similarity = similarity / self.temperature
                    score = base_similarity
                    score = score - self.spatial_weight * distance
                    score = score + self.building_weight * compatibility.log()
                    cue_center = F.normalize(_gather_centers(center_cue, candidate_index), dim=-1)
                    cue_score = self.cue_gain * (cue_values * cue_center).sum(-1)
                    score = score + cue_score
                    base_magnitudes.append(base_similarity.detach().abs().mean())
                    cue_magnitudes.append(cue_score.detach().abs().mean())
                    occupied = mass.gather(1, candidate_index) > 1e-6
                    score = score.masked_fill(~(valid[candidate][None] & occupied), -torch.inf)
                    scores.append(score)
                assignment = torch.stack(scores, dim=1).softmax(dim=1)

        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(1)
        entropy = (entropy / math.log(9.0)).reshape(batch, 1, height, width)
        return dict(q=assignment, ids=ids, shape=(height, width),
                    grid_shape=(grid_h, grid_w), count=count, entropy_map=entropy,
                    base_score_mean_abs=torch.stack(base_magnitudes).mean(),
                    cue_score_mean_abs=torch.stack(cue_magnitudes).mean())


class QVariantFront(SRACTransformerFront):
    """Same frozen A1 and downstream region path; selectable Q intervention."""

    def __init__(self, arm: str = "baseline", cue_gain: float = 0.1, **front_config):
        if arm not in ARMS:
            raise ValueError(f"Unknown Q arm: {arm}")
        super().__init__(**front_config)
        self.arm = arm
        self.cue_gain = float(cue_gain)
        if arm in CUE_ARMS:
            old = self.partition
            new = CuePartition(
                in_dim=front_config.get("decoder_width", 64),
                attn_dim=front_config.get("decoder_width", 64),
                stride=old.stride, iterations=old.iterations,
                temperature=old.temperature, spatial_weight=old.spatial_weight,
                building_weight=old.building_weight, cue_gain=cue_gain,
            )
            # Inherited Q weights are identical at initialization across arms.
            state = new.state_dict()
            state.update(old.state_dict())
            new.load_state_dict(state, strict=True)
            self.partition = new

    def load_original_adapter(self, state: dict) -> None:
        """Load one common completed original-front checkpoint into every arm."""
        expected = set(self.adapter_state_dict())
        additional = expected - set(state)
        required_additional = {"partition.cue_proj.weight"} if self.arm in CUE_ARMS else set()
        if additional != required_additional or set(state) - expected:
            raise ValueError("Original front is incompatible with selected Q arm")
        result = self.load_state_dict(state, strict=False)
        missing_adapter = {name for name in result.missing_keys if not name.startswith("base.")}
        if missing_adapter != required_additional or result.unexpected_keys:
            raise ValueError("Original front adapter load failed")

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.arm not in CUE_ARMS:
            return super().forward(pre, post)

        # Mirror the archived SRAC forward exactly, supplying only the new cue
        # to the otherwise unchanged region, projection and output path.
        self._captured_rgb_pyramid = None
        loc, old_pixel, auxiliary = self.base(pre, post, return_aux=True)
        if self._captured_rgb_pyramid is None:
            raise RuntimeError("RGB Swin feature hook did not run")
        rgb_pyramid = [self.base._to_nchw(feature) for feature in self._captured_rgb_pyramid[1:]]
        self._captured_rgb_pyramid = None
        building, fused_roi = auxiliary["p_building"], auxiliary["fused_roi"]
        structural = self.decoder(rgb_pyramid)
        rgb, sar = auxiliary["rgb_feature"], auxiliary["sar_feature"]
        if self.arm == "rgb_capacity":
            sar = rgb
        elif self.arm == "shifted_sar":
            sar = torch.roll(sar, shifts=(max(1, sar.shape[-2] // 2), max(1, sar.shape[-1] // 2)),
                             dims=(-2, -1))
        partition = self.partition(structural, building, torch.cat((rgb, sar), dim=1))
        boundary = soft_assignment_boundary(partition["q"], partition["ids"], partition["shape"])
        adapted, content_only = self.adapter(fused_roi, building, boundary, partition["entropy_map"])
        tokens, mass, geometry, valid = self.embedding(
            adapted, fused_roi, partition["q"], partition["ids"], partition["shape"], partition["count"])
        region_logits = self.region_classifier(tokens)
        probability = project_region_probabilities(
            region_logits, partition["q"], partition["ids"], partition["shape"], pre.shape[-2:])
        return dict(**partition, tokens=tokens, mass=mass, geometry=geometry,
                    valid_regions=valid, region_logits=region_logits, prob=probability,
                    loc=loc, old_pixel=old_pixel, soft_boundary=boundary,
                    content_only=content_only, adapted_content=adapted,
                    assignment_entropy=partition["entropy_map"].mean())


def load_q_checkpoint(path, device, a1_override=None, allow_smoke=False):
    """Load a completed adapter-only Q checkpoint for read-only evaluation."""
    if Path(path).name != "last.pth":
        raise ValueError("Formal Q comparison uses the matched final-step last.pth, never best.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != FORMAT or checkpoint.get("run_complete") is not True:
        raise ValueError("Expected a completed Q-factorial checkpoint")
    if checkpoint.get("completed_steps") != checkpoint.get("arguments", {}).get("steps"):
        raise ValueError("Q training did not complete its declared budget")
    if (checkpoint.get("step") != checkpoint["completed_steps"]
            or type(checkpoint.get("smoke")) is not bool
            or (checkpoint["smoke"] and not allow_smoke)):
        raise ValueError("Q checkpoint is not an allowed final-step weight")
    if (checkpoint.get("a1_unchanged") is not True
            or checkpoint.get("downstream_unchanged") is not True
            or checkpoint.get("immutable", {}).get("frozen_downstream") is not True):
        raise ValueError("Q checkpoint did not preserve its frozen model paths")
    if checkpoint.get("source_hash") != q_source_hash() or checkpoint.get("inherited_source_hash") != inherited_source_hash():
        raise ValueError("Q or inherited SRAC code differs from trained source")
    a1_path = Path(a1_override or checkpoint["a1_checkpoint"])
    if not a1_path.is_file() or digest(a1_path) != checkpoint["a1_sha256"]:
        raise ValueError("The frozen A1 file required by this Q checkpoint is unavailable")
    front_path = Path(checkpoint["front_checkpoint"])
    if not front_path.is_file() or digest(front_path) != checkpoint["front_sha256"]:
        raise ValueError("The common initial front checkpoint changed")
    model = QVariantFront(arm=checkpoint["arm"], cue_gain=checkpoint["cue_gain"],
                          **checkpoint["config"]).to(device)
    load_a1_strict(model, a1_path)
    model.load_adapter_state_dict(checkpoint["adapter"])
    model.base.requires_grad_(False).eval()
    return model, checkpoint
