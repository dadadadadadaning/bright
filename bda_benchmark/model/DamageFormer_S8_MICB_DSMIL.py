# -*- coding: utf-8 -*-
"""S8/P3: multi-instance cross-modal building pooling for BRIGHT.

This module deliberately keeps the proven S3V2 pixel decoder and adds an
auxiliary DSMIL-style building head.  A building is a *bag* and feature-grid
locations inside the building are its *instances*.  Consequently, a Damaged
building label supervises the bag output, never every local token.

Default P3 behaviour:
    * no prototype/cache memory;
    * no building-logit broadcast at inference (``mil_residual_scale=0``);
    * RGB, SAR, cross-modal difference/product, and fused features are kept;
    * global, critical, soft-top, attention-bag and coverage statistics are
      used by hierarchical DamageExist/Severity heads;
    * RGB-core/SAR-ring asymmetric pooling is an optional later P5 ablation.

The optional residual is included only for a later, validation-locked
ablation.  Leave it at zero for the first scientific comparison.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .DamageFormer_S3_BGMP_K3 import DamageFormer_S3_BGMP_K3
else:  # Preserve direct-file smoke tests from the benchmark root.
    from model.DamageFormer_S3_BGMP_K3 import DamageFormer_S3_BGMP_K3


@dataclass
class BuildingBagSpec:
    """One building region.

    ``mask`` may be at image or feature resolution and is kept on CPU until
    pooling. ``label`` uses BRIGHT values 1/2/3, or -1 when unavailable.
    """

    batch_index: int
    mask: torch.Tensor
    label: int = -1
    area: int = 0


def _fallback_connected_components(
    mask: np.ndarray,
    minimum_area: int = 1,
) -> List[np.ndarray]:
    """Dependency-free 4-connected components."""

    mask = np.asarray(mask, dtype=np.bool_)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    output: List[np.ndarray] = []
    for y0, x0 in np.argwhere(mask):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        queue: deque[Tuple[int, int]] = deque([(int(y0), int(x0))])
        ys: List[int] = []
        xs: List[int] = []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= yy < height
                    and 0 <= xx < width
                    and mask[yy, xx]
                    and not visited[yy, xx]
                ):
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        if len(ys) >= int(minimum_area):
            component = np.zeros_like(mask, dtype=np.bool_)
            component[np.asarray(ys), np.asarray(xs)] = True
            output.append(component)
    return output


def connected_components_4(
    mask: torch.Tensor | np.ndarray,
    minimum_area: int = 1,
) -> List[np.ndarray]:
    """Return CPU boolean component masks using 4-connectivity.

    Filtering component IDs before allocating dense HxW masks avoids a severe
    cost on fragmented predictions.  The default preserves historical output.
    """

    if isinstance(mask, torch.Tensor):
        array = mask.detach().to(device="cpu").numpy().astype(np.bool_, copy=False)
    else:
        array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got {array.shape}")
    if int(minimum_area) < 1:
        raise ValueError("minimum_area must be positive")
    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(
            array,
            structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        )
        areas = np.bincount(labels.reshape(-1), minlength=int(count) + 1)
        kept = np.flatnonzero(areas[1:] >= int(minimum_area)) + 1
        return [labels == int(index) for index in kept.tolist()]
    except ImportError:
        return _fallback_connected_components(array, int(minimum_area))


def build_training_bag_specs(
    labels_loc: torch.Tensor,
    labels_clf: torch.Tensor,
    min_area: int = 32,
    max_area: Optional[int] = None,
) -> List[BuildingBagSpec]:
    """Build training bags without assigning the bag label to local tokens.

    Components are extracted separately for BRIGHT labels 1/2/3.  This avoids
    merging two touching buildings with different damage labels.  It does not
    magically separate touching buildings of the same class; if an instance-id
    annotation becomes available, pass those masks directly as ``bag_specs``.
    """

    if labels_loc.ndim == 4 and labels_loc.shape[1] == 1:
        labels_loc = labels_loc[:, 0]
    if labels_clf.ndim == 4 and labels_clf.shape[1] == 1:
        labels_clf = labels_clf[:, 0]
    if labels_loc.ndim != 3 or labels_clf.ndim != 3:
        raise ValueError("labels_loc and labels_clf must have shape [B,H,W]")
    if labels_loc.shape != labels_clf.shape:
        raise ValueError("localization and classification targets must share a shape")
    if min_area <= 0:
        raise ValueError("min_area must be positive")

    loc_cpu = labels_loc.detach().to(device="cpu")
    clf_cpu = labels_clf.detach().to(device="cpu")
    output: List[BuildingBagSpec] = []
    for batch_index in range(loc_cpu.shape[0]):
        valid_building = (loc_cpu[batch_index] > 0) & (loc_cpu[batch_index] != 255)
        for class_label in (1, 2, 3):
            semantic_mask = valid_building & (clf_cpu[batch_index] == class_label)
            for component in connected_components_4(semantic_mask):
                area = int(component.sum())
                if area < min_area or (max_area is not None and area > max_area):
                    continue
                output.append(
                    BuildingBagSpec(
                        batch_index=batch_index,
                        mask=torch.from_numpy(component.copy()),
                        label=class_label,
                        area=area,
                    )
                )
    return output


def build_predicted_bag_specs(
    loc_logits_feature: torch.Tensor,
    min_area_feature: int = 2,
) -> List[BuildingBagSpec]:
    """Build inference bags only from localization predictions (no GT leakage)."""

    predicted = loc_logits_feature.detach().argmax(dim=1).to(device="cpu") > 0
    output: List[BuildingBagSpec] = []
    for batch_index in range(predicted.shape[0]):
        for component in connected_components_4(predicted[batch_index]):
            area = int(component.sum())
            if area < min_area_feature:
                continue
            output.append(
                BuildingBagSpec(
                    batch_index=batch_index,
                    mask=torch.from_numpy(component.copy()),
                    label=-1,
                    area=area,
                )
            )
    return output


def hierarchical_log_probs(
    exist_logits: torch.Tensor,
    severity_logits: torch.Tensor,
) -> torch.Tensor:
    """Convert DamageExist and conditional Severity logits to 3-class log-P.

    Output order is Intact, Damaged, Destroyed.  Exponentiating each row sums
    to one, so the result can be passed directly to ``F.nll_loss``.
    """

    if exist_logits.ndim != 2 or exist_logits.shape[1] != 2:
        raise ValueError("exist_logits must be [N,2]")
    if severity_logits.shape != exist_logits.shape:
        raise ValueError("severity_logits must have the same [N,2] shape")
    log_exist = F.log_softmax(exist_logits, dim=1)
    log_severity = F.log_softmax(severity_logits, dim=1)
    return torch.stack(
        (
            log_exist[:, 0],
            log_exist[:, 1] + log_severity[:, 0],
            log_exist[:, 1] + log_severity[:, 1],
        ),
        dim=1,
    )


class MICBDSMILHead(nn.Module):
    """Multi-Instance Cross-Modal Building pooling with a DSMIL-style stream."""

    def __init__(
        self,
        in_channels: int = 256,
        token_dim: int = 128,
        topk: int = 8,
        max_tokens: int = 256,
        attention_temperature: float = 0.2,
        mask_threshold: float = 0.05,
        ring_kernel: int = 5,
        use_asymmetric_regions: bool = False,
    ):
        super().__init__()
        if token_dim <= 0 or topk <= 0 or max_tokens <= 0:
            raise ValueError("token_dim, topk and max_tokens must be positive")
        if token_dim % 8 != 0:
            raise ValueError("token_dim must be divisible by 8 for GroupNorm")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if ring_kernel < 3 or ring_kernel % 2 == 0:
            raise ValueError("ring_kernel must be odd and at least 3")

        self.token_dim = int(token_dim)
        self.topk = int(topk)
        self.max_tokens = int(max_tokens)
        self.attention_temperature = float(attention_temperature)
        self.mask_threshold = float(mask_threshold)
        self.ring_kernel = int(ring_kernel)
        self.use_asymmetric_regions = bool(use_asymmetric_regions)

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, token_dim, 1, bias=False),
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
        )
        self.sar_proj = nn.Sequential(
            nn.Conv2d(in_channels, token_dim, 1, bias=False),
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
        )
        self.fused_proj = nn.Sequential(
            nn.Conv2d(in_channels, token_dim, 1, bias=False),
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
        )
        # The five terms retain modality identity and projected change evidence.
        self.cross_token = nn.Sequential(
            nn.Conv2d(token_dim * 5, token_dim, 1, bias=False),
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
        )

        self.local_exist = nn.Linear(token_dim, 1)
        self.local_severity = nn.Linear(token_dim, 2)
        self.query = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)

        # Exist uses global + critical + soft-top + DSMIL attention bag.
        self.exist_head = nn.Sequential(
            nn.Linear(token_dim * 4 + 1, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim, 2),
        )
        # Severity adds asymmetric RGB-core/SAR-ring and a coverage scalar.
        self.severity_head = nn.Sequential(
            nn.Linear(token_dim * 5 + 3, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim, 2),
        )

    @staticmethod
    def _weighted_pool(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        denom = weight.sum().clamp_min(1e-6)
        return (feature * weight.unsqueeze(0)).sum(dim=(1, 2)) / denom

    def _aggregate_occurrence_critical(
        self,
        tokens: torch.Tensor,
        local_exist_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the occurrence-critical token and scalar evidence.

        P3 deliberately keeps the historical hard DSMIL choice.  The method
        is a narrow extension point for isolated experiments; overriding it
        must not change parameter names or the default checkpoint contract.
        """

        critical_index = local_exist_logits.argmax()
        critical = tokens[critical_index]
        local_evidence = local_exist_logits.max()[None]
        return critical, local_evidence

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        tensor = mask.detach().to(device=device, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"bag mask must be [H,W], got {tuple(tensor.shape)}")
        return F.interpolate(tensor[None, None], size=size, mode="area")[0, 0].clamp_(0, 1)

    def _region_weights(self, full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        binary = (full > self.mask_threshold).to(dtype=full.dtype)[None, None]
        # Erosion through complement max-pooling; tiny cores fall back to full.
        core = 1.0 - F.max_pool2d(1.0 - binary, kernel_size=3, stride=1, padding=1)
        expanded = F.max_pool2d(
            binary,
            kernel_size=self.ring_kernel,
            stride=1,
            padding=self.ring_kernel // 2,
        )
        ring = (expanded - binary).clamp_min(0)[0, 0]
        core = core[0, 0]
        if float(core.sum().item()) <= 1e-6:
            core = full
        if float(ring.sum().item()) <= 1e-6:
            ring = full
        return core, ring

    def _select_tokens(
        self,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        change_energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep half high-change and half spatially spread tokens when necessary."""

        indices = torch.nonzero(weights > self.mask_threshold, as_tuple=False)
        if indices.shape[0] == 0:
            indices = torch.nonzero(weights > 0, as_tuple=False)
        if indices.shape[0] == 0:
            raise RuntimeError("a non-empty component vanished at feature resolution")

        flat = indices[:, 0] * weights.shape[1] + indices[:, 1]
        flat_tokens = tokens.permute(1, 2, 0).reshape(-1, self.token_dim)[flat]
        flat_energy = change_energy.reshape(-1)[flat]
        if flat_tokens.shape[0] <= self.max_tokens:
            return flat_tokens, flat

        high_count = self.max_tokens // 2
        high = torch.topk(flat_energy, k=high_count, largest=True, sorted=False).indices
        spread_count = self.max_tokens - high_count
        spread = torch.linspace(
            0,
            flat_tokens.shape[0] - 1,
            steps=spread_count,
            device=flat_tokens.device,
        ).round().long()
        chosen = torch.unique(torch.cat((high, spread), dim=0), sorted=False)
        if chosen.numel() < self.max_tokens:
            remaining = torch.ones(flat_tokens.shape[0], device=flat_tokens.device, dtype=torch.bool)
            remaining[chosen] = False
            fill = torch.nonzero(remaining, as_tuple=False)[: self.max_tokens - chosen.numel(), 0]
            chosen = torch.cat((chosen, fill), dim=0)
        return flat_tokens[chosen[: self.max_tokens]], flat[chosen[: self.max_tokens]]

    def forward(
        self,
        rgb_feature: torch.Tensor,
        sar_feature: torch.Tensor,
        fused_feature: torch.Tensor,
        bag_specs: Sequence[BuildingBagSpec],
    ) -> Dict[str, object]:
        if not (rgb_feature.shape == sar_feature.shape == fused_feature.shape):
            raise ValueError("RGB, SAR and fused feature maps must share [B,C,h,w]")

        rgb = self.rgb_proj(rgb_feature)
        sar = self.sar_proj(sar_feature)
        fused = self.fused_proj(fused_feature)
        token_map = self.cross_token(
            torch.cat((rgb, sar, torch.abs(rgb - sar), rgb * sar, fused), dim=1)
        )
        change_energy = torch.abs(rgb - sar).mean(dim=1)
        size = token_map.shape[-2:]

        exist_outputs: List[torch.Tensor] = []
        severity_outputs: List[torch.Tensor] = []
        labels: List[int] = []
        coverages: List[torch.Tensor] = []
        attention_sums: List[torch.Tensor] = []
        intact_guard: List[torch.Tensor] = []
        local_records: List[Dict[str, torch.Tensor | int]] = []

        for spec in bag_specs:
            if not 0 <= int(spec.batch_index) < token_map.shape[0]:
                raise ValueError(f"invalid bag batch index {spec.batch_index}")
            batch_index = int(spec.batch_index)
            full = self._resize_mask(spec.mask, size, token_map.device)
            if float(full.sum().item()) <= 1e-6:
                continue
            try:
                tokens, flat_indices = self._select_tokens(
                    token_map[batch_index], full, change_energy[batch_index]
                )
            except RuntimeError:
                continue

            local_exist_logits = self.local_exist(tokens)[:, 0]
            local_severity_logits = self.local_severity(tokens)
            critical, local_evidence = self._aggregate_occurrence_critical(
                tokens,
                local_exist_logits,
            )

            count = min(self.topk, tokens.shape[0])
            top_indices = torch.topk(local_exist_logits, k=count, largest=True).indices
            soft_top = tokens[top_indices].mean(dim=0)
            global_token = self._weighted_pool(token_map[batch_index], full)

            # Differentiable severity evidence. Unlike assigning the bag class
            # to each token, only the pooled local evidence receives bag loss.
            severity_confidence = local_severity_logits.amax(dim=1)
            severity_attention = torch.softmax(
                severity_confidence / self.attention_temperature,
                dim=0,
            )
            severity_context = (severity_attention[:, None] * tokens).sum(dim=0)
            severity_peak = local_severity_logits.amax(dim=0)

            queries = F.normalize(self.query(tokens), dim=1)
            critical_query = F.normalize(self.query(critical[None]), dim=1)[0]
            attention = torch.softmax(
                (queries @ critical_query) / self.attention_temperature,
                dim=0,
            )
            attention_bag = (attention[:, None] * self.value(tokens)).sum(dim=0)
            coverage = torch.sigmoid(local_exist_logits).mean()
            if self.use_asymmetric_regions:
                core, ring = self._region_weights(full)
                rgb_region = self._weighted_pool(rgb[batch_index], core)
                sar_region = self._weighted_pool(sar[batch_index], ring)
            else:
                # Pure P3 control: both modalities use the full building mask.
                rgb_region = self._weighted_pool(rgb[batch_index], full)
                sar_region = self._weighted_pool(sar[batch_index], full)

            exist_input = torch.cat(
                (
                    global_token,
                    critical,
                    soft_top,
                    attention_bag,
                    local_evidence,
                ),
                dim=0,
            )
            severity_input = torch.cat(
                (
                    global_token,
                    attention_bag,
                    severity_context,
                    rgb_region,
                    sar_region,
                    coverage[None],
                    severity_peak,
                ),
                dim=0,
            )
            exist_outputs.append(self.exist_head(exist_input))
            severity_outputs.append(self.severity_head(severity_input))
            labels.append(int(spec.label))
            coverages.append(coverage)
            attention_sums.append(attention.sum())
            if int(spec.label) == 1:
                intact_guard.append(local_exist_logits)
            local_records.append(
                {
                    "batch_index": batch_index,
                    "flat_indices": flat_indices,
                    "local_exist_logits": local_exist_logits.detach(),
                    "local_severity_logits": local_severity_logits.detach(),
                }
            )

        if exist_outputs:
            exist_logits = torch.stack(exist_outputs, dim=0)
            severity_logits = torch.stack(severity_outputs, dim=0)
            label_tensor = torch.tensor(labels, device=token_map.device, dtype=torch.long)
            coverage_tensor = torch.stack(coverages)
            attention_tensor = torch.stack(attention_sums)
        else:
            exist_logits = token_map.new_empty((0, 2))
            severity_logits = token_map.new_empty((0, 2))
            label_tensor = torch.empty((0,), device=token_map.device, dtype=torch.long)
            coverage_tensor = token_map.new_empty((0,))
            attention_tensor = token_map.new_empty((0,))

        guard_logits = (
            torch.cat(intact_guard, dim=0) if intact_guard else token_map.new_empty((0,))
        )
        return {
            "exist_logits": exist_logits,
            "severity_logits": severity_logits,
            "log_probs3": hierarchical_log_probs(exist_logits, severity_logits),
            "labels": label_tensor,
            "coverage": coverage_tensor,
            "attention_sum": attention_tensor,
            "intact_guard_logits": guard_logits,
            "local_records": local_records,
            "bag_count": int(exist_logits.shape[0]),
        }


class DamageFormer_S8_MICB_DSMIL(DamageFormer_S3_BGMP_K3):
    """S3V2-compatible pixel model plus the P3 MICB-DSMIL auxiliary head."""

    def __init__(
        self,
        roi_alpha: float = 0.1,
        detach_building_prior: bool = True,
        token_dim: int = 128,
        mil_topk: int = 8,
        max_tokens_per_bag: int = 256,
        mil_residual_scale: float = 0.0,
        predicted_min_area_feature: int = 2,
        use_asymmetric_regions: bool = False,
    ):
        # A base checkpoint is required by the trainer, so do not download Swin.
        super().__init__(
            roi_alpha=roi_alpha,
            detach_building_prior=detach_building_prior,
            use_pretrained_swin=False,
        )
        if mil_residual_scale < 0:
            raise ValueError("mil_residual_scale must be non-negative")
        self.micb_dsmil = MICBDSMILHead(
            in_channels=256,
            token_dim=token_dim,
            topk=mil_topk,
            max_tokens=max_tokens_per_bag,
            use_asymmetric_regions=use_asymmetric_regions,
        )
        self.mil_residual_scale = float(mil_residual_scale)
        self.predicted_min_area_feature = int(predicted_min_area_feature)

    @staticmethod
    def _scatter_bag_residual(
        bag_result: Dict[str, object],
        bag_specs: Sequence[BuildingBagSpec],
        output_size: Tuple[int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Optional building-level residual; disabled in the P3 default."""

        residual = torch.zeros((batch_size, 3, *output_size), dtype=dtype, device=device)
        counts = torch.zeros((batch_size, 1, *output_size), dtype=dtype, device=device)
        log_probs = bag_result["log_probs3"]
        if not isinstance(log_probs, torch.Tensor):
            raise TypeError("bag_result['log_probs3'] must be a tensor")
        valid_index = 0
        for spec in bag_specs:
            if valid_index >= log_probs.shape[0]:
                break
            mask = MICBDSMILHead._resize_mask(spec.mask, output_size, device)
            if float(mask.sum().item()) <= 1e-6:
                continue
            centered = log_probs[valid_index] - log_probs[valid_index].mean()
            residual[int(spec.batch_index)] += centered[:, None, None].to(dtype) * mask
            counts[int(spec.batch_index)] += mask
            valid_index += 1
        return residual / counts.clamp_min(1.0)

    def forward(
        self,
        pre_data: torch.Tensor,
        post_data: torch.Tensor,
        bag_specs: Optional[Sequence[BuildingBagSpec]] = None,
        return_aux: bool = False,
    ):
        rgb_feature, sar_feature = self._encode_and_fuse(pre_data, post_data)

        loc_logits_feature = self.clf_1(rgb_feature)
        p_building = torch.softmax(loc_logits_feature, dim=1)[:, 1:2]
        building_prior = p_building.detach() if self.detach_building_prior else p_building
        roi_weight = self.roi_alpha + (1.0 - self.roi_alpha) * building_prior

        fused_feature = self.fusion_layer_3(torch.cat((rgb_feature, sar_feature), dim=1))
        fused_roi = fused_feature * roi_weight
        damage_logits_feature = self.clf_2(fused_roi)

        active_specs: Sequence[BuildingBagSpec] = bag_specs or []
        if not active_specs and self.mil_residual_scale > 0:
            active_specs = build_predicted_bag_specs(
                loc_logits_feature,
                min_area_feature=self.predicted_min_area_feature,
            )

        bag_result: Optional[Dict[str, object]] = None
        if active_specs:
            bag_result = self.micb_dsmil(
                rgb_feature=rgb_feature,
                sar_feature=sar_feature,
                fused_feature=fused_roi,
                bag_specs=active_specs,
            )
            if self.mil_residual_scale > 0 and int(bag_result["bag_count"]) > 0:
                residual = self._scatter_bag_residual(
                    bag_result,
                    active_specs,
                    output_size=damage_logits_feature.shape[-2:],
                    batch_size=damage_logits_feature.shape[0],
                    dtype=damage_logits_feature.dtype,
                    device=damage_logits_feature.device,
                )
                damage_logits_feature = damage_logits_feature.clone()
                damage_logits_feature[:, 1:4] += self.mil_residual_scale * residual

        output_loc = F.interpolate(
            loc_logits_feature,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output_damage = F.interpolate(
            damage_logits_feature,
            size=post_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if not return_aux:
            return output_loc, output_damage
        return output_loc, output_damage, {
            "p_building": p_building,
            "roi_weight": roi_weight,
            "rgb_feature": rgb_feature,
            "sar_feature": sar_feature,
            "fused_roi": fused_roi,
            "bag_result": bag_result,
            "bag_count": 0 if bag_result is None else int(bag_result["bag_count"]),
            "mil_residual_scale": self.mil_residual_scale,
            "prototype_bypassed": True,
        }


__all__ = [
    "BuildingBagSpec",
    "DamageFormer_S8_MICB_DSMIL",
    "MICBDSMILHead",
    "build_predicted_bag_specs",
    "build_training_bag_specs",
    "connected_components_4",
    "hierarchical_log_probs",
]
