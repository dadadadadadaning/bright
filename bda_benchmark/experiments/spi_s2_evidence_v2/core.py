from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_s2_evidence_ladder_v1.evidence import (
    im4cd_reliability_evidence,
    relation_difference_evidence,
)


BASE_CHANNEL_NAMES = tuple(f"utility_{index:02d}" for index in range(35))
DIRECT_CHANNEL_NAMES = (
    "cosine_distance",
    "absolute_difference_mean",
    "absolute_difference_std",
    "absolute_difference_max",
)


@dataclass(frozen=True)
class EvidenceMap:
    values: torch.Tensor
    channel_names: tuple[str, ...]
    metadata: Dict[str, float | int | str]


@torch.no_grad()
def direct_difference_evidence(
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    output_size: tuple[int, int],
) -> EvidenceMap:
    if rgb_feature.shape != sar_feature.shape:
        raise ValueError("direct evidence requires equal RGB/SAR shapes")
    rgb = F.normalize(rgb_feature.float(), dim=1, eps=1e-6)
    sar = F.normalize(sar_feature.float(), dim=1, eps=1e-6)
    absolute = (rgb - sar).abs()
    values = torch.cat(
        (
            1.0 - (rgb * sar).sum(1, keepdim=True),
            absolute.mean(1, keepdim=True),
            absolute.std(1, keepdim=True, unbiased=False),
            absolute.amax(1, keepdim=True),
        ),
        dim=1,
    )
    values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)
    return EvidenceMap(values, DIRECT_CHANNEL_NAMES, {"source": "fused_multiscale_256"})


@torch.no_grad()
def relation_evidence(
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    output_size: tuple[int, int],
) -> EvidenceMap:
    result = relation_difference_evidence(rgb_feature, sar_feature, output_size)
    return EvidenceMap(
        result.values,
        result.channel_names,
        {"source": "fused_multiscale_256", "neighbourhood": "8-connected"},
    )


@torch.no_grad()
def matching_evidence(
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    output_size: tuple[int, int],
    *,
    max_side: int = 80,
    window_size: int = 5,
) -> EvidenceMap:
    height, width = rgb_feature.shape[-2:]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale < 1.0:
        size = (max(3, round(height * scale)), max(3, round(width * scale)))
        rgb_feature = F.interpolate(rgb_feature, size=size, mode="bilinear", align_corners=False)
        sar_feature = F.interpolate(sar_feature, size=size, mode="bilinear", align_corners=False)
    result = im4cd_reliability_evidence(
        rgb_feature, sar_feature, output_size, window_size=window_size
    )
    return EvidenceMap(
        result.values,
        result.channel_names,
        {
            "source": "fused_multiscale_256",
            "matching_height": int(rgb_feature.shape[-2]),
            "matching_width": int(rgb_feature.shape[-1]),
            "window_size": int(window_size),
        },
    )


def _grid_labels(height: int, width: int, cell_size: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    grid_width = (width + cell_size - 1) // cell_size
    return (yy // cell_size) * grid_width + (xx // cell_size)


def _slic_labels(rgb: torch.Tensor, node_budget: int) -> torch.Tensor:
    try:
        from skimage.segmentation import slic
    except ImportError as error:
        raise RuntimeError("fixed SLIC audit requires scikit-image") from error
    image = rgb.detach().float().permute(1, 2, 0).cpu().numpy()
    image = image - image.min(axis=(0, 1), keepdims=True)
    image = image / np.maximum(image.max(axis=(0, 1), keepdims=True), 1e-6)
    labels = slic(
        image,
        n_segments=int(node_budget),
        compactness=10.0,
        sigma=0.0,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    )
    return torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(rgb.device)


@torch.no_grad()
def partition_evidence(
    pre_rgb: torch.Tensor,
    relation_low: torch.Tensor,
    output_size: tuple[int, int],
    *,
    mode: str,
    cell_size: int = 4,
) -> EvidenceMap:
    """Pool the same relation signal with equal-request-budget Grid or RGB-SLIC."""

    if mode not in {"grid", "slic"}:
        raise ValueError("partition mode must be grid or slic")
    if pre_rgb.shape[0] != 1 or relation_low.shape[0] != 1:
        raise ValueError("partition evidence currently requires batch size one")
    height, width = relation_low.shape[-2:]
    grid_h = (height + cell_size - 1) // cell_size
    grid_w = (width + cell_size - 1) // cell_size
    requested_nodes = grid_h * grid_w
    rgb_low = F.interpolate(pre_rgb.float(), size=(height, width), mode="bilinear", align_corners=False)
    labels = (
        _grid_labels(height, width, cell_size, relation_low.device)
        if mode == "grid"
        else _slic_labels(rgb_low[0], requested_nodes)
    )
    labels = labels.long()
    node_count = int(labels.max().item()) + 1
    flat_labels = labels.reshape(-1)
    scalar = relation_low.float().mean(1)[0].reshape(-1)
    counts = torch.bincount(flat_labels, minlength=node_count).float().clamp_min(1.0)
    sums = torch.bincount(flat_labels, weights=scalar, minlength=node_count)
    squared = torch.bincount(flat_labels, weights=scalar.square(), minlength=node_count)
    means = sums / counts
    stds = (squared / counts - means.square()).clamp_min(0.0).sqrt()
    mean_map = means[flat_labels].reshape(height, width)
    std_map = stds[flat_labels].reshape(height, width)
    error_map = (scalar - means[flat_labels]).abs().reshape(height, width)
    mass_map = (counts[flat_labels] / counts.max().clamp_min(1.0)).reshape(height, width)
    values = torch.stack((mean_map, std_map, error_map, mass_map), dim=0)[None]
    values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)
    return EvidenceMap(
        values,
        ("region_relation_mean", "region_relation_std", "relation_reconstruction_error", "normalized_mass"),
        {
            "partition": mode,
            "cell_size": int(cell_size),
            "requested_nodes": int(requested_nodes),
            "actual_nodes": int(node_count),
            "shared_rgb_sar_partition": "true",
        },
    )


@torch.no_grad()
def build_evidence_maps(
    pre_rgb: torch.Tensor,
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    output_size: tuple[int, int],
    *,
    groups: tuple[str, ...],
    matching_max_side: int = 80,
    matching_window: int = 5,
    cell_size: int = 4,
) -> Dict[str, EvidenceMap]:
    output: Dict[str, EvidenceMap] = {}
    if "direct" in groups:
        output["direct"] = direct_difference_evidence(rgb_feature, sar_feature, output_size)
    relation_low = None
    if any(name in groups for name in ("relation", "grid", "slic")):
        relation_low = relation_evidence(
            rgb_feature, sar_feature, tuple(rgb_feature.shape[-2:])
        )
    if "relation" in groups:
        output["relation"] = EvidenceMap(
            F.interpolate(relation_low.values, size=output_size, mode="bilinear", align_corners=False),
            relation_low.channel_names,
            relation_low.metadata,
        )
    if "matching" in groups:
        output["matching"] = matching_evidence(
            rgb_feature,
            sar_feature,
            output_size,
            max_side=matching_max_side,
            window_size=matching_window,
        )
    for mode in ("grid", "slic"):
        if mode in groups:
            output[mode] = partition_evidence(
                pre_rgb,
                relation_low.values,
                output_size,
                mode=mode,
                cell_size=cell_size,
            )
    return output
