from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_cluster_detail_d1_v1.core import (
    fixed_edge_maps,
    fixed_sar_texture_maps,
)
from bda_benchmark.experiments.spi_complete_building_memory_v1.core import (
    binary_auroc,
    macro_f1,
    standardize,
)
from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import RGB_MEAN, RGB_STD


ROLE_NAMES = ("whole", "core", "ring", "edge", "anomaly", "stable")


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float()[None, None], size=size, mode="area")[0, 0] >= .25


def role_masks(mask: torch.Tensor, anomaly: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build label-free spatial roles on the evidence-map grid."""
    mask = mask.bool()
    eroded = ~F.max_pool2d((~mask).float()[None, None], 3, 1, 1).bool()[0, 0]
    core = mask & eroded
    if not core.any():
        core = mask
    ring = mask & ~core
    if not ring.any():
        ring = mask
    edge = ring
    values = anomaly[mask]
    high = torch.quantile(values.float(), .75) if values.numel() else anomaly.new_tensor(0.)
    low = torch.quantile(values.float(), .25) if values.numel() else anomaly.new_tensor(0.)
    return {
        "whole": mask,
        "core": core,
        "ring": ring,
        "edge": edge,
        "anomaly": mask & (anomaly >= high),
        "stable": mask & (anomaly <= low),
    }


def masked_moments(maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Channel-wise mean/std/upper quartile; maps is [C,H,W]."""
    values = maps[:, mask]
    if values.numel() == 0:
        return maps.new_zeros(maps.shape[0] * 3)
    return torch.cat((values.mean(1), values.std(1, unbiased=False), torch.quantile(values, .75, dim=1)))


def independent_evidence_maps(rgb: torch.Tensor, sar: torch.Tensor, side: int = 256):
    """Fixed-filter evidence read directly from the two sensors, never P3 features."""
    size = (min(side, rgb.shape[-2]), min(side, rgb.shape[-1]))
    mean = torch.as_tensor(RGB_MEAN, dtype=rgb.dtype, device=rgb.device).reshape(1, 3, 1, 1)
    std = torch.as_tensor(RGB_STD, dtype=rgb.dtype, device=rgb.device).reshape(1, 3, 1, 1)
    rgb_raw = rgb.float() * std + mean
    rgb_small = F.interpolate(rgb_raw, size=size, mode="bilinear", align_corners=False)
    sar_small = F.interpolate(sar[:, :1].float(), size=size, mode="bilinear", align_corners=False)
    gray = (.2989 * rgb_small[:, :1] + .5870 * rgb_small[:, 1:2] + .1140 * rgb_small[:, 2:3])
    rgb_maps, _ = fixed_edge_maps(gray)
    sar_maps, _ = fixed_sar_texture_maps(sar_small)
    return rgb_maps[0], sar_maps[0]


def build_independent_card(rgb_maps: torch.Tensor, sar_maps: torch.Tensor,
                           full_mask: torch.Tensor, geometry: torch.Tensor,
                           tokens_per_role: int = 3) -> dict[str, torch.Tensor]:
    mask = resize_mask(full_mask, rgb_maps.shape[-2:]).to(rgb_maps.device)
    if not mask.any():
        raise RuntimeError("building vanished on independent evidence grid")
    anomaly = sar_maps.abs().mean(0)
    roles = role_masks(mask, anomaly)
    rgb_structure = torch.cat([masked_moments(rgb_maps, roles[name]) for name in ("whole", "core", "ring")])
    sar_physics = torch.cat([masked_moments(sar_maps, roles[name]) for name in ROLE_NAMES])
    rgb_role = {name: masked_moments(rgb_maps, roles[name]) for name in ROLE_NAMES}
    sar_role = {name: masked_moments(sar_maps, roles[name]) for name in ROLE_NAMES}
    relative = torch.cat((
        rgb_role["core"] - rgb_role["ring"],
        sar_role["core"] - sar_role["ring"],
        rgb_role["anomaly"] - rgb_role["stable"],
        sar_role["anomaly"] - sar_role["stable"],
    ))
    area_fraction = mask.float().mean()
    boundary_fraction = roles["ring"].sum().float() / mask.sum().clamp_min(1)
    sar_inside = sar_maps[:, mask]
    quality = torch.cat((geometry.float(), torch.tensor([
        float(area_fraction), float(boundary_fraction),
        float(sar_inside.std(unbiased=False)), float(anomaly[mask].median()),
    ], device=geometry.device)))

    # Coordinate-bearing role tokens preserve spatial variation instead of one prototype.
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, rgb_maps.shape[-2], device=rgb_maps.device),
        torch.linspace(-1, 1, rgb_maps.shape[-1], device=rgb_maps.device), indexing="ij",
    )
    pixel = torch.cat((rgb_maps, sar_maps, yy[None], xx[None], anomaly[None]), 0).permute(1, 2, 0)
    tokens, token_mask = [], []
    for name in ROLE_NAMES:
        where = roles[name]
        score = anomaly[where]
        if name == "stable":
            score = -score
        count = min(tokens_per_role, int(where.sum()))
        selected = score.argsort(descending=True, stable=True)[:count]
        value = pixel[where][selected] if count else pixel.new_zeros((0, pixel.shape[-1]))
        packed = pixel.new_zeros((tokens_per_role, pixel.shape[-1]))
        valid = torch.zeros(tokens_per_role, dtype=torch.bool, device=pixel.device)
        if count:
            packed[:count] = value
            valid[:count] = True
        tokens.append(packed); token_mask.append(valid)
    return {
        "rgb_structure": rgb_structure,
        "sar_physics": sar_physics,
        "relative": relative,
        "quality": quality,
        "role_tokens": torch.stack(tokens),
        "role_token_mask": torch.stack(token_mask),
    }


def group_matrix(cache: dict, groups: tuple[str, ...], statistics: dict | None = None):
    statistics = {} if statistics is None else dict(statistics)
    parts = []
    for name in groups:
        value = cache[name].float()
        if name == "role_tokens":
            valid = cache["role_token_mask"].float()[..., None]
            value = ((value * valid).sum(2) / valid.sum(2).clamp_min(1)).flatten(1)
        value, mean, std = standardize(value, statistics.get(name + "_mean"), statistics.get(name + "_std"))
        statistics[name + "_mean"] = mean; statistics[name + "_std"] = std
        parts.append(value)
    return torch.cat(parts, 1), statistics


class ConditionalResidualModel(nn.Module):
    """Independent evidence predicts only the residual left after P3."""
    def __init__(self, input_dim: int, hidden: int = 128, prototypes_per_class: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.residual = nn.Linear(hidden, 4)
        self.prototypes = nn.Parameter(torch.randn(4, prototypes_per_class, hidden))
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, value: torch.Tensor, p3_probability: torch.Tensor):
        embedding = F.normalize(self.encoder(value.float()), dim=1, eps=1e-8)
        logits = p3_probability.float().clamp_min(1e-6).log() + self.residual(embedding)
        prototype = F.normalize(self.prototypes, dim=2)
        prototype_logits = 10 * torch.einsum("nd,ckd->nck", embedding, prototype).amax(2)
        return logits, embedding, prototype_logits


def cross_event_contrastive(embedding: torch.Tensor, label: torch.Tensor,
                            event: torch.Tensor, temperature: float = .12) -> torch.Tensor:
    similarity = embedding @ embedding.T / temperature
    eye = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (label[:, None] == label[None]) & (event[:, None] != event[None]) & ~eye
    valid = positive.any(1)
    if not valid.any():
        return embedding.sum() * 0
    similarity = similarity.masked_fill(eye, -1e4)
    logp = similarity - torch.logsumexp(similarity, 1, keepdim=True)
    return -((logp * positive).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()


def prototype_diversity(prototypes: torch.Tensor, margin: float = .25) -> torch.Tensor:
    p = F.normalize(prototypes, dim=2)
    similarity = torch.einsum("ckd,cjd->ckj", p, p)
    eye = torch.eye(p.shape[1], dtype=torch.bool, device=p.device)[None]
    return F.relu(similarity.masked_fill(eye, -1) - margin).mean()


def multiclass_report(target: torch.Tensor, probability: torch.Tensor, events: list[str]) -> dict:
    auc = [binary_auroc(target == c, probability[:, c]) for c in range(4)]
    valid = [x for x in auc if x == x]
    prediction = probability.argmax(1)
    event_f1 = {}
    for event in sorted(set(events)):
        mask = torch.tensor([x == event for x in events])
        event_f1[event] = macro_f1(target[mask], prediction[mask])
    return {"macro_ovr_auroc": sum(valid) / max(len(valid), 1), "auroc_by_class": auc,
            "macro_f1": macro_f1(target, prediction),
            "event_macro_f1": sum(event_f1.values()) / max(len(event_f1), 1), "events": event_f1}


def kcenter_indices(value: torch.Tensor, maximum: int) -> torch.Tensor:
    """Deterministic farthest-first subset for one event/class stratum."""
    if len(value) <= maximum:
        return torch.arange(len(value))
    x = F.normalize(value.float(), dim=1, eps=1e-8)
    centre = F.normalize(x.mean(0), dim=0)
    first = int((x @ centre).argmin())
    selected = [first]
    distance = 1 - x @ x[first]
    for _ in range(1, maximum):
        nxt = int(distance.argmax()); selected.append(nxt)
        distance = torch.minimum(distance, 1 - x @ x[nxt])
    return torch.tensor(selected, dtype=torch.long)
