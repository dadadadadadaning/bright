from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


VARIANTS = {
    "mm0_frozen_a0": {"use_memory": False, "train_head": False, "exclude_event": False, "shuffle_labels": False},
    "mm1_no_memory": {"use_memory": False, "train_head": True, "exclude_event": False, "shuffle_labels": False},
    "mm2_memory": {"use_memory": True, "train_head": True, "exclude_event": False, "shuffle_labels": False},
    "mm4_event_excluded": {"use_memory": True, "train_head": True, "exclude_event": True, "shuffle_labels": False},
    "mm5_shuffled_labels": {"use_memory": True, "train_head": True, "exclude_event": True, "shuffle_labels": True},
}


def get_variant(name: str) -> Dict[str, bool]:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name}; choose from {sorted(VARIANTS)}")
    return dict(VARIANTS[name])


def event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_list(path: str | Path) -> list[str]:
    rows = [row.strip() for row in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        raise RuntimeError(f"empty data list: {path}")
    return rows


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_batch(batch: Sequence[object], device: torch.device):
    pre, post, location, damage, names = batch[:5]
    return (
        pre.to(device, non_blocking=True).float(),
        post.to(device, non_blocking=True).float(),
        location.to(device, non_blocking=True).long(),
        damage.to(device, non_blocking=True).long(),
        list(names),
    )


def class_fraction_maps(
    damage: torch.Tensor,
    location: torch.Tensor,
    output_size: Tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return building-class fractions [B,3,h,w] and valid-pixel fraction."""
    valid = (damage != 255) & (location != 255)
    building = valid & (location > 0)
    denom = F.adaptive_avg_pool2d(building[:, None].float(), output_size)
    fractions = []
    for label in (1, 2, 3):
        count = F.adaptive_avg_pool2d((building & (damage == label))[:, None].float(), output_size)
        fractions.append(count / denom.clamp_min(1e-6))
    return torch.cat(fractions, dim=1), denom


def reliable_token_targets(
    damage: torch.Tensor,
    location: torch.Tensor,
    p_building: torch.Tensor,
    purity_threshold: float = 0.8,
    building_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fractions, support = class_fraction_maps(damage, location, p_building.shape[-2:])
    purity, index = fractions.max(dim=1)
    labels = index + 1
    reliable = (
        (support[:, 0] >= 0.5)
        & (purity >= purity_threshold)
        & (p_building[:, 0] >= building_threshold)
    )
    return labels, reliable, purity


def multimodal_descriptor(rgb: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
    """Stable raw token descriptor: RGB, SAR and their absolute discrepancy."""
    rgb_unit = F.normalize(rgb.float(), dim=1)
    sar_unit = F.normalize(sar.float(), dim=1)
    difference = F.normalize((rgb_unit - sar_unit).abs(), dim=1)
    return torch.cat((rgb_unit, sar_unit, difference), dim=1)


def occurrence_loss(
    adjusted_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    delta: torch.Tensor,
    damage: torch.Tensor,
    location: torch.Tensor,
    false_positive_weight: float,
    nonharm_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    fractions, support = class_fraction_maps(damage, location, adjusted_logits.shape[-2:])
    intact = fractions[:, 0]
    affected = fractions[:, 1:].sum(dim=1)
    positive = (support[:, 0] >= 0.5) & (affected >= 0.5)
    negative = (support[:, 0] >= 0.5) & (intact >= 0.8)
    occurrence_logit = torch.logsumexp(adjusted_logits[:, 2:4].float(), dim=1) - torch.logsumexp(
        adjusted_logits[:, 0:2].float(), dim=1
    )
    terms = []
    negative_loss = occurrence_logit.sum() * 0.0
    if bool(positive.any()):
        terms.append(F.binary_cross_entropy_with_logits(occurrence_logit[positive], torch.ones_like(occurrence_logit[positive])))
    if bool(negative.any()):
        negative_loss = F.binary_cross_entropy_with_logits(
            occurrence_logit[negative], torch.zeros_like(occurrence_logit[negative])
        )
        terms.append(negative_loss)
    if not terms:
        zero = adjusted_logits.float().sum() * 0.0
        return zero, {"positive_tokens": 0, "negative_tokens": 0, "nonharm_tokens": 0}
    occurrence = sum(terms) / len(terms)
    with torch.no_grad():
        parent_pred = parent_logits.argmax(dim=1)
        parent_conf = torch.softmax(parent_logits.float(), dim=1).amax(dim=1)
        target = fractions.argmax(dim=1) + 1
        safe = negative & (parent_pred == target) & (parent_conf >= 0.8)
    nonharm = delta[:, 0][safe].float().square().mean() if bool(safe.any()) else occurrence * 0.0
    total = occurrence + float(false_positive_weight) * negative_loss + float(nonharm_weight) * nonharm
    return total, {
        "occurrence_loss": float(occurrence.detach().item()),
        "nonharm_loss": float(nonharm.detach().item()),
        "false_positive_loss": float(negative_loss.detach().item()),
        "positive_tokens": int(positive.sum().item()),
        "negative_tokens": int(negative.sum().item()),
        "nonharm_tokens": int(safe.sum().item()),
    }


def event_bootstrap(deltas: Mapping[str, float], repeats: int = 10000, seed: int = 3407):
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    if values.size == 0:
        return {"mean_pp": 0.0, "ci95_low_pp": 0.0, "ci95_high_pp": 0.0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(repeats, values.size))].mean(axis=1)
    return {
        "mean_pp": float(100.0 * values.mean()),
        "ci95_low_pp": float(100.0 * np.quantile(means, 0.025)),
        "ci95_high_pp": float(100.0 * np.quantile(means, 0.975)),
        "repeats": repeats,
    }
