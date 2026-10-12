from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn

from bda_benchmark.experiments.damageformer_evidence_relation_memory_v1.cache import (
    candidate_mask,
    global_region_mean,
    load_stage1_cache,
    stable_image_split,
)
from bda_benchmark.experiments.damageformer_evidence_relation_memory_v1.metrics import (
    average_precision,
    binary_auroc,
    evaluate_score,
    exact_single_utility,
)

I, D, X = 0, 1, 2


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_matrix(cache: dict, variant: str) -> torch.Tensor:
    """C0-C3 nested evidence; every increment has one causal interpretation."""
    f = cache["features"].float()
    if f.ndim != 2 or f.shape[1] % 4:
        raise ValueError(f"expected concatenated mean/std/top1/top2, got {tuple(f.shape)}")
    d = f.shape[1] // 4
    base = torch.cat((cache["base_probs"].float(), cache["uncertainty"].float()[:, None]), 1)
    if variant == "C0":
        return base
    if variant == "C1":
        return torch.cat((base, f[:, :d]), 1)
    if variant == "C2":
        return torch.cat((base, f[:, : 2 * d]), 1)
    if variant == "C3":
        return torch.cat((base, f, cache["peak_mean_delta"].float()[:, None]), 1)
    raise ValueError(variant)


class Specialist(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


@torch.no_grad()
def score_model(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    result = torch.empty(len(x), dtype=torch.float32)
    for start in range(0, len(x), 4096):
        batch = x[start:start + 4096].to(device, non_blocking=True)
        result[start:start + len(batch)] = model(batch).float().cpu()
    return result


def clean_id_candidate(cache: dict, subset: torch.Tensor, purity: float = 0.70) -> torch.Tensor:
    labels = cache["labels"].long()
    return (
        subset.bool() & candidate_mask(cache)
        & (cache["label_purity"].float() >= purity)
        & ((labels == I) | (labels == D))
    )


def balanced_sample(labels: torch.Tensor, mask: torch.Tensor, size: int, rng: random.Random) -> torch.Tensor:
    intact = torch.where(mask & (labels == I))[0].tolist()
    damaged = torch.where(mask & (labels == D))[0].tolist()
    if not intact or not damaged:
        raise RuntimeError("specialist training requires both Intact and missed-Damaged candidates")
    rows = [rng.choice(damaged if index % 2 else intact) for index in range(size)]
    rng.shuffle(rows)
    return torch.tensor(rows, dtype=torch.long)


def evaluate_coverages(cache: dict, score: torch.Tensor, candidate: torch.Tensor, name: str) -> dict:
    return {
        str(cov): evaluate_score(cache, score, candidate, cov, f"{name}@{cov}")
        for cov in (0.01, 0.02, 0.05)
    }


def score_diagnostics(cache: dict, score: torch.Tensor, mask: torch.Tensor) -> dict:
    clean = clean_id_candidate(cache, mask)
    labels = (cache["labels"].long()[clean] == D).long()
    values = score[clean]
    return {
        "count": int(clean.sum()),
        "damaged": int(labels.sum()),
        "ap": average_precision(labels, values) if len(values) else None,
        "auroc": binary_auroc(labels, values) if len(values) else None,
    }


def standardizer(value: torch.Tensor) -> tuple[float, float]:
    value = value.float()
    return float(value.mean()), float(value.std(unbiased=False).clamp_min(1.0e-6))


def z_apply(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (value.float() - float(mean)) / max(float(std), 1.0e-6)


def event_ids(events: Iterable[str]) -> tuple[torch.Tensor, dict[str, int]]:
    events = [str(value) for value in events]
    mapping = {value: index for index, value in enumerate(sorted(set(events)))}
    return torch.tensor([mapping[value] for value in events], dtype=torch.long), mapping


def normalize(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value.float(), dim=-1)


def event_balanced_prototype_score(
    train: dict,
    query: dict,
    support_mask: torch.Tensor,
    query_rows: torch.Tensor,
) -> torch.Tensor:
    """Class/event-balanced prototype delta, excluding the query event."""
    train_key = normalize(global_region_mean(train))
    query_key = normalize(global_region_mean(query)[query_rows])
    train_label = train["labels"].long()
    train_event = [str(value) for value in train["events"]]
    query_event = [str(query["events"][int(row)]) for row in query_rows]
    output = torch.empty(len(query_rows), dtype=torch.float32)
    cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for event in sorted(set(query_event)):
        class_proto = []
        for label in (I, D):
            per_event = []
            for source_event in sorted(set(train_event)):
                if source_event == event:
                    continue
                mask = support_mask & (train_label == label) & torch.tensor(
                    [value == source_event for value in train_event], dtype=torch.bool
                )
                if bool(mask.any()):
                    per_event.append(normalize(train_key[mask].mean(0, keepdim=True))[0])
            if not per_event:
                raise RuntimeError(f"no cross-event prototype support event={event} label={label}")
            class_proto.append(normalize(torch.stack(per_event).mean(0, keepdim=True))[0])
        cache[event] = (class_proto[0], class_proto[1])
    for event in sorted(set(query_event)):
        local = torch.tensor([value == event for value in query_event], dtype=torch.bool)
        pi, pd = cache[event]
        output[local] = query_key[local] @ pd - query_key[local] @ pi
    return output


def utility_targets(cache: dict, rows: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        100.0 * exact_single_utility(cache["base_pixel_cm"], cache["building_pixel_cm"][int(row)])
        for row in rows
    ], dtype=torch.float32)


def five_features(z_current: torch.Tensor, z_history: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        z_current,
        z_history,
        torch.minimum(z_current, z_history),
        (z_current - z_history).abs(),
        torch.log1p(area.float()),
    ), 1)


def selected_event_counts(cache: dict, selected: Iterable[int]) -> dict:
    result: Dict[str, int] = {}
    for row in selected:
        event = str(cache["events"][int(row)])
        result[event] = result.get(event, 0) + 1
    return result

