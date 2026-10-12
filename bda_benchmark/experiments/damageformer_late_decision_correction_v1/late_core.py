from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

# Pixel labels in BRIGHT dense maps.
BG, INTACT_PX, DAMAGED_PX, DESTROYED_PX = 0, 1, 2, 3
# Building labels in cached 3-class building records.
I, D, X = 0, 1, 2


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def event_name(sample_name: str) -> str:
    s = str(sample_name)
    head, sep, tail = s.rpartition("_")
    return head if sep and tail.isdigit() else s


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion4(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    pred = pred.detach().long().reshape(-1).cpu()
    target = target.detach().long().reshape(-1).cpu()
    valid = (target != ignore_index) & (target >= 0) & (target < 4) & (pred >= 0) & (pred < 4)
    idx = target[valid] * 4 + pred[valid]
    return torch.bincount(idx, minlength=16).reshape(4, 4).long()


def metrics_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    cm = cm.double()
    tp = cm.diag()
    gt = cm.sum(1)
    pd = cm.sum(0)
    union = (gt + pd - tp).clamp_min(1.0)
    iou = tp / union
    f1 = 2.0 * tp / (gt + pd).clamp_min(1.0)
    d_total = gt[DAMAGED_PX].clamp_min(1.0)
    return {
        "miou4": float(iou.mean()),
        "valid_miou": float(iou[1:4].mean()),
        "background_iou": float(iou[BG]),
        "intact_iou": float(iou[INTACT_PX]),
        "damaged_iou": float(iou[DAMAGED_PX]),
        "destroyed_iou": float(iou[DESTROYED_PX]),
        "intact_f1": float(f1[INTACT_PX]),
        "damaged_f1": float(f1[DAMAGED_PX]),
        "destroyed_f1": float(f1[DESTROYED_PX]),
        "damaged_to_intact": float(cm[DAMAGED_PX, INTACT_PX] / d_total),
    }


def apply_override_to_cm(
    base_cm: torch.Tensor,
    building_cms: torch.Tensor,
    selected: torch.Tensor | Sequence[int],
    *,
    mode: str = "intact_only",
) -> torch.Tensor:
    """Apply a building-level late decision exactly in confusion-matrix space.

    building_cms[n, gt, pred] is the pixel confusion contribution inside one
    predicted building. Predicted-building masks are disjoint, so adjustments
    are additive.

    intact_only: only current-predicted Intact pixels become Damaged.
    whole: every pixel inside selected predicted buildings becomes Damaged.
    """
    out = base_cm.clone().long()
    if isinstance(selected, torch.Tensor) and selected.dtype == torch.bool:
        ids = torch.nonzero(selected, as_tuple=False).flatten().tolist()
    elif isinstance(selected, torch.Tensor):
        ids = selected.flatten().tolist()
    else:
        ids = [int(x) for x in selected]

    for idx in ids:
        b = building_cms[int(idx)].long()
        if mode == "intact_only":
            moved = b[:, INTACT_PX].clone()
            out[:, INTACT_PX] -= moved
            out[:, DAMAGED_PX] += moved
        elif mode == "whole":
            moved_total = b.sum(1)
            out -= b
            out[:, DAMAGED_PX] += moved_total
        else:
            raise ValueError(f"unknown override mode: {mode}")
    return out


def candidate_mask(cache: Dict[str, object], rule: str = "base_intact") -> torch.Tensor:
    """Observable candidate rule only. No GT label/purity is consulted."""
    probs = cache["base_probs"].float()
    pred = probs.argmax(1)
    if rule == "base_intact":
        return pred == I
    if rule == "top2_id":
        top2 = probs.topk(k=2, dim=1).indices
        return ((top2 == I) | (top2 == D)).all(dim=1)
    if rule == "union":
        top2 = probs.topk(k=2, dim=1).indices
        return (pred == I) | (((top2 == I) | (top2 == D)).all(dim=1))
    raise ValueError(rule)


def supervised_train_mask(cache: Dict[str, object], rule: str, min_purity: float = 0.70) -> torch.Tensor:
    """Training-only GT mask. Safe because it is never used at Val inference."""
    labels = cache["labels"].long()
    purity = cache["label_purity"].float()
    return candidate_mask(cache, rule) & ((labels == I) | (labels == D)) & (purity >= min_purity)


def binary_auroc(scores: torch.Tensor, labels01: torch.Tensor) -> float:
    s = scores.detach().double().cpu().numpy()
    y = labels01.detach().long().cpu().numpy()
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i, rank = 0, 1.0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg = (rank + rank + (j - i) - 1.0) / 2.0
        ranks[order[i:j]] = avg
        rank += j - i
        i = j
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)


def zscore_on_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x.float())
    vals = x[mask].float()
    if vals.numel() == 0:
        return out
    mu = vals.mean()
    sd = vals.std(unbiased=False).clamp_min(1e-6)
    out[mask] = (vals - mu) / sd
    return out


def fixed_budget_rows(
    score: torch.Tensor,
    cache: Dict[str, object],
    *,
    candidate: torch.Tensor,
    coverages: Iterable[float] = (0.01, 0.02, 0.05),
    override_mode: str = "intact_only",
) -> List[Dict[str, object]]:
    """Select by score using only observable candidate mask; evaluate with GT afterwards."""
    ids = torch.nonzero(candidate, as_tuple=False).flatten()
    order = ids[torch.argsort(score[ids], descending=True)] if len(ids) else ids
    labels = cache["labels"].long()
    purity = cache["label_purity"].float()
    base_cm = cache["base_pixel_cm"].long()
    bcms = cache["building_pixel_cm"].long()
    rows: List[Dict[str, object]] = []
    for cov in coverages:
        k = max(1, min(len(order), int(math.ceil(float(cov) * len(order))))) if len(order) else 0
        chosen = order[:k]
        y = labels[chosen] if k else torch.empty(0, dtype=torch.long)
        clean = purity[chosen] >= 0.70 if k else torch.empty(0, dtype=torch.bool)
        rescue = int(((y == D) & clean).sum())
        harm = int(((y == I) & clean).sum())
        selected_x = int(((y == X) & clean).sum())
        selected_unknown = int(((y < 0) | (~clean)).sum()) if k else 0
        cm = apply_override_to_cm(base_cm, bcms, chosen, mode=override_mode)
        m = metrics_from_cm(cm)
        rows.append({
            "coverage": float(cov),
            "candidate_count": int(len(ids)),
            "selected": int(k),
            "rescue_buildings_clean": rescue,
            "harm_intact_buildings_clean": harm,
            "selected_destroyed_buildings_clean": selected_x,
            "selected_unknown_or_impure": selected_unknown,
            "building_precision_clean_id": rescue / max(1, rescue + harm),
            **m,
        })
    return rows


def assert_cache_identity(cache_path: str | Path, checkpoint_path: str | Path, expected_split: str | None = None) -> Dict[str, object]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    ckpt_sha = sha256_file(checkpoint_path)
    if expected_split and str(cache.get("split")) != expected_split:
        raise RuntimeError(f"cache split={cache.get('split')} expected={expected_split}")
    if cache.get("source_checkpoint_sha256") != ckpt_sha:
        raise RuntimeError(
            "CACHE IDENTITY FAIL: "
            f"cache={cache.get('source_checkpoint_sha256')} checkpoint={ckpt_sha}"
        )
    if bool(cache.get("test_accessed", False)):
        raise RuntimeError("TEST access forbidden")
    return cache
