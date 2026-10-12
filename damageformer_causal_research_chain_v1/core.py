from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

CLASS_NAMES = ("Intact", "Damaged", "Destroyed")
I, D, X = 0, 1, 2


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def reject_test_path(*paths: str | Path) -> None:
    for value in paths:
        if "test" in Path(value).name.lower():
            raise RuntimeError(f"TEST ACCESS IS FORBIDDEN: {value}")


def load_cache(path: str | Path, *, expected_split: str | None = None) -> Dict[str, object]:
    reject_test_path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError("cache must be a dict")
    if bool(data.get("test_accessed", False)):
        raise RuntimeError("cache declares test_accessed=True")
    split = str(data.get("split", ""))
    if expected_split and split != expected_split:
        raise RuntimeError(f"expected split={expected_split!r}, got {split!r}")
    required = ("features", "labels", "events", "names", "base_probs", "region_offsets", "region_tokens")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"cache missing keys: {missing}")
    n = int(data["labels"].shape[0])
    if int(data["features"].shape[0]) != n or len(data["events"]) != n or len(data["names"]) != n:
        raise ValueError("cache building-axis lengths disagree")
    return data


def cache_fingerprint(path: str | Path) -> Dict[str, object]:
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def binary_auroc(scores: torch.Tensor, labels01: torch.Tensor) -> float:
    """Tie-aware AUROC without sklearn. labels01 must contain 0/1."""
    s = scores.detach().double().cpu().numpy()
    y = labels01.detach().long().cpu().numpy()
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    rank = 1.0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1.0)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += j - i
        i = j
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, classes: int = 3) -> torch.Tensor:
    pred = pred.detach().long().cpu()
    target = target.detach().long().cpu()
    valid = (target >= 0) & (target < classes)
    idx = target[valid] * classes + pred[valid]
    return torch.bincount(idx, minlength=classes * classes).reshape(classes, classes).double()


def classification_metrics(logits_or_scores: torch.Tensor, labels: torch.Tensor, events: Sequence[str] | None = None) -> Dict[str, object]:
    if logits_or_scores.ndim != 2 or logits_or_scores.shape[1] != 3:
        raise ValueError("expected Nx3 scores")
    pred = logits_or_scores.argmax(dim=1)
    cm = confusion_matrix(pred, labels, 3)
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1.0)
    recall = tp / cm.sum(1).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = float(tp.sum() / cm.sum().clamp_min(1.0))

    id_mask = (labels == I) | (labels == D)
    di_auroc = binary_auroc(
        (logits_or_scores[:, D] - logits_or_scores[:, I])[id_mask],
        (labels[id_mask] == D).long(),
    ) if bool(id_mask.any()) else float("nan")
    d_total = cm[D].sum().clamp_min(1.0)
    d_to_i = float(cm[D, I] / d_total)

    out: Dict[str, object] = {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "intact_f1": float(f1[I]),
        "damaged_f1": float(f1[D]),
        "destroyed_f1": float(f1[X]),
        "intact_recall": float(recall[I]),
        "damaged_recall": float(recall[D]),
        "destroyed_recall": float(recall[X]),
        "damaged_to_intact": d_to_i,
        "di_auroc": float(di_auroc),
        "confusion": cm.tolist(),
        "count": int(labels.numel()),
        "class_counts": [int((labels == c).sum()) for c in range(3)],
    }
    if events is not None:
        event_rows = {}
        scores_cpu = logits_or_scores.detach().cpu()
        labels_cpu = labels.detach().cpu()
        unique = sorted(set(str(x) for x in events))
        event_macro = []
        event_damaged = []
        for e in unique:
            ids = torch.tensor([str(x) == e for x in events], dtype=torch.bool)
            if not bool(ids.any()):
                continue
            m = classification_metrics(scores_cpu[ids], labels_cpu[ids], None)
            event_rows[e] = {
                "macro_f1": m["macro_f1"],
                "damaged_f1": m["damaged_f1"],
                "damaged_recall": m["damaged_recall"],
                "damaged_to_intact": m["damaged_to_intact"],
                "count": m["count"],
            }
            event_macro.append(float(m["macro_f1"]))
            event_damaged.append(float(m["damaged_f1"]))
        out["event_count"] = len(event_rows)
        out["event_macro_f1_mean"] = float(np.mean(event_macro)) if event_macro else 0.0
        out["event_macro_f1_worst"] = float(np.min(event_macro)) if event_macro else 0.0
        out["event_damaged_f1_mean"] = float(np.mean(event_damaged)) if event_damaged else 0.0
        out["event_by_name"] = event_rows
    return out


def separation_ratio(features: torch.Tensor, labels: torch.Tensor, a: int = I, b: int = D) -> float:
    x = features.detach().float().cpu()
    ya = x[labels.cpu() == a]
    yb = x[labels.cpu() == b]
    if len(ya) < 2 or len(yb) < 2:
        return float("nan")
    ma, mb = ya.mean(0), yb.mean(0)
    inter = float(torch.norm(ma - mb))
    intra = float(ya.std(0, unbiased=False).norm() + yb.std(0, unbiased=False).norm())
    return inter / max(intra, 1e-8)


def event_entropy(assignments: torch.Tensor, events: Sequence[str], prototype_count: int) -> Dict[str, object]:
    values = []
    details: Dict[str, object] = {}
    events = [str(e) for e in events]
    for k in range(prototype_count):
        idx = assignments.detach().cpu() == k
        names = [events[i] for i in torch.nonzero(idx, as_tuple=False).flatten().tolist()]
        if not names:
            details[str(k)] = {"count": 0, "entropy": 0.0, "dominant_fraction": 0.0, "events": {}}
            values.append(0.0)
            continue
        uniq, cnt = np.unique(np.asarray(names, dtype=object), return_counts=True)
        p = cnt.astype(np.float64) / cnt.sum()
        h = float(-(p * np.log(p + 1e-12)).sum())
        h_norm = h / max(math.log(max(2, len(uniq))), 1e-12)
        details[str(k)] = {
            "count": len(names),
            "entropy": h_norm,
            "dominant_fraction": float(p.max()),
            "events": {str(u): int(c) for u, c in zip(uniq, cnt)},
        }
        values.append(h_norm)
    return {"mean_normalized_entropy": float(np.mean(values)) if values else 0.0, "by_prototype": details}


def top2_is_id(scores: torch.Tensor) -> torch.Tensor:
    top2 = scores.topk(k=2, dim=1).indices
    return ((top2 == I) | (top2 == D)).all(dim=1)


def rescue_harm_from_scores(
    base_scores: torch.Tensor,
    new_scores: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, object]:
    base = base_scores.argmax(1)
    new = new_scores.argmax(1)
    rescue_mask = (labels == D) & (base == I) & (new == D)
    harm_mask = (labels == I) & (base == I) & (new == D)
    base_missed = (labels == D) & (base == I)
    base_correct_i = (labels == I) & (base == I)
    rescue = int(rescue_mask.sum())
    harm = int(harm_mask.sum())
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "rescue_rate": rescue / max(1, int(base_missed.sum())),
        "harm_rate": harm / max(1, int(base_correct_i.sum())),
        "correction_precision": rescue / max(1, rescue + harm),
        "base_missed_damaged": int(base_missed.sum()),
        "base_correct_intact": int(base_correct_i.sum()),
        "changed_predictions": int((base != new).sum()),
    }


def coverage_curve(
    ranking_score: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    coverages: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> List[Dict[str, float]]:
    ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if ids.numel() == 0:
        return []
    order = ids[torch.argsort(ranking_score[ids], descending=True)]
    rows = []
    total_pos = max(1, int((positive_mask & candidate_mask).sum()))
    total_neg = max(1, int((negative_mask & candidate_mask).sum()))
    for cov in coverages:
        k = max(1, min(len(order), int(math.ceil(float(cov) * len(order)))))
        chosen = order[:k]
        rescue = int(positive_mask[chosen].sum())
        harm = int(negative_mask[chosen].sum())
        rows.append({
            "coverage": float(cov),
            "selected": int(k),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "rescue_rate": rescue / total_pos,
            "harm_rate": harm / total_neg,
            "precision": rescue / max(1, rescue + harm),
        })
    return rows


def deterministic_partition(names: Sequence[str], events: Sequence[str], labels: torch.Tensor, fraction: float = 0.2, seed: int = 3407) -> Tuple[torch.Tensor, torch.Tensor]:
    """Class-event stratified deterministic fit/calibration split for train-only verifier work."""
    n = len(names)
    fit = torch.ones(n, dtype=torch.bool)
    cal = torch.zeros(n, dtype=torch.bool)
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, (e, y) in enumerate(zip(events, labels.tolist())):
        groups.setdefault((str(e), int(y)), []).append(i)
    for key, ids in groups.items():
        ids = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{names[i]}:{i}".encode()).hexdigest())
        take = int(round(len(ids) * fraction))
        if len(ids) >= 5:
            take = max(1, take)
        else:
            take = 0
        for i in ids[:take]:
            fit[i] = False
            cal[i] = True
    return fit, cal
