"""Pure metric helpers for the controlled P5 ablation study."""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


def event_from_name(name: str) -> str:
    """Strip only the terminal tile number and retain the disaster identity."""
    return re.sub(r"_\d+$", "", str(name))


def metrics_from_confusion(histogram: Sequence[Sequence[float]]) -> Dict[str, float]:
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.shape != (4, 4):
        raise ValueError(f"expected a 4x4 confusion matrix, got {hist.shape}")
    intersection = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
    precision_den = hist.sum(0)
    recall_den = hist.sum(1)
    precision = np.divide(intersection, precision_den, out=np.zeros(4), where=precision_den > 0)
    recall = np.divide(intersection, recall_den, out=np.zeros(4), where=recall_den > 0)
    f1_den = precision + recall
    f1 = np.divide(2 * precision * recall, f1_den, out=np.zeros(4), where=f1_den > 0)

    def flow(source: int, target: int) -> float:
        denominator = hist[source].sum()
        return float(hist[source, target] / denominator) if denominator > 0 else 0.0

    return {
        "foreground_miou": float(iou[1:4].mean()),
        "official_miou4": float(iou.mean()),
        "intact_iou": float(iou[1]),
        "damaged_iou": float(iou[2]),
        "destroyed_iou": float(iou[3]),
        "damaged_f1": float(f1[2]),
        "destroyed_f1": float(f1[3]),
        "damaged_to_intact": flow(2, 1),
        "destroyed_to_intact": flow(3, 1),
        "intact_to_damaged": flow(1, 2),
    }


def aggregate_by_event(rows: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[str(row["event"])].append(np.asarray(row["confusion"], dtype=np.float64))
    output: Dict[str, Dict[str, object]] = {}
    for event, matrices in sorted(grouped.items()):
        hist = np.stack(matrices).sum(0)
        output[event] = {
            "images": len(matrices),
            "confusion": hist.astype(np.int64).tolist(),
            "metrics": metrics_from_confusion(hist),
        }
    return output


def paired_bootstrap_gain(
    candidate_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    repeats: int = 2000,
    seed: int = 3407,
) -> Dict[str, float]:
    """Image-paired bootstrap of foreground-mIoU gain, returned in percentage points."""
    candidate = {str(row["name"]): np.asarray(row["confusion"], dtype=np.float64)
                 for row in candidate_rows}
    reference = {str(row["name"]): np.asarray(row["confusion"], dtype=np.float64)
                 for row in reference_rows}
    if candidate.keys() != reference.keys():
        missing = sorted(reference.keys() - candidate.keys())[:3]
        extra = sorted(candidate.keys() - reference.keys())[:3]
        raise ValueError(f"paired image sets differ; missing={missing}, extra={extra}")
    names = sorted(candidate)
    if not names:
        raise ValueError("cannot bootstrap an empty evaluation")
    cand = np.stack([candidate[name] for name in names])
    ref = np.stack([reference[name] for name in names])
    observed = 100.0 * (
        metrics_from_confusion(cand.sum(0))["foreground_miou"]
        - metrics_from_confusion(ref.sum(0))["foreground_miou"]
    )
    rng = np.random.default_rng(seed)
    gains = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, len(names), len(names))
        gains[index] = 100.0 * (
            metrics_from_confusion(cand[sample].sum(0))["foreground_miou"]
            - metrics_from_confusion(ref[sample].sum(0))["foreground_miou"]
        )
    return {
        "observed_gain_pp": float(observed),
        "ci95_low_pp": float(np.quantile(gains, 0.025)),
        "ci95_high_pp": float(np.quantile(gains, 0.975)),
        "repeats": int(repeats),
    }


def compare_events(
    candidate: Mapping[str, Mapping[str, object]],
    reference: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    names = sorted(set(candidate) & set(reference))
    gains = {
        name: 100.0 * (
            float(candidate[name]["metrics"]["foreground_miou"])
            - float(reference[name]["metrics"]["foreground_miou"])
        )
        for name in names
    }
    return {
        "event_count": len(names),
        "nonnegative_events": sum(value >= 0.0 for value in gains.values()),
        "improved_events": sum(value > 0.0 for value in gains.values()),
        "worst_event": min(gains, key=gains.get) if gains else "",
        "worst_gain_pp": min(gains.values()) if gains else 0.0,
        "event_gain_pp": gains,
    }

