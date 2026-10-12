"""Pure numerical utilities for the BIOS E0 relation-evidence audit.

The functions in this module neither contain a trainable projection nor use a
ground-truth label while constructing a score.  Labels are consumed only by
the metric functions after all structural signals have been computed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def binary_auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    positive = int(y.sum())
    if positive == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positive)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def spearman_correlation(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    if len(y) < 2 or np.unique(y).size < 2 or np.unique(s).size < 2:
        return None
    yr = _average_ranks(y)
    sr = _average_ranks(s)
    value = np.corrcoef(yr, sr)[0, 1]
    return float(value) if np.isfinite(value) else None


def shift_token_grid(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    grid_size: int,
    dy: int,
    dx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift tokens without circular wraparound and return the shifted validity."""
    if tokens.ndim != 2 or tokens.shape[0] != grid_size * grid_size:
        raise ValueError("tokens must have shape [grid_size**2, channels]")
    source = tokens.reshape(grid_size, grid_size, -1)
    source_valid = valid.reshape(grid_size, grid_size)
    output = torch.zeros_like(source)
    output_valid = torch.zeros_like(source_valid)
    src_y0, src_y1 = max(0, -dy), min(grid_size, grid_size - dy)
    src_x0, src_x1 = max(0, -dx), min(grid_size, grid_size - dx)
    dst_y0, dst_y1 = max(0, dy), min(grid_size, grid_size + dy)
    dst_x0, dst_x1 = max(0, dx), min(grid_size, grid_size + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
        output_valid[dst_y0:dst_y1, dst_x0:dst_x1] = source_valid[
            src_y0:src_y1, src_x0:src_x1
        ]
    return output.reshape_as(tokens), output_valid.reshape_as(valid)


def deterministic_valid_shuffle(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    output = tokens.clone()
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() < 2:
        return output
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    order = torch.randperm(indices.numel(), generator=generator).to(indices.device)
    output[indices] = tokens[indices[order]]
    return output


def _pair_masks(grid_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )
    coordinates = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    distance = (coordinates[:, None] - coordinates[None]).abs().amax(2)
    eye = torch.eye(grid_size * grid_size, dtype=torch.bool, device=device)
    local = (distance <= 1) & ~eye
    global_mask = ~eye
    return local, global_mask


def _masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


def relation_token_anomaly(
    rgb_tokens: torch.Tensor,
    sar_tokens: torch.Tensor,
    valid: torch.Tensor,
    *,
    grid_size: int,
    spatial: str = "local",
    relation: str = "sim",
    neighbour_count: int = 4,
) -> torch.Tensor:
    """Return one label-free structural anomaly score per valid ROI token."""
    if rgb_tokens.shape[0] != sar_tokens.shape[0] or valid.numel() != rgb_tokens.shape[0]:
        raise ValueError("RGB, SAR and validity token counts must match")
    if spatial not in ("local", "global", "local_global"):
        raise ValueError("spatial must be local, global or local_global")
    if relation not in ("sim", "sd"):
        raise ValueError("relation must be sim or sd")
    rgb = F.normalize(rgb_tokens.float(), dim=1, eps=1e-6)
    sar = F.normalize(sar_tokens.float(), dim=1, eps=1e-6)
    rgb_similarity = rgb @ rgb.T
    sar_similarity = sar @ sar.T
    discrepancy = (rgb_similarity - sar_similarity).abs()
    local, global_mask = _pair_masks(grid_size, rgb_tokens.device)
    valid_pairs = valid[:, None] & valid[None, :]

    def one(spatial_mask: torch.Tensor) -> torch.Tensor:
        eligible = spatial_mask & valid_pairs
        if relation == "sim":
            return _masked_row_mean(discrepancy, eligible)
        # SDIR-style diagnostic: preserve evidence from both the most similar
        # and the most dissimilar RGB neighbours.  No labels choose either set.
        output = discrepancy.new_zeros(discrepancy.shape[0])
        for row in range(discrepancy.shape[0]):
            candidates = torch.nonzero(eligible[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            count = min(int(neighbour_count), int(candidates.numel()))
            similarities = rgb_similarity[row, candidates]
            nearest = candidates[torch.topk(similarities, count).indices]
            farthest = candidates[torch.topk(-similarities, count).indices]
            output[row] = 0.5 * (
                discrepancy[row, nearest].mean() + discrepancy[row, farthest].mean()
            )
        return output

    if spatial == "local":
        result = one(local)
    elif spatial == "global":
        result = one(global_mask)
    else:
        result = 0.5 * (one(local) + one(global_mask))
    return result[valid]


def direct_token_anomaly(
    rgb_tokens: torch.Tensor,
    sar_tokens: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if rgb_tokens.shape != sar_tokens.shape:
        raise ValueError("direct baseline requires equal RGB/SAR token dimensions")
    cosine = F.cosine_similarity(rgb_tokens.float(), sar_tokens.float(), dim=1, eps=1e-6)
    return (1.0 - cosine)[valid]


def _largest_component_fraction(mask: torch.Tensor) -> float:
    """Fraction of active cells contained in the largest 4-connected component."""
    array = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    active = int(array.sum())
    if active == 0:
        return 0.0
    visited = np.zeros_like(array, dtype=bool)
    largest = 0
    for y, x in np.argwhere(array):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < array.shape[0] and 0 <= nx < array.shape[1]
                    and array[ny, nx] and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        largest = max(largest, size)
    return largest / active


def aggregate_token_scores(
    values: torch.Tensor,
    threshold: float = 0.25,
    *,
    valid: torch.Tensor | None = None,
    grid_size: int | None = None,
    statistics: Sequence[str] | None = None,
) -> dict[str, float]:
    requested = tuple(statistics or (
        "mean", "std", "max", "q75", "q90", "top20", "area", "concentration"
    ))
    supported = {"mean", "std", "max", "q75", "q90", "top20", "area", "concentration"}
    if not set(requested).issubset(supported):
        raise ValueError(f"unsupported BIOS statistics: {set(requested) - supported}")
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {key: float("nan") for key in requested}
    count = max(1, int(np.ceil(0.20 * values.numel())))
    result: dict[str, float] = {}
    if "mean" in requested:
        result["mean"] = float(values.mean())
    if "std" in requested:
        result["std"] = float(values.std(unbiased=False))
    if "max" in requested:
        result["max"] = float(values.max())
    if "q75" in requested:
        result["q75"] = float(torch.quantile(values, 0.75))
    if "q90" in requested:
        result["q90"] = float(torch.quantile(values, 0.90))
    if "top20" in requested:
        result["top20"] = float(torch.topk(values, count).values.mean())
    if "area" in requested:
        result["area"] = float((values > float(threshold)).float().mean())
    if "concentration" in requested:
        concentration = float("nan")
        if valid is not None and grid_size is not None:
            if valid.numel() != grid_size * grid_size or int(valid.sum()) != values.numel():
                raise ValueError("valid grid does not match the supplied token scores")
            anomaly = torch.zeros_like(valid, dtype=torch.bool)
            anomaly[valid] = values > float(threshold)
            concentration = _largest_component_fraction(anomaly.reshape(grid_size, grid_size))
        result["concentration"] = concentration
    return result


def signal_metrics(rows: Sequence[Mapping[str, object]], signal: str) -> dict[str, object]:
    usable = [row for row in rows if np.isfinite(float(row[signal]))]

    def binary(classes: Iterable[int], positive: int) -> dict[str, object]:
        selected = [row for row in usable if int(row["label"]) in set(classes)]
        labels = [int(int(row["label"]) == positive) for row in selected]
        scores = [float(row[signal]) for row in selected]
        return {
            "count": len(selected),
            "positive": int(sum(labels)),
            "auroc": binary_auroc(labels, scores),
            "auprc": binary_auprc(labels, scores),
        }

    def miss(predictions: set[int]) -> dict[str, object]:
        selected = [
            row for row in usable
            if int(row["label"]) in (1, 2) and int(row["p3_prediction"]) in predictions
        ]
        labels = [int(int(row["label"]) == 2) for row in selected]
        scores = [float(row[signal]) for row in selected]
        by_event: dict[str, dict[str, list[float] | list[int]]] = defaultdict(
            lambda: {"labels": [], "scores": []}
        )
        for row, label, score in zip(selected, labels, scores):
            event = str(row["event"])
            by_event[event]["labels"].append(label)  # type: ignore[union-attr]
            by_event[event]["scores"].append(score)  # type: ignore[union-attr]
        event_auc = {
            event: binary_auroc(value["labels"], value["scores"])  # type: ignore[arg-type]
            for event, value in by_event.items()
        }
        comparable = {key: value for key, value in event_auc.items() if value is not None}
        return {
            "count": len(selected),
            "positive": int(sum(labels)),
            "auroc": binary_auroc(labels, scores),
            "auprc": binary_auprc(labels, scores),
            "event_auroc": event_auc,
            "event_macro_auroc": (
                float(np.mean(list(comparable.values()))) if comparable else None
            ),
            "event_worst_auroc": min(comparable.values()) if comparable else None,
            "nonnegative_events": int(sum(value > 0.5 for value in comparable.values())),
            "events_above_0p55": int(sum(value > 0.55 for value in comparable.values())),
            "events_above_0p60": int(sum(value > 0.60 for value in comparable.values())),
            "comparable_events": len(comparable),
        }

    labels = [int(row["label"]) - 1 for row in usable]
    scores = [float(row[signal]) for row in usable]
    class_values = {
        name: [float(row[signal]) for row in usable if int(row["label"]) == index]
        for index, name in ((1, "intact"), (2, "damaged"), (3, "destroyed"))
    }
    return {
        "signal": signal,
        "regions": len(usable),
        "class_counts": {key: len(value) for key, value in class_values.items()},
        "class_means": {
            key: (float(np.mean(value)) if value else None) for key, value in class_values.items()
        },
        "class_medians": {
            key: (float(np.median(value)) if value else None) for key, value in class_values.items()
        },
        "intact_vs_damaged": binary((1, 2), 2),
        "damaged_vs_destroyed": binary((2, 3), 3),
        "intact_vs_destroyed": binary((1, 3), 3),
        "p3_intact_miss": miss({1}),
        "p3_nonaffected_miss": miss({0, 1}),
        "severity_spearman": spearman_correlation(labels, scores),
    }


def affected_metrics(rows: Sequence[Mapping[str, object]], signal: str) -> dict[str, object]:
    usable = [row for row in rows if np.isfinite(float(row[signal]))]
    labels = [int(int(row["label"]) >= 2) for row in usable]
    scores = [float(row[signal]) for row in usable]
    return {
        "count": len(usable), "positive": int(sum(labels)),
        "auroc": binary_auroc(labels, scores), "auprc": binary_auprc(labels, scores),
    }


def complete_signal_metrics(rows: Sequence[Mapping[str, object]], signal: str) -> dict[str, object]:
    result = signal_metrics(rows, signal)
    result["affected"] = affected_metrics(rows, signal)
    return result


def selection_key(metrics: Mapping[str, object]) -> tuple[float, float, int, float]:
    miss = metrics["p3_nonaffected_miss"]  # type: ignore[index]
    global_auc = miss.get("auroc")  # type: ignore[union-attr]
    event_auc = miss.get("event_macro_auroc")  # type: ignore[union-attr]
    direction = miss.get("nonnegative_events", 0)  # type: ignore[union-attr]
    ordinal = metrics.get("severity_spearman")
    return (
        -1.0 if global_auc is None else float(global_auc),
        -1.0 if event_auc is None else float(event_auc),
        int(direction),
        -1.0 if ordinal is None else float(ordinal),
    )


__all__ = [
    "aggregate_token_scores", "binary_auprc", "binary_auroc",
    "complete_signal_metrics", "deterministic_valid_shuffle",
    "direct_token_anomaly", "relation_token_anomaly", "selection_key",
    "shift_token_grid", "spearman_correlation",
]
