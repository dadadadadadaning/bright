# -*- coding: utf-8 -*-
"""Vectorised coarse-to-fine multi-view retrieval for CMR-M2."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict

import torch
import torch.nn.functional as F


def _new_accumulator(classes: Sequence[int], k_values: Sequence[int]) -> Dict[str, object]:
    return {
        "total": 0,
        "hits": {int(k): 0 for k in k_values},
        "ap_sum": 0.0,
        "class_hits": {
            int(cls): {int(k): [0, 0] for k in k_values}
            for cls in classes
        },
        "class_ap_sum": {int(cls): 0.0 for cls in classes},
        "class_ap_count": {int(cls): 0 for cls in classes},
        "confusion": None,
    }


def _update_accumulator(
    accumulator: Dict[str, object],
    ranked_labels: torch.Tensor,
    valid_rank: torch.Tensor,
    truth: torch.Tensor,
    *,
    k_values: Sequence[int],
    ap_k: int,
    num_classes: int,
) -> None:
    if accumulator["confusion"] is None:
        accumulator["confusion"] = torch.zeros(
            (num_classes, num_classes), dtype=torch.long
        )
    prediction = ranked_labels[:, 0]
    flat = truth * num_classes + prediction
    accumulator["confusion"] += torch.bincount(
        flat.cpu(), minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    for k in k_values:
        use_k = min(int(k), ranked_labels.shape[1])
        hit = (
            (ranked_labels[:, :use_k] == truth[:, None])
            & valid_rank[:, :use_k]
        ).any(dim=1)
        accumulator["hits"][int(k)] += int(hit.sum())
        for cls in accumulator["class_hits"]:
            class_mask = truth == int(cls)
            count = int(class_mask.sum())
            if count:
                accumulator["class_hits"][cls][int(k)][0] += int(
                    hit[class_mask].sum()
                )
                accumulator["class_hits"][cls][int(k)][1] += count

    use_ap = min(int(ap_k), ranked_labels.shape[1])
    relevant = (
        (ranked_labels[:, :use_ap] == truth[:, None])
        & valid_rank[:, :use_ap]
    ).float()
    ranks = torch.arange(
        1, use_ap + 1, dtype=torch.float32, device=truth.device
    )[None, :]
    precision = relevant.cumsum(dim=1) / ranks
    relevant_count = relevant.sum(dim=1)
    average_precision = (precision * relevant).sum(dim=1)
    average_precision = torch.where(
        relevant_count > 0,
        average_precision / relevant_count.clamp_min(1.0),
        torch.zeros_like(average_precision),
    )
    accumulator["ap_sum"] += float(average_precision.sum())
    for cls in accumulator["class_ap_sum"]:
        class_mask = truth == int(cls)
        count = int(class_mask.sum())
        if count:
            accumulator["class_ap_sum"][cls] += float(
                average_precision[class_mask].sum()
            )
            accumulator["class_ap_count"][cls] += count
    accumulator["total"] += int(truth.numel())


def _finalise_accumulator(
    accumulator: Mapping[str, object],
    *,
    k_values: Sequence[int],
    ap_k: int,
) -> Dict[str, object]:
    total = max(int(accumulator["total"]), 1)
    result: Dict[str, object] = {
        f"cross_event_R@{int(k)}": accumulator["hits"][int(k)] / total
        for k in k_values
    }
    result[f"cross_event_mAP@{int(ap_k)}"] = accumulator["ap_sum"] / total
    result["per_class"] = {}
    for cls, class_hits in accumulator["class_hits"].items():
        class_result = {
            f"R@{int(k)}": (
                class_hits[int(k)][0] / class_hits[int(k)][1]
                if class_hits[int(k)][1]
                else 0.0
            )
            for k in k_values
        }
        class_count = accumulator["class_ap_count"][cls]
        class_result[f"mAP@{int(ap_k)}"] = (
            accumulator["class_ap_sum"][cls] / class_count
            if class_count
            else 0.0
        )
        result["per_class"][int(cls)] = class_result
    class_maps = [
        float(values[f"mAP@{int(ap_k)}"])
        for values in result["per_class"].values()
    ]
    result[f"macro_mAP@{int(ap_k)}"] = (
        sum(class_maps) / len(class_maps) if class_maps else 0.0
    )
    confusion = accumulator["confusion"]
    result["nn_confusion"] = {
        f"{gt}->{pred}": int(confusion[gt, pred])
        for gt in range(confusion.shape[0])
        for pred in range(confusion.shape[1])
        if int(confusion[gt, pred]) > 0
    }
    result["num_queries"] = int(accumulator["total"])
    return result


@torch.no_grad()
def multiview_rerank_sweep(
    memory_mean: torch.Tensor,
    memory_views: Mapping[str, torch.Tensor],
    memory_labels: torch.Tensor,
    memory_events: torch.Tensor,
    query_mean: torch.Tensor,
    query_views: Mapping[str, torch.Tensor],
    query_labels: torch.Tensor,
    query_events: torch.Tensor,
    *,
    lambdas: Sequence[float] = (0.10, 0.25, 0.50, 1.0),
    coarse_k: int = 20,
    k_values: Sequence[int] = (1, 5, 10),
    ap_k: int = 10,
    query_chunk: int = 512,
    device: torch.device | str | None = None,
    print_progress: bool = True,
) -> Dict[str, object]:
    """Retrieve with Mean, then rerank the fixed pool with a second key."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if int(coarse_k) < max(max(map(int, k_values)), int(ap_k)):
        raise ValueError("coarse_k must cover every reported retrieval rank")
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    view_names = tuple(memory_views)
    if set(view_names) != set(query_views):
        raise ValueError("memory/query reranking views differ")
    lambdas = tuple(sorted(set(float(value) for value in lambdas if float(value) > 0)))
    if not lambdas and view_names:
        raise ValueError("at least one positive reranking lambda is required")

    memory_mean = F.normalize(memory_mean.float(), dim=1, eps=1e-8).to(device)
    query_mean = F.normalize(query_mean.float(), dim=1, eps=1e-8).cpu()
    memory_labels = memory_labels.long().flatten().to(device)
    memory_events = memory_events.long().flatten().to(device)
    query_labels = query_labels.long().flatten().cpu()
    query_events = query_events.long().flatten().cpu()
    memory_views_gpu = {
        name: F.normalize(value.float(), dim=1, eps=1e-8).to(device)
        for name, value in memory_views.items()
    }
    query_views_cpu = {
        name: F.normalize(value.float(), dim=1, eps=1e-8).cpu()
        for name, value in query_views.items()
    }
    if memory_mean.shape[0] != memory_labels.numel():
        raise ValueError("memory tensors do not align")
    if query_mean.shape[0] != query_labels.numel():
        raise ValueError("query tensors do not align")
    for name in view_names:
        if memory_views_gpu[name].shape[0] != memory_mean.shape[0]:
            raise ValueError(f"memory view {name} does not align")
        if query_views_cpu[name].shape[0] != query_mean.shape[0]:
            raise ValueError(f"query view {name} does not align")

    configs: Dict[str, Dict[str, object]] = {
        "mean_only": {"view": None, "lambda": 0.0}
    }
    for name in view_names:
        for weight in lambdas:
            key = f"mean+{name}@{weight:g}"
            configs[key] = {"view": name, "lambda": float(weight)}
    classes = sorted(map(int, torch.unique(query_labels).tolist()))
    accumulators = {
        name: _new_accumulator(classes, k_values) for name in configs
    }
    num_classes = int(
        max(memory_labels.max().item(), query_labels.max().item()) + 1
    )
    take_k = min(int(coarse_k), memory_mean.shape[0])

    for start in range(0, query_mean.shape[0], int(query_chunk)):
        end = min(start + int(query_chunk), query_mean.shape[0])
        q_mean = query_mean[start:end].to(device, non_blocking=True)
        truth = query_labels[start:end].to(device, non_blocking=True)
        event = query_events[start:end].to(device, non_blocking=True)
        similarity = q_mean @ memory_mean.t()
        same_event = event[:, None] == memory_events[None, :]
        similarity.masked_fill_(same_event, float("-inf"))
        if bool(((~same_event).sum(dim=1) == 0).any()):
            raise RuntimeError("a query has no cross-event coarse reference")
        coarse_values, coarse_indices = torch.topk(
            similarity, k=take_k, dim=1, largest=True, sorted=True
        )
        coarse_valid = torch.isfinite(coarse_values)
        coarse_labels = memory_labels[coarse_indices]
        _update_accumulator(
            accumulators["mean_only"],
            coarse_labels,
            coarse_valid,
            truth,
            k_values=k_values,
            ap_k=ap_k,
            num_classes=num_classes,
        )

        for view_name in view_names:
            q_view = query_views_cpu[view_name][start:end].to(
                device, non_blocking=True
            )
            candidate_view = memory_views_gpu[view_name][coarse_indices]
            view_similarity = torch.einsum("bd,bkd->bk", q_view, candidate_view)
            for weight in lambdas:
                config_name = f"mean+{view_name}@{weight:g}"
                final_score = coarse_values + float(weight) * view_similarity
                final_score = final_score.masked_fill(~coarse_valid, float("-inf"))
                order = torch.argsort(final_score, dim=1, descending=True, stable=True)
                ranked_labels = torch.gather(coarse_labels, 1, order)
                ranked_valid = torch.gather(coarse_valid, 1, order)
                _update_accumulator(
                    accumulators[config_name],
                    ranked_labels,
                    ranked_valid,
                    truth,
                    k_values=k_values,
                    ap_k=ap_k,
                    num_classes=num_classes,
                )
        chunk_index = start // int(query_chunk)
        if print_progress and (
            start == 0 or end == query_mean.shape[0] or chunk_index % 10 == 0
        ):
            print(f"CMR multiview eval: {end}/{query_mean.shape[0]}", flush=True)

    results = {
        name: _finalise_accumulator(
            accumulator, k_values=k_values, ap_k=ap_k
        )
        for name, accumulator in accumulators.items()
    }
    for result in results.values():
        result["num_reference"] = int(memory_mean.shape[0])
        result["same_event_excluded"] = True
        result["coarse_k"] = int(coarse_k)
    return {"configs": configs, "results": results}

