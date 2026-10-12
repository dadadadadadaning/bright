# -*- coding: utf-8 -*-
"""Fast, protocol-locked utilities for the isolated CMR-M1 experiment."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from model.cmr.pooling import CMR_KEY_VARIANTS


def load_cmr_cache(
    path: str,
    variant: str,
    *,
    expected_role: str | None = None,
    print_schema: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int], Mapping[str, object]]:
    """Load one CMR-P0 cache without changing its frozen protocol."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"CMR cache must be a mapping: {path}")
    required = ("key_variants", "labels", "events", "image_ids", "metadata")
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError(f"CMR cache {path} is missing: {missing}")
    metadata = payload["metadata"]
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("CMR-M1 requires an unmodified CMR-P0 cache")
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("CMR-M1 refuses a cache derived from Test")
    if expected_role is not None and metadata.get("role") != expected_role:
        raise RuntimeError(
            f"CMR cache role mismatch: {metadata.get('role')} != {expected_role}"
        )
    if variant not in CMR_KEY_VARIANTS:
        raise ValueError(f"unknown CMR key variant: {variant}")
    variants = payload["key_variants"]
    if variant not in variants:
        raise KeyError(f"cache does not contain CMR key variant: {variant}")

    feature = variants[variant].float().cpu()
    labels = payload["labels"].long().flatten().cpu()
    events = list(map(str, payload["events"]))
    if feature.ndim != 2 or feature.shape[0] != labels.numel():
        raise ValueError("CMR features and labels must be aligned matrices")
    if len(events) != labels.numel():
        raise ValueError("CMR event list length does not match labels")
    if not bool(torch.isfinite(feature).all()):
        raise FloatingPointError("CMR cache contains non-finite features")

    event_map = {name: index for index, name in enumerate(sorted(set(events)))}
    event_codes = torch.tensor([event_map[name] for name in events], dtype=torch.long)
    if print_schema:
        print("=" * 88)
        print(f"CMR cache      : {Path(path).resolve()}")
        print(f"Role           : {metadata.get('role')}")
        print(f"Variant        : {variant}")
        print(f"Regions        : {feature.shape[0]}")
        print(f"Feature dim    : {feature.shape[1]}")
        print(f"Class counts   : {torch.bincount(labels, minlength=3).tolist()}")
        print(f"Events         : {len(event_map)}")
        print("Test accessed  : False")
    return feature, labels, event_codes, event_map, payload


def frozen_memory_view(
    feature: torch.Tensor,
    labels: torch.Tensor,
    event_codes: torch.Tensor,
    train_cache: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the exact class-event balanced reference bank frozen by P0."""

    if "memory_indices" not in train_cache:
        raise KeyError(
            "Train cache does not contain 'memory_indices'. "
            "CMR-M1 must reuse the frozen CMR-P0 memory protocol."
        )
    indices = train_cache["memory_indices"]
    if not torch.is_tensor(indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
    indices = indices.cpu().long().reshape(-1)
    if indices.numel() == 0:
        raise RuntimeError("memory_indices is empty")
    if int(indices.min()) < 0 or int(indices.max()) >= feature.shape[0]:
        raise RuntimeError("memory_indices contains out-of-range values")
    if torch.unique(indices).numel() != indices.numel():
        raise RuntimeError("memory_indices contains duplicate references")
    return feature[indices], labels[indices], event_codes[indices], indices


def class_event_index(
    labels: torch.Tensor, event_codes: torch.Tensor
) -> Dict[int, Dict[int, torch.Tensor]]:
    pools: Dict[int, Dict[int, torch.Tensor]] = {}
    for cls in range(3):
        pools[cls] = {}
        for event in torch.unique(event_codes[labels == cls]).tolist():
            indices = torch.nonzero(
                (labels == cls) & (event_codes == int(event)), as_tuple=False
            ).flatten()
            if indices.numel():
                pools[cls][int(event)] = indices
        if len(pools[cls]) < 2:
            raise RuntimeError(
                f"class {cls} needs at least two events for cross-event positives"
            )
    return pools


def sample_class_event_batch(
    pools: Mapping[int, Mapping[int, torch.Tensor]],
    *,
    events_per_class: int,
    samples_per_event: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = []
    for cls in range(3):
        available = sorted(pools[cls])
        take_events = min(int(events_per_class), len(available))
        order = torch.randperm(len(available), generator=generator)[:take_events]
        for position in order.tolist():
            indices = pools[cls][available[int(position)]]
            if indices.numel() >= int(samples_per_event):
                local = torch.randperm(indices.numel(), generator=generator)[
                    : int(samples_per_event)
                ]
            else:
                local = torch.randint(
                    indices.numel(),
                    (int(samples_per_event),),
                    generator=generator,
                )
            selected.append(indices[local])
    if not selected:
        raise RuntimeError("class-event sampler produced an empty batch")
    batch = torch.cat(selected)
    return batch[torch.randperm(batch.numel(), generator=generator)]


@torch.no_grad()
def cross_event_retrieval_eval(
    reference_z: torch.Tensor,
    reference_y: torch.Tensor,
    reference_e: torch.Tensor,
    query_z: torch.Tensor,
    query_y: torch.Tensor,
    query_e: torch.Tensor,
    *,
    k_values: Sequence[int] = (1, 5, 10),
    ap_k: int = 10,
    query_chunk: int = 1024,
    device: torch.device | str | None = None,
    print_progress: bool = True,
) -> Dict[str, object]:
    """Vectorised cosine retrieval with same-event references excluded."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    if not k_values or min(k_values) <= 0 or int(ap_k) <= 0:
        raise ValueError("retrieval K values must be positive")

    reference_z = F.normalize(reference_z.float(), dim=1, eps=1e-8).to(device)
    reference_y = reference_y.long().flatten().to(device)
    reference_e = reference_e.long().flatten().to(device)
    query_z = F.normalize(query_z.float(), dim=1, eps=1e-8).cpu()
    query_y = query_y.long().flatten().cpu()
    query_e = query_e.long().flatten().cpu()
    if reference_z.shape[0] != reference_y.numel() or reference_y.numel() != reference_e.numel():
        raise ValueError("reference tensors must align")
    if query_z.shape[0] != query_y.numel() or query_y.numel() != query_e.numel():
        raise ValueError("query tensors must align")
    if reference_z.shape[1] != query_z.shape[1]:
        raise ValueError("reference and query embedding dimensions differ")

    max_k = max(max(k_values), int(ap_k))
    total_hits = {k: 0 for k in k_values}
    per_class_hits = {
        int(cls): {k: [0, 0] for k in k_values}
        for cls in torch.unique(query_y).tolist()
    }
    per_class_ap_sum = {int(cls): 0.0 for cls in torch.unique(query_y).tolist()}
    per_class_ap_count = {int(cls): 0 for cls in torch.unique(query_y).tolist()}
    event_codes_present = sorted(int(event) for event in torch.unique(query_e).tolist())
    per_event_accumulator = {
        event: {
            "count": 0,
            "r1_hits": 0,
            "ap_sum": 0.0,
            "class_ap_sum": {
                int(cls): 0.0 for cls in torch.unique(query_y).tolist()
            },
            "class_count": {
                int(cls): 0 for cls in torch.unique(query_y).tolist()
            },
        }
        for event in event_codes_present
    }
    num_classes = int(max(reference_y.max().item(), query_y.max().item()) + 1)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total_queries = 0
    ap_sum = 0.0

    for start in range(0, query_z.shape[0], int(query_chunk)):
        end = min(start + int(query_chunk), query_z.shape[0])
        qz = query_z[start:end].to(device, non_blocking=True)
        qy = query_y[start:end].to(device, non_blocking=True)
        qe = query_e[start:end].to(device, non_blocking=True)
        similarity = qz @ reference_z.t()
        same_event = qe[:, None] == reference_e[None, :]
        similarity.masked_fill_(same_event, float("-inf"))
        valid_count = (~same_event).sum(dim=1)
        if bool((valid_count == 0).any()):
            raise RuntimeError("at least one query has no cross-event reference")

        take_k = min(max_k, reference_z.shape[0])
        top_values, top_indices = torch.topk(
            similarity, k=take_k, dim=1, largest=True, sorted=True
        )
        valid_rank = torch.isfinite(top_values)
        ranked_labels = reference_y[top_indices]
        prediction = ranked_labels[:, 0]
        flat = qy * num_classes + prediction
        confusion += torch.bincount(
            flat.cpu(), minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

        for k in k_values:
            use_k = min(k, ranked_labels.shape[1])
            hit = (
                (ranked_labels[:, :use_k] == qy[:, None])
                & valid_rank[:, :use_k]
            ).any(dim=1)
            total_hits[k] += int(hit.sum())
            for cls in per_class_hits:
                class_mask = qy == int(cls)
                count = int(class_mask.sum())
                if count:
                    per_class_hits[cls][k][0] += int(hit[class_mask].sum())
                    per_class_hits[cls][k][1] += count

        use_ap = min(int(ap_k), ranked_labels.shape[1])
        relevant = (
            (ranked_labels[:, :use_ap] == qy[:, None])
            & valid_rank[:, :use_ap]
        ).float()
        ranks = torch.arange(
            1, use_ap + 1, device=device, dtype=torch.float32
        )[None, :]
        precision = relevant.cumsum(dim=1) / ranks
        relevant_count = relevant.sum(dim=1)
        average_precision = (precision * relevant).sum(dim=1)
        average_precision = torch.where(
            relevant_count > 0,
            average_precision / relevant_count.clamp_min(1.0),
            torch.zeros_like(average_precision),
        )
        r1_hit = prediction == qy
        for event in torch.unique(qe).tolist():
            event_mask = qe == int(event)
            event_stats = per_event_accumulator[int(event)]
            event_stats["count"] += int(event_mask.sum())
            event_stats["r1_hits"] += int(r1_hit[event_mask].sum())
            event_stats["ap_sum"] += float(average_precision[event_mask].sum())
            for cls in event_stats["class_count"]:
                class_event_mask = event_mask & (qy == int(cls))
                class_count = int(class_event_mask.sum())
                if class_count:
                    event_stats["class_ap_sum"][cls] += float(
                        average_precision[class_event_mask].sum()
                    )
                    event_stats["class_count"][cls] += class_count
        ap_sum += float(average_precision.sum())
        for cls in per_class_ap_sum:
            class_mask = qy == int(cls)
            class_count = int(class_mask.sum())
            if class_count:
                per_class_ap_sum[cls] += float(
                    average_precision[class_mask].sum()
                )
                per_class_ap_count[cls] += class_count
        total_queries += int(qy.numel())
        chunk_index = start // int(query_chunk)
        if print_progress and (
            start == 0 or end == query_z.shape[0] or chunk_index % 5 == 0
        ):
            print(f"Retrieval eval: {end}/{query_z.shape[0]}", flush=True)

    result: Dict[str, object] = {
        f"cross_event_R@{k}": total_hits[k] / max(total_queries, 1)
        for k in k_values
    }
    result[f"cross_event_mAP@{int(ap_k)}"] = ap_sum / max(total_queries, 1)
    result["per_class"] = {
        cls: {
            **{
                f"R@{k}": (
                    values[k][0] / values[k][1] if values[k][1] else 0.0
                )
                for k in k_values
            },
            f"mAP@{int(ap_k)}": (
                per_class_ap_sum[cls] / per_class_ap_count[cls]
                if per_class_ap_count[cls]
                else 0.0
            ),
        }
        for cls, values in per_class_hits.items()
    }
    class_map_values = [
        float(values[f"mAP@{int(ap_k)}"])
        for values in result["per_class"].values()
    ]
    result[f"macro_mAP@{int(ap_k)}"] = (
        sum(class_map_values) / len(class_map_values)
        if class_map_values
        else 0.0
    )
    result["nn_confusion"] = {
        f"{gt}->{pred}": int(confusion[gt, pred])
        for gt in range(num_classes)
        for pred in range(num_classes)
        if int(confusion[gt, pred]) > 0
    }
    result["num_queries"] = int(total_queries)
    result["num_reference"] = int(reference_z.shape[0])
    result["same_event_excluded"] = True
    result["per_event"] = {}
    for event, accumulator in per_event_accumulator.items():
        class_map = {
            cls: accumulator["class_ap_sum"][cls] / accumulator["class_count"][cls]
            for cls in accumulator["class_count"]
            if accumulator["class_count"][cls] > 0
        }
        result["per_event"][str(event)] = {
            "queries": int(accumulator["count"]),
            "R@1": accumulator["r1_hits"] / max(accumulator["count"], 1),
            f"mAP@{int(ap_k)}": accumulator["ap_sum"] / max(accumulator["count"], 1),
            f"macro_mAP@{int(ap_k)}": (
                sum(class_map.values()) / len(class_map) if class_map else 0.0
            ),
            "per_class_mAP": class_map,
        }
    return result


@torch.no_grad()
def relation_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    *,
    num_pairs: int = 200000,
    seed: int = 3407,
) -> Dict[str, object]:
    """Estimate whether class similarity dominates event similarity."""

    if int(num_pairs) < 3:
        raise ValueError("num_pairs must be at least three")
    x = F.normalize(features.float().cpu(), dim=1, eps=1e-8)
    labels = labels.long().flatten().cpu()
    events = events.long().flatten().cpu()
    if x.shape[0] != labels.numel() or labels.numel() != events.numel():
        raise ValueError("relation-statistic tensors must align")
    target = int(num_pairs) // 3
    generator = torch.Generator().manual_seed(int(seed))
    groups = {
        "same_class_cross_event": [],
        "diff_class_same_event": [],
        "diff_class_cross_event": [],
    }
    attempts = 0
    maximum_attempts = max(int(num_pairs) * 30, 10000)
    batch_size = min(max(target * 2, 4096), 262144)
    while min(sum(item.numel() for item in chunks) for chunks in groups.values()) < target:
        if attempts >= maximum_attempts:
            break
        draw = min(batch_size, maximum_attempts - attempts)
        first = torch.randint(x.shape[0], (draw,), generator=generator)
        second = torch.randint(x.shape[0], (draw,), generator=generator)
        attempts += draw
        not_self = first != second
        same_class = labels[first] == labels[second]
        same_event = events[first] == events[second]
        masks = {
            "same_class_cross_event": not_self & same_class & (~same_event),
            "diff_class_same_event": not_self & (~same_class) & same_event,
            "diff_class_cross_event": not_self & (~same_class) & (~same_event),
        }
        for name, mask in masks.items():
            have = sum(item.numel() for item in groups[name])
            need = target - have
            if need <= 0:
                continue
            chosen = torch.nonzero(mask, as_tuple=False).flatten()[:need]
            if chosen.numel():
                similarity = (x[first[chosen]] * x[second[chosen]]).sum(dim=1)
                groups[name].append(similarity)

    merged = {}
    for name, chunks in groups.items():
        values = torch.cat(chunks) if chunks else torch.empty(0)
        if values.numel() < target:
            raise RuntimeError(
                f"could not sample enough relation pairs for {name}: "
                f"{values.numel()} < {target}"
            )
        merged[name] = values[:target]
    same_cross = float(merged["same_class_cross_event"].mean())
    different_same = float(merged["diff_class_same_event"].mean())
    different_cross = float(merged["diff_class_cross_event"].mean())
    return {
        "same_class_cross_event_cos": same_cross,
        "diff_class_same_event_cos": different_same,
        "diff_class_cross_event_cos": different_cross,
        "delta_cross_event": same_cross - different_same,
        "num_pairs": {name: int(values.numel()) for name, values in merged.items()},
    }
