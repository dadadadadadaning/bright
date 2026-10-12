# -*- coding: utf-8 -*-
"""Pure scoring and sample-selection utilities for CMR-C0 visual diagnosis."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from model.cmr_c0.case_card import class_logmeanexp, directional_topk_score


VIS_SCORE_PROTOCOL = "CMR_C0_ISOLATED_visual_diagnostic_scores_v1"
DIAGNOSTIC_GROUPS = (
    "rescue_target",
    "harm",
    "critical_degradation",
    "maximum_disagreement",
)


def _support_lookup(support: Mapping[str, object], maximum_index: int) -> torch.Tensor:
    memory_local = support["memory_local_indices"].long()
    size = max(maximum_index + 1, int(memory_local.max()) + 1)
    lookup = torch.full((size,), -1, dtype=torch.long)
    lookup[memory_local] = torch.arange(memory_local.numel())
    return lookup


def _pair_variant_scores(
    query_features: torch.Tensor,
    memory_features: torch.Tensor,
    query_indices: Mapping[str, torch.Tensor],
    memory_indices: Mapping[str, torch.Tensor],
    variants: Sequence[str],
    *,
    neighbor_k: int,
    maximum_shared_similarity_elements: int = 2_000_000,
) -> Dict[str, torch.Tensor]:
    q = F.normalize(query_features.float(), dim=1, eps=1e-8)
    m = F.normalize(memory_features.float(), dim=1, eps=1e-8)
    if q.ndim != 2 or m.ndim != 2 or q.shape[1] != m.shape[1]:
        raise ValueError("query/memory feature matrices are incompatible")
    shared = (
        q @ m.t()
        if q.shape[0] * m.shape[0] <= int(maximum_shared_similarity_elements)
        else None
    )
    result: Dict[str, torch.Tensor] = {}
    for variant in variants:
        qi = query_indices[variant].long().to(q.device)
        mi = memory_indices[variant].long().to(m.device)
        if qi.numel() == 0 or mi.numel() == 0:
            raise RuntimeError(f"empty critical set for {variant}")
        if shared is None:
            result[variant] = directional_topk_score(
                q[qi], m[mi], neighbor_k=neighbor_k
            )
            continue
        similarity = shared.index_select(0, qi).index_select(1, mi)
        take = min(max(int(neighbor_k), 1), similarity.shape[1])
        result[variant] = similarity.topk(
            take, dim=1, largest=True, sorted=False
        ).values.mean()
    return result


def _local_feature(scores: Mapping[int, Sequence[float]], temperature: float) -> list[float]:
    intact = torch.tensor(list(scores[0]), dtype=torch.float32)
    damaged = torch.tensor(list(scores[1]), dtype=torch.float32)
    s_i = class_logmeanexp(intact, temperature)
    s_d = class_logmeanexp(damaged, temperature)
    return [
        float(s_i), float(s_d), float(s_d - s_i),
        float(intact.max()), float(damaged.max()),
        float(damaged.max() - intact.max()),
    ]


def _base_feature(
    coarse: Mapping[int, Sequence[float]],
    probability: torch.Tensor,
    best_damaged_rank: int,
    candidate_count: int,
    temperature: float,
) -> list[float]:
    intact = torch.tensor(list(coarse[0]), dtype=torch.float32)
    damaged = torch.tensor(list(coarse[1]), dtype=torch.float32)
    s_i = class_logmeanexp(intact, temperature)
    s_d = class_logmeanexp(damaged, temperature)
    sorted_probability = probability.float().sort(descending=True).values
    return [
        float(s_i), float(s_d), float(s_d - s_i),
        float(intact.max()), float(damaged.max()),
        float(damaged.max() - intact.max()),
        float(len(damaged)) / float(len(damaged) + len(intact)),
        1.0 - float(sorted_probability[0] - sorted_probability[1]),
        float(probability[1:].sum()),
        float(best_damaged_rank) / float(candidate_count),
    ]


@torch.inference_mode()
def score_validation_queries(
    query: Mapping[str, object],
    support: Mapping[str, object],
    *,
    variants: Sequence[str],
    device: str,
    neighbor_k: int = 3,
    class_temperature: float = 0.10,
    progress_every: int = 100,
) -> list[Dict[str, object]]:
    """Recompute only the frozen per-query evidence needed for visual diagnosis."""
    variants = tuple(map(str, variants))
    if "all" not in variants or len(set(variants)) != len(variants):
        raise ValueError("visual diagnosis requires unique variants including 'all'")
    for variant in variants:
        if variant not in query["critical_indices"] or variant not in support["critical_indices"]:
            raise KeyError(f"cache is missing variant {variant}")
    auditable = (query["labels"].long() == 0) | (query["labels"].long() == 1)
    rows = torch.nonzero(auditable, as_tuple=False).flatten()
    lookup = _support_lookup(support, int(query["candidate_indices"].max()))
    run_device = torch.device(device)
    records: list[Dict[str, object]] = []

    for progress, row in enumerate(rows.tolist(), 1):
        candidate_indices = query["candidate_indices"][row].long()
        candidate_labels = query["candidate_labels"][row].long()
        candidate_scores = query["candidate_scores"][row].float()
        local: Dict[str, Dict[int, list[float]]] = {
            variant: {0: [], 1: []} for variant in variants
        }
        best: Dict[str, Dict[int, Dict[str, object] | None]] = {
            variant: {0: None, 1: None} for variant in variants
        }
        coarse: Dict[int, list[float]] = {0: [], 1: []}
        best_damaged_rank = -1
        best_damaged_coarse = -float("inf")
        query_features = query["features"][row].to(run_device)
        query_indices = {
            variant: query["critical_indices"][variant][row]
            for variant in variants
        }
        for rank, (memory_index, label, coarse_score) in enumerate(
            zip(candidate_indices.tolist(), candidate_labels.tolist(), candidate_scores.tolist()), 1
        ):
            label = int(label)
            if label not in (0, 1):
                continue
            if memory_index >= lookup.numel() or int(lookup[memory_index]) < 0:
                continue
            support_row = int(lookup[memory_index])
            pair_scores = _pair_variant_scores(
                query_features,
                support["features"][support_row].to(run_device),
                query_indices,
                {
                    variant: support["critical_indices"][variant][support_row]
                    for variant in variants
                },
                variants,
                neighbor_k=neighbor_k,
            )
            for variant, tensor_score in pair_scores.items():
                score = float(tensor_score.detach().cpu())
                local[variant][label].append(score)
                previous = best[variant][label]
                if previous is None or score > float(previous["local_score"]):
                    best[variant][label] = {
                        "support_row": support_row,
                        "memory_index": int(memory_index),
                        "rank": int(rank),
                        "local_score": score,
                        "coarse_score": float(coarse_score),
                    }
            coarse[label].append(float(coarse_score))
            if label == 1 and float(coarse_score) > best_damaged_coarse:
                best_damaged_coarse = float(coarse_score)
                best_damaged_rank = int(rank)

        if not coarse[0] or not coarse[1]:
            continue
        if any(not local[variant][0] or not local[variant][1] for variant in variants):
            raise RuntimeError("visual scoring lost variant/class alignment")
        base_feature = _base_feature(
            coarse, query["base_probabilities"][row], best_damaged_rank,
            int(candidate_indices.numel()), class_temperature,
        )
        variant_records: Dict[str, object] = {}
        for variant in variants:
            local_feature = _local_feature(local[variant], class_temperature)
            variant_records[variant] = {
                "local_feature": local_feature,
                "s_i": local_feature[0],
                "s_d": local_feature[1],
                "delta_d_i": local_feature[2],
                "direct_prediction": int(local_feature[2] > 0.0),
                "best_intact": best[variant][0],
                "best_damaged": best[variant][1],
            }
        records.append({
            "query_row": int(row),
            "source_region_index": int(query["source_region_indices"][row]),
            "image_id": str(query["image_ids"][row]),
            "event": str(query["events"][row]),
            "disaster_type": str(query["disaster_types"][row]),
            "label": int(query["labels"][row]),
            "base_prediction": int(query["base_predictions"][row]),
            "base_probabilities": query["base_probabilities"][row].float().tolist(),
            "bbox": query["bboxes"][row].long().tolist(),
            "area": int(query["areas"][row]),
            "identity_mode": str(query["identity_mode"][row]),
            "region_fingerprint": str(query["region_fingerprints"][row]),
            "base_feature": base_feature,
            "variants": variant_records,
        })
        if progress_every > 0 and (progress % progress_every == 0 or progress == rows.numel()):
            print(
                f"CMR-C0-VIS scores: {progress}/{rows.numel()} retained={len(records)}",
                flush=True,
            )
    if not records:
        raise RuntimeError("visual diagnosis retained no Validation records")
    return records


def apply_json_calibrators(
    records: list[Dict[str, object]],
    audit: Mapping[str, object],
    variants: Sequence[str],
) -> None:
    """Attach frozen Validation probabilities from Train-fitted calibrators."""
    for variant in variants:
        group = audit["results"][f"{variant}_combined"]
        model = group["final_calibrator"]
        mean = torch.tensor(model["mean"], dtype=torch.float32)
        std = torch.tensor(model["std"], dtype=torch.float32).clamp_min(1e-8)
        coefficient = torch.tensor(model["coefficient"], dtype=torch.float32)
        intercept = float(model["intercept"])
        features = torch.tensor([
            list(record["base_feature"])
            + list(record["variants"][variant]["local_feature"])
            for record in records
        ], dtype=torch.float32)
        if features.shape[1] != coefficient.numel():
            raise RuntimeError(f"calibrator feature mismatch for {variant}")
        score = torch.sigmoid(((features - mean) / std) @ coefficient + intercept)
        for record, value in zip(records, score.tolist()):
            record["variants"][variant]["combined_probability"] = float(value)


def _event_balanced_top(
    candidates: Sequence[Mapping[str, object]],
    *,
    rank_key: str,
    per_event: int,
    maximum: int,
) -> list[int]:
    by_event: Dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in candidates:
        by_event[str(record["event"])].append(record)
    chosen: list[Mapping[str, object]] = []
    for event in sorted(by_event):
        rows = sorted(
            by_event[event], key=lambda item: float(item[rank_key]), reverse=True
        )
        chosen.extend(rows[:max(int(per_event), 1)])
    chosen.sort(key=lambda item: float(item[rank_key]), reverse=True)
    return [int(item["query_row"]) for item in chosen[:max(int(maximum), 1)]]


def select_diagnostic_samples(
    records: Sequence[Dict[str, object]],
    *,
    selected_variant: str,
    per_event: int = 2,
    maximum_per_group: int = 24,
) -> Dict[str, list[int]]:
    """Select four fixed diagnostic groups without using Test."""
    prepared: list[Dict[str, object]] = []
    for source in records:
        record = dict(source)
        label = int(record["label"])
        target = label == 1
        p3_predicted_intact = int(record["base_prediction"]) == 0
        all_row = record["variants"]["all"]
        critical_row = record["variants"][selected_variant]
        all_prediction = bool(int(all_row["direct_prediction"]))
        critical_prediction = bool(int(critical_row["direct_prediction"]))
        all_probability = float(all_row["combined_probability"])
        critical_probability = float(critical_row["combined_probability"])
        record["is_rescue_target"] = (
            target and p3_predicted_intact and critical_prediction
        )
        record["is_harm"] = (
            (not target) and p3_predicted_intact and critical_prediction
        )
        record["is_degradation"] = (
            all_prediction == target and critical_prediction != target
        )
        record["rescue_rank"] = float(critical_row["delta_d_i"])
        record["harm_rank"] = float(critical_row["delta_d_i"])
        record["degradation_rank"] = abs(
            float(critical_row["delta_d_i"]) - float(all_row["delta_d_i"])
        )
        record["disagreement_rank"] = abs(critical_probability - all_probability)
        prepared.append(record)

    definitions = {
        "rescue_target": (
            [row for row in prepared if row["is_rescue_target"]], "rescue_rank"
        ),
        "harm": ([row for row in prepared if row["is_harm"]], "harm_rank"),
        "critical_degradation": (
            [row for row in prepared if row["is_degradation"]], "degradation_rank"
        ),
        "maximum_disagreement": (prepared, "disagreement_rank"),
    }
    return {
        group: _event_balanced_top(
            candidates, rank_key=rank_key, per_event=per_event,
            maximum=maximum_per_group,
        )
        for group, (candidates, rank_key) in definitions.items()
    }


def cosine_similarity_matrix(
    query_features: torch.Tensor,
    memory_features: torch.Tensor,
    query_indices: torch.Tensor,
    memory_indices: torch.Tensor,
) -> torch.Tensor:
    q = F.normalize(query_features[query_indices.long()].float(), dim=1, eps=1e-8)
    m = F.normalize(memory_features[memory_indices.long()].float(), dim=1, eps=1e-8)
    if q.numel() == 0 or m.numel() == 0:
        raise ValueError("similarity matrix requires non-empty critical sets")
    return q @ m.t()


def expanded_crop_box(
    bbox: Sequence[int],
    image_shape: Sequence[int],
    *,
    padding_fraction: float = 0.25,
    minimum_padding: int = 12,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, bbox)
    height, width = map(int, image_shape)
    pad_x = max(int(minimum_padding), int(math.ceil((x2 - x1) * padding_fraction)))
    pad_y = max(int(minimum_padding), int(math.ceil((y2 - y1) * padding_fraction)))
    return (
        max(0, x1 - pad_x), max(0, y1 - pad_y),
        min(width, x2 + pad_x), min(height, y2 + pad_y),
    )


def validate_visual_score_payload(payload: Mapping[str, object]) -> None:
    if payload.get("protocol") != VIS_SCORE_PROTOCOL:
        raise RuntimeError("unknown CMR-C0 visual-score protocol")
    if bool(payload.get("test_accessed", True)):
        raise RuntimeError("CMR-C0 visual diagnosis refuses Test-derived scores")
    variants = list(payload.get("variants", []))
    if "all" not in variants or len(set(variants)) != len(variants):
        raise RuntimeError("invalid visual-score variants")
    records = payload.get("records", [])
    if not records:
        raise RuntimeError("visual-score payload contains no records")
    rows = [int(record["query_row"]) for record in records]
    if len(rows) != len(set(rows)):
        raise RuntimeError("visual-score query rows are not unique")


def summarize_manual_observations(
    rows: Sequence[Mapping[str, object]],
    annotation_fields: Sequence[str],
) -> Dict[str, object]:
    """Count completed qualitative annotations by group and event.

    Values are deliberately treated as observer-provided categories.  This
    utility summarizes them but never converts qualitative notes into a new
    performance metric or an automatic go/no-go decision.
    """
    if not rows:
        raise ValueError("manual observation table is empty")
    fields = tuple(map(str, annotation_fields))
    if not fields:
        raise ValueError("annotation_fields is empty")

    def summarise(partition: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        field_rows: Dict[str, object] = {}
        for field in fields:
            values = [
                str(row.get(field, "")).strip()
                for row in partition
                if str(row.get(field, "")).strip()
            ]
            counts = Counter(values)
            field_rows[field] = {
                "annotated": len(values),
                "missing": len(partition) - len(values),
                "counts": dict(sorted(counts.items())),
                "most_common": (
                    counts.most_common(1)[0][0] if counts else None
                ),
            }
        return {"cases": len(partition), "fields": field_rows}

    by_group: Dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_group_event: Dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        group = str(row.get("group", "")).strip()
        event = str(row.get("event", "")).strip()
        if not group or not event:
            raise ValueError("every observation requires group and event")
        by_group[group].append(row)
        by_group_event[(group, event)].append(row)
    return {
        "scope": "qualitative annotations only; no automatic scientific decision",
        "overall": summarise(rows),
        "by_group": {
            group: summarise(partition)
            for group, partition in sorted(by_group.items())
        },
        "by_group_event": {
            f"{group}::{event}": summarise(partition)
            for (group, event), partition in sorted(by_group_event.items())
        },
    }


__all__ = [
    "DIAGNOSTIC_GROUPS", "VIS_SCORE_PROTOCOL", "apply_json_calibrators",
    "cosine_similarity_matrix", "expanded_crop_box",
    "score_validation_queries", "select_diagnostic_samples",
    "summarize_manual_observations",
    "validate_visual_score_payload",
]
