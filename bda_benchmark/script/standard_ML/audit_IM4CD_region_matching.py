# -*- coding: utf-8 -*-
"""Validation-only IM4CD-to-BRIGHT matching evidence audit.

The audit compares the historical co-located feature difference against the
paper-inspired hard local response-peak geometry.  It never changes P3 logits,
never updates memory with validation samples, and never accepts a test list.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from script.standard_ML.audit_RM_region_retrieval import load_cache
from script.standard_ML.im4cd_matching_runtime import (
    binary_auroc,
    build_augmented_keys,
)
from script.standard_ML.rm_region_retrieval_runtime import (
    class_balanced_knn,
    complement_audit,
    metrics_from_predictions,
    validate_disjoint_ids,
)


def _probe_signal(
    query: Mapping[str, object],
    family: str,
    names: Sequence[str],
) -> Dict[str, object]:
    vectors = query[family].float()
    labels = query["labels"].long()
    base = query["base_predictions"].long()
    # The principal BRIGHT error pool: P3 missed Damaged versus confidently
    # retained Intact among regions that P3 calls Intact.
    eligible = ((labels == 1) & (base == 0)) | ((labels == 0) & (base == 0))
    target = ((labels == 1) & (base == 0))[eligible].long()
    events = [
        query["events"][int(index)]
        for index in torch.nonzero(eligible).flatten()
    ]
    rows = []
    for index, name in enumerate(names):
        score = vectors[eligible, index]
        auc = binary_auroc(target, score)
        direction = 1.0 if auc >= 0.5 else -1.0
        oriented = auc if auc >= 0.5 else 1.0 - auc
        event_values = {}
        for event in sorted(set(events)):
            mask = torch.tensor([item == event for item in events], dtype=torch.bool)
            if int(target[mask].sum()) == 0 or int((target[mask] == 0).sum()) == 0:
                continue
            value = binary_auroc(target[mask], direction * score[mask])
            if math.isfinite(value):
                event_values[event] = value
        rows.append(
            {
                "name": str(name),
                "auroc": float(oriented),
                "direction": "higher_is_missed_damaged" if direction > 0 else "lower_is_missed_damaged",
                "event_macro_auroc": (
                    sum(event_values.values()) / len(event_values)
                    if event_values
                    else float("nan")
                ),
                "comparable_events": len(event_values),
                "event_auroc": event_values,
                "positive_missed_damaged": int(target.sum()),
                "negative_correct_intact": int((target == 0).sum()),
            }
        )
    return max(
        rows,
        key=lambda row: (
            row["event_macro_auroc"] if math.isfinite(row["event_macro_auroc"]) else -1.0,
            row["auroc"],
        ),
    ) | {"all_signals": rows}


def _retrieval(
    memory: Mapping[str, object],
    query: Mapping[str, object],
    *,
    family: str,
    normalizer_key: str,
    weight: float,
    topk: int,
    beta: float,
    max_neighbors_per_event: int,
    device: torch.device,
) -> Dict[str, object]:
    normalizer = memory["metadata"][normalizer_key]
    memory_key = build_augmented_keys(
        memory["keys"], memory[family], normalizer, weight
    ).to(device)
    query_key = build_augmented_keys(
        query["keys"], query[family], normalizer, weight
    ).to(device)
    result = class_balanced_knn(
        query_key,
        memory_key,
        memory["labels"],
        topk_per_class=topk,
        beta=beta,
        query_events=query["events"],
        memory_events=memory["events"],
        exclude_same_event=True,
        max_neighbors_per_event=max_neighbors_per_event,
    )
    prediction = result["prediction"]
    metrics = metrics_from_predictions(query["labels"], prediction, query["events"])
    base_metrics = metrics_from_predictions(
        query["labels"], query["base_predictions"], query["events"]
    )
    event_delta = {
        event: 100.0
        * (metrics["events"][event]["macro_f1"] - base_metrics["events"][event]["macro_f1"])
        for event in metrics["events"]
    }
    return {
        "family": family,
        "weight": float(weight),
        "metrics": metrics,
        "base_metrics": base_metrics,
        "complement": complement_audit(
            query["labels"], query["base_predictions"], prediction
        ),
        "event_macro_f1_delta_pp_vs_p3": event_delta,
        "nonnegative_events_vs_p3": sum(value >= 0.0 for value in event_delta.values()),
        "mean_unique_neighbor_events_by_class": [
            float(value)
            for value in result["unique_neighbor_events_by_class"].mean(dim=0)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser("Audit IM4CD hard-peak evidence in BRIGHT regions")
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default="0,0.25,0.5,1.0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    args = parser.parse_args()

    memory = load_cache(args.memory, "memory")
    query = load_cache(args.queries, "query")
    validate_disjoint_ids(memory["image_ids"], query["image_ids"])
    for cache in (memory, query):
        if not bool(cache["metadata"].get("matching_stats_enabled", False)):
            raise RuntimeError("IM4CD audit requires caches built with --matching_stats")
        for key in ("colocated_stats", "matching_stats"):
            if key not in cache:
                raise KeyError(f"matching cache is missing {key!r}")
    if memory["metadata"]["p3_checkpoint_sha256"] != query["metadata"]["p3_checkpoint_sha256"]:
        raise RuntimeError("memory/query P3 checkpoint mismatch")

    weights = [float(value) for value in args.weights.split(",") if value.strip()]
    device = torch.device(args.device)
    colocated_probe = _probe_signal(
        query,
        "colocated_stats",
        query["metadata"]["colocated_stat_names"],
    )
    matching_probe = _probe_signal(
        query,
        "matching_stats",
        query["metadata"]["matching_stat_names"],
    )
    colocated = [
        _retrieval(
            memory,
            query,
            family="colocated_stats",
            normalizer_key="colocated_stat_normalizer",
            weight=weight,
            topk=args.topk,
            beta=args.beta,
            max_neighbors_per_event=args.max_neighbors_per_event,
            device=device,
        )
        for weight in weights
    ]
    matching = [
        _retrieval(
            memory,
            query,
            family="matching_stats",
            normalizer_key="matching_stat_normalizer",
            weight=weight,
            topk=args.topk,
            beta=args.beta,
            max_neighbors_per_event=args.max_neighbors_per_event,
            device=device,
        )
        for weight in weights
    ]
    order = lambda row: (
        row["metrics"]["event_macro_f1"],
        row["metrics"]["macro_f1"],
        row["metrics"]["class_recall"][1],
    )
    best_colocated = max(colocated, key=order)
    best_matching = max(matching, key=order)
    event_gain_pp = 100.0 * (
        best_matching["metrics"]["event_macro_f1"]
        - best_colocated["metrics"]["event_macro_f1"]
    )
    checks = {
        "matching_probe_beats_colocated_by_0_03_auc": (
            matching_probe["auroc"] >= colocated_probe["auroc"] + 0.03
        ),
        "matching_probe_event_macro_at_least_0_60": (
            matching_probe["event_macro_auroc"] >= 0.60
        ),
        "matching_retrieval_event_macro_plus_0_30pp": event_gain_pp >= 0.30,
        "damaged_net_complement_positive": (
            best_matching["complement"]["per_class"]["damaged"]["net_complement"] > 0
        ),
        "ten_events_nondecreasing_vs_p3": best_matching["nonnegative_events_vs_p3"] >= 10,
    }
    payload: Dict[str, object] = {
        "protocol": "IM4CD_BRIGHT_region_matching_audit_v1",
        "scope": "validation-only parameter-free evidence audit; not full IM4CD reproduction",
        "global_motion_model": "dominant discrete translation on frozen P3 feature grid",
        "memory": str(Path(args.memory).resolve()),
        "queries": str(Path(args.queries).resolve()),
        "colocated_probe": colocated_probe,
        "matching_probe": matching_probe,
        "colocated_retrieval": colocated,
        "matching_retrieval": matching,
        "best_colocated": best_colocated,
        "best_matching": best_matching,
        "matching_event_macro_gain_pp_vs_colocated": event_gain_pp,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "ADMIT_MATCHING_STATS_TO_RM_RETRIEVAL_FUSION"
            if all(checks.values())
            else "STOP_MATCHING_TRANSFER_DO_NOT_ADD_PIXEL_RESIDUAL"
        ),
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print("IM4CD-to-BRIGHT parameter-free region matching audit")
    print(
        f"Co-located probe : {colocated_probe['name']} | AUC={colocated_probe['auroc']:.4f} "
        f"| event={colocated_probe['event_macro_auroc']:.4f}"
    )
    print(
        f"Matching probe   : {matching_probe['name']} | AUC={matching_probe['auroc']:.4f} "
        f"| event={matching_probe['event_macro_auroc']:.4f}"
    )
    print(
        f"Co-located kNN   : w={best_colocated['weight']} | macro={100*best_colocated['metrics']['macro_f1']:.4f} "
        f"| event={100*best_colocated['metrics']['event_macro_f1']:.4f}"
    )
    print(
        f"Matching kNN     : w={best_matching['weight']} | macro={100*best_matching['metrics']['macro_f1']:.4f} "
        f"| event={100*best_matching['metrics']['event_macro_f1']:.4f}"
    )
    print(f"Checks           : {checks}")
    print(f"Decision         : {payload['decision']}")
    print(f"Saved            : {output}")
    print("Test accessed    : False")


if __name__ == "__main__":
    main()
