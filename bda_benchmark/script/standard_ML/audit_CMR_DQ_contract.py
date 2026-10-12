# -*- coding: utf-8 -*-
"""No-training contract audit for the train-only dynamic Memory queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.cmr.dynamic_queue import CMREventClassQueue
from script.standard_ML.cmr_metric_utils import load_cmr_cache


def main() -> None:
    parser = argparse.ArgumentParser("CMR train-only dynamic queue audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capacity_per_event_class", type=int, default=200)
    parser.add_argument("--positive_k", type=int, default=5)
    parser.add_argument("--negative_k", type=int, default=10)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--minimum_purity", type=float, default=0.75)
    parser.add_argument("--minimum_coverage", type=float, default=0.50)
    args = parser.parse_args()

    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    purity = torch.as_tensor(cache.get("purity", cache.get("purities", torch.ones(labels.numel())))).float()
    coverage = torch.as_tensor(cache.get("coverage", cache.get("coverages", torch.ones(labels.numel())))).float()
    quality = (purity >= args.minimum_purity) & (coverage >= args.minimum_coverage)
    queue = CMREventClassQueue(
        feature.shape[1], len(event_map), capacity_per_cell=args.capacity_per_event_class
    )
    written = queue.enqueue(
        feature, labels, events, confidence=purity * coverage, quality_mask=quality
    )
    anchors = (labels == 1) & quality
    positive = queue.retrieve_cross_event(
        feature[anchors], events[anchors], class_index=1,
        topk=args.positive_k, maximum_per_event=args.max_per_event,
    )
    negative = queue.retrieve_cross_event(
        feature[anchors], events[anchors], class_index=0,
        topk=args.negative_k, maximum_per_event=args.max_per_event,
    )
    positive_complete = positive["valid"].sum(dim=1) == args.positive_k
    negative_complete = negative["valid"].sum(dim=1) == args.negative_k
    positive_events = torch.tensor([
        len(set(row[row >= 0].tolist())) for row in positive["events"]
    ])
    same_event_positive = bool(
        ((positive["events"] == events[anchors, None]) & positive["valid"]).any()
    )
    same_event_negative = bool(
        ((negative["events"] == events[anchors, None]) & negative["valid"]).any()
    )
    positive_mean = positive["scores"][positive["valid"]].mean()
    negative_mean = negative["scores"][negative["valid"]].mean()
    occupied = queue.valid.any(dim=2)
    occurrence_event_coverage = int((occupied[0] & occupied[1]).sum())
    inverse_event_map = {value: key for key, value in event_map.items()}
    empty_cells = [
        {"class": int(cls), "event": inverse_event_map[int(event)]}
        for cls, event in torch.nonzero(~occupied, as_tuple=False).tolist()
    ]
    checks = {
        "at_least_95pct_positive_lists_complete": float(positive_complete.float().mean()) >= 0.95,
        "at_least_95pct_negative_lists_complete": float(negative_complete.float().mean()) >= 0.95,
        "positive_support_covers_at_least_three_events": float((positive_events >= 3).float().mean()) >= 0.95,
        "same_event_support_is_excluded": not same_event_positive and not same_event_negative,
        "damaged_positive_more_similar_than_intact_hard_negative": float(positive_mean - negative_mean) >= 0.03,
        # H1-DQ uses Damaged positives and Intact negatives. Destroyed is
        # explicitly ignored, so requiring every X/event cell made the old
        # contract stricter than the declared training objective.
        "intact_and_damaged_queue_cover_at_least_10_events": occurrence_event_coverage >= 10,
    }
    payload = {
        "protocol": "CMR_DQ_event_class_train_only_queue_audit_v1",
        "scope": "train-only detached frozen Mean keys; no inference Memory",
        "queue": {
            "capacity_per_event_class": args.capacity_per_event_class,
            "written": written,
            "valid_entries": int(queue.valid.sum()),
            "total_seen": queue.total_seen.tolist(),
            "occupied_event_class_cells": occupied.tolist(),
            "empty_event_class_cells": empty_cells,
            "intact_and_damaged_joint_event_coverage": occurrence_event_coverage,
        },
        "anchors": int(anchors.sum()),
        "positive_complete_rate": float(positive_complete.float().mean()),
        "negative_complete_rate": float(negative_complete.float().mean()),
        "positive_event_coverage_rate": float((positive_events >= 3).float().mean()),
        "positive_similarity": float(positive_mean),
        "intact_hard_negative_similarity": float(negative_mean),
        "positive_minus_negative": float(positive_mean - negative_mean),
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "ADMIT_DYNAMIC_QUEUE_TO_H1_TRAINING_ONLY"
            if all(checks.values()) else "STOP_DYNAMIC_QUEUE_CONTRACT_FAILED"
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 104)
    print("CMR-DQ train-only event x class queue audit")
    print(f"Written/valid  : {written}/{int(queue.valid.sum())}")
    print(f"D anchors      : {int(anchors.sum())}")
    print(f"D+/I- cosine   : {float(positive_mean):.4f}/{float(negative_mean):.4f}")
    print(f"Checks         : {checks}")
    print(f"Decision       : {payload['decision']}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
