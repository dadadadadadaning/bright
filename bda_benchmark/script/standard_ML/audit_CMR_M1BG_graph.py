# -*- coding: utf-8 -*-
"""CMR-M1BG-A: build and audit a train-only global Damaged graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from script.standard_ML.cmr_global_graph_utils import (
    build_event_diverse_graph,
    graph_audit,
    graph_gate,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M1BG global Damaged graph audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--graph_output", required=True)
    parser.add_argument("--audit_output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    damaged_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    damaged_feature = feature[damaged_indices]
    damaged_events = events[damaged_indices]
    neighbours, similarities = build_event_diverse_graph(
        damaged_feature,
        damaged_events,
        topk=args.topk,
        max_per_event=args.max_per_event,
        chunk_size=args.chunk_size,
        device=args.device,
        print_progress=True,
    )
    audit = graph_audit(
        damaged_feature,
        damaged_events,
        neighbours,
        similarities,
        max_per_event=args.max_per_event,
        seed=args.seed,
    )
    checks = graph_gate(audit)
    passed = all(checks.values())
    graph_path = Path(args.graph_output)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": "CMR_M1BG_train_only_global_damaged_graph_v1",
            "damaged_train_indices": damaged_indices,
            "neighbour_train_indices": damaged_indices[neighbours],
            "neighbour_cosine": similarities,
            "damaged_events": damaged_events,
            "event_map": event_map,
            "config": vars(args),
            "p3_checkpoint_sha256": cache["metadata"].get("p3_checkpoint_sha256"),
            "audit": audit,
            "checks": checks,
            "passed": passed,
            "test_accessed": False,
        },
        graph_path,
    )
    decision = (
        "ADMIT_GLOBAL_GRAPH_TO_DAMAGED_ANCHOR_TRAINING"
        if passed
        else "STOP_GLOBAL_GRAPH_DO_NOT_TRAIN_OR_FUSE"
    )
    payload = {
        "protocol": "CMR_M1BG_global_graph_audit_v1",
        "graph": str(graph_path),
        "audit": audit,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "test_accessed": False,
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=" * 104)
    print("CMR-M1BG global Damaged graph audit")
    print(f"Anchors        : {audit['anchors']}")
    print(f"Top-k/random   : {audit['topk_cosine_mean']:.4f}/{audit['random_cosine_mean']:.4f}")
    print(f"Similarity gap : {audit['similarity_gap']:+.4f}")
    print(f"Unique events  : {audit['unique_neighbour_events_mean']:.2f}")
    print(f"Mutual gain    : {audit['mutual_neighbour_gain']:+.4f}")
    print(f"Checks         : {checks}")
    print(f"Decision       : {decision}")
    print(f"Saved graph    : {graph_path}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
