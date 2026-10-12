# -*- coding: utf-8 -*-
"""CMR-H0: mine and audit Train-only P3 occurrence hard examples.

This stage reads only the natural Train CMR cache.  It creates fixed
cross-event Damaged positives and high-similarity Intact negatives for P3
training.  It never trains a model, never changes a prediction, and refuses
Validation/Test caches through the cache-role contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from script.standard_ML.cmr_hard_mining_utils import (
    event_diverse_cross_event_topk,
    hard_mining_audit,
    hard_mining_gate,
    select_missed_damaged_anchors,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser("CMR-H0 Train-only hard-example audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--pairs_output", required=True)
    parser.add_argument("--audit_output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--positive_topk", type=int, default=5)
    parser.add_argument("--negative_topk", type=int, default=10)
    parser.add_argument("--positive_max_per_event", type=int, default=1)
    parser.add_argument("--negative_max_per_event", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache,
        "mean",
        expected_role="train",
        print_schema=True,
    )
    metadata = cache["metadata"]
    if metadata.get("source_split") != "train":
        raise RuntimeError("CMR-H0 accepts only an exact Train cache")
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("CMR-H0 refuses any cache that accessed Test")
    if "base_predictions" not in cache:
        raise KeyError("CMR-H0 requires frozen P3 base_predictions in Train cache")
    base_predictions = cache["base_predictions"].long().flatten().cpu()
    if base_predictions.shape != labels.shape:
        raise RuntimeError("base_predictions and labels do not align")

    anchor_indices = select_missed_damaged_anchors(labels, base_predictions)
    if anchor_indices.numel() == 0:
        raise RuntimeError("no GT-Damaged/P3-Intact hard anchors were found")
    damaged_reference_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    intact_reference_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()

    positive_indices, positive_scores, positive_unique_events = (
        event_diverse_cross_event_topk(
            feature[anchor_indices],
            events[anchor_indices],
            feature[damaged_reference_indices],
            events[damaged_reference_indices],
            damaged_reference_indices,
            topk=args.positive_topk,
            max_per_event=args.positive_max_per_event,
            chunk_size=args.chunk_size,
            device=args.device,
            print_prefix="CMR-H0 Damaged positives",
        )
    )
    negative_indices, negative_scores, negative_unique_events = (
        event_diverse_cross_event_topk(
            feature[anchor_indices],
            events[anchor_indices],
            feature[intact_reference_indices],
            events[intact_reference_indices],
            intact_reference_indices,
            topk=args.negative_topk,
            max_per_event=args.negative_max_per_event,
            chunk_size=args.chunk_size,
            device=args.device,
            print_prefix="CMR-H0 hard Intact negatives",
        )
    )

    audit = hard_mining_audit(
        feature,
        labels,
        base_predictions,
        events,
        anchor_indices,
        positive_indices,
        positive_scores,
        positive_unique_events,
        negative_indices,
        negative_scores,
        negative_unique_events,
        max_per_event=args.positive_max_per_event,
        seed=args.seed,
    )
    checks = hard_mining_gate(audit)
    passed = all(checks.values())
    decision = (
        "ADMIT_CMR_H0_TO_TRAIN_ONLY_P3_H1_DO_NOT_FUSE_OR_TEST"
        if passed
        else "STOP_OR_REVISE_HARD_PAIRS_DO_NOT_TRAIN_FUSE_OR_TEST"
    )

    pair_payload = {
        "protocol": "CMR_H0_train_only_occurrence_hard_pairs_v1",
        "source_cache": str(Path(args.train_cache).resolve()),
        "source_split": "train",
        "feature_variant": "mean",
        "anchor_definition": "gt_damaged_and_frozen_p3_intact",
        "positive_definition": "gt_damaged_cross_event_event_diverse",
        "negative_definition": "gt_intact_cross_event_high_cosine_event_diverse",
        "anchor_indices": anchor_indices,
        "positive_indices": positive_indices,
        "positive_cosine": positive_scores,
        "negative_indices": negative_indices,
        "negative_cosine": negative_scores,
        "anchor_events": events[anchor_indices],
        "event_map": event_map,
        "anchor_image_ids": [cache["image_ids"][int(i)] for i in anchor_indices],
        "config": vars(args),
        "p3_checkpoint": metadata.get("p3_checkpoint"),
        "p3_checkpoint_sha256": metadata.get("p3_checkpoint_sha256"),
        "audit": audit,
        "checks": checks,
        "passed": passed,
        "inference_uses_memory": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    pairs_path = Path(args.pairs_output)
    atomic_torch_save(pair_payload, pairs_path)

    result = {
        "protocol": "CMR_H0_train_only_occurrence_hard_mining_audit_v1",
        "pairs": str(pairs_path),
        "source_split": "train",
        "audit": audit,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "inference_uses_memory": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 112)
    print("CMR-H0 Train-only occurrence hard-example audit")
    print(f"Train regions      : {audit['train_regions']}")
    print(f"Damaged anchors    : {audit['missed_damaged_anchors']} / {audit['total_damaged_regions']}")
    print(f"Anchor events      : {audit['anchor_events']} | max share={audit['maximum_anchor_event_share']:.3f}")
    print(
        f"D-positive cosine : {audit['positive_cosine_mean']:.4f} | "
        f"random-D={audit['random_damaged_cosine_mean']:.4f} | "
        f"gain={audit['positive_similarity_gain_over_random']:+.4f}"
    )
    print(
        f"Hard-I cosine      : {audit['hard_intact_cosine_mean']:.4f} | "
        f"D+ beats I-={100*audit['top1_positive_beats_hard_intact_fraction']:.2f}%"
    )
    print(f"Checks             : {checks}")
    print(f"Decision           : {decision}")
    print(f"Saved pairs        : {pairs_path}")
    print(f"Saved audit        : {audit_path}")
    print("Validation accessed: False")
    print("Test accessed      : False")


if __name__ == "__main__":
    main()
