# -*- coding: utf-8 -*-
"""CMR-M3O: no-training Top-20 oracle/headroom audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from script.standard_ML.cmr_metric_utils import frozen_memory_view, load_cmr_cache
from script.standard_ML.cmr_pairwise_utils import (
    candidate_metrics,
    frozen_topk_candidates,
    oracle_gate,
    oracle_ranking,
)
from script.standard_ML.rm_region_retrieval_runtime import validate_disjoint_ids


def build_split(
    name: str,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    query_e: torch.Tensor,
    memory_x: torch.Tensor,
    memory_y: torch.Tensor,
    memory_e: torch.Tensor,
    *,
    topk: int,
    query_chunk: int,
    device: str,
) -> tuple[dict, dict]:
    indices, scores = frozen_topk_candidates(
        memory_x, memory_e, query_x, query_e,
        topk=topk, query_chunk=query_chunk, device=device,
    )
    labels = memory_y[indices]
    baseline = candidate_metrics(labels, query_y)
    oracle_labels = oracle_ranking(labels, query_y)
    oracle = candidate_metrics(oracle_labels, query_y)
    cache = {
        "protocol": "CMR_M3_frozen_mean_topk_candidates_v1",
        "role": name,
        "query_features": query_x.float(),
        "query_labels": query_y.long(),
        "query_events": query_e.long(),
        "candidate_indices": indices.long(),
        "candidate_scores": scores.float(),
        "candidate_labels": labels.long(),
        "coarse_k": int(topk),
        "same_event_excluded": True,
        "test_accessed": False,
    }
    result = {"baseline": baseline, "oracle": oracle}
    return cache, result


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M3 frozen Top-K oracle audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coarse_k", type=int, default=20)
    parser.add_argument("--query_chunk", type=int, default=1024)
    parser.add_argument("--minimum_damaged_headroom", type=float, default=0.05)
    parser.add_argument("--minimum_damaged_coverage", type=float, default=0.95)
    args = parser.parse_args()

    train_x, train_y, train_e, train_map, train = load_cmr_cache(
        args.train_cache, "mean", expected_role="train"
    )
    val_x, val_y, val_e, val_map, val = load_cmr_cache(
        args.val_cache, "mean", expected_role="query"
    )
    validate_disjoint_ids(train["image_ids"], val["image_ids"])
    if train_map != val_map:
        raise RuntimeError("Train/Validation event maps differ")
    if train["metadata"].get("p3_checkpoint_sha256") != val["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("Train/Validation P3 checkpoint hashes differ")
    memory_x, memory_y, memory_e, memory_indices = frozen_memory_view(
        train_x, train_y, train_e, train
    )

    train_candidate, train_result = build_split(
        "train", train_x, train_y, train_e, memory_x, memory_y, memory_e,
        topk=args.coarse_k, query_chunk=args.query_chunk, device=args.device,
    )
    val_candidate, val_result = build_split(
        "validation", val_x, val_y, val_e, memory_x, memory_y, memory_e,
        topk=args.coarse_k, query_chunk=args.query_chunk, device=args.device,
    )
    for cache in (train_candidate, val_candidate):
        cache["memory_features"] = memory_x.float()
        cache["memory_labels"] = memory_y.long()
        cache["memory_events"] = memory_e.long()
        cache["memory_indices_in_train"] = memory_indices.long()
    train_path = Path(args.train_candidates)
    val_path = Path(args.val_candidates)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_candidate, train_path)
    torch.save(val_candidate, val_path)

    checks = oracle_gate(
        train_result["baseline"], train_result["oracle"],
        damaged_headroom=args.minimum_damaged_headroom,
        minimum_damaged_coverage=args.minimum_damaged_coverage,
    )
    passed = all(checks.values())
    payload = {
        "protocol": "CMR_M3O_train_admitted_top20_oracle_v1",
        "coarse_key": "frozen_raw_mean",
        "coarse_k": int(args.coarse_k),
        "memory_items": int(memory_x.shape[0]),
        "train": train_result,
        "validation_diagnostic": val_result,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_TO_CMR_M3_PAIRWISE_TRAINING"
            if passed else "STOP_PAIRWISE_ROUTE_TOP20_HEADROOM_INSUFFICIENT"
        ),
        "train_candidates": str(train_path.resolve()),
        "val_candidates": str(val_path.resolve()),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print("CMR-M3O frozen Raw Mean Top-20 oracle audit")
    for split, result in (("Train", train_result), ("Validation diagnostic", val_result)):
        raw, oracle = result["baseline"], result["oracle"]
        print(
            f"{split:<22} | Raw D-R1={raw['per_class'][1]['R@1']:.4f} | "
            f"Oracle D-R1={oracle['per_class'][1]['R@1']:.4f} | "
            f"headroom={oracle['per_class'][1]['R@1']-raw['per_class'][1]['R@1']:+.4f}"
        )
    print(f"Checks          : {checks}")
    print(f"Decision        : {payload['decision']}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()

