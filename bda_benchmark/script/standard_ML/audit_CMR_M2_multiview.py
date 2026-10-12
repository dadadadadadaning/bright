# -*- coding: utf-8 -*-
"""CMR-M2: train-locked Mean-to-Critical coarse-to-fine retrieval audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from script.standard_ML.cmr_metric_utils import (
    frozen_memory_view,
    load_cmr_cache,
    relation_statistics,
)
from script.standard_ML.cmr_multiview_utils import multiview_rerank_sweep
from script.standard_ML.rm_region_retrieval_runtime import validate_disjoint_ids


ALLOWED_RERANK_VIEWS = ("hard_critical", "exist_embedding")


def admission_checks(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    ap_k: int,
    minimum_damaged_gain: float,
) -> Dict[str, bool]:
    return {
        "damaged_r1_plus_declared_gain": (
            float(candidate["per_class"][1]["R@1"])
            >= float(baseline["per_class"][1]["R@1"])
            + float(minimum_damaged_gain)
        ),
        "damaged_r5_nondecreasing": (
            float(candidate["per_class"][1]["R@5"])
            >= float(baseline["per_class"][1]["R@5"])
        ),
        "destroyed_r1_drop_at_most_0_02": (
            float(candidate["per_class"][2]["R@1"])
            >= float(baseline["per_class"][2]["R@1"]) - 0.02
        ),
        "intact_r1_nondecreasing": (
            float(candidate["per_class"][0]["R@1"])
            >= float(baseline["per_class"][0]["R@1"])
        ),
        "macro_map_improves": (
            float(candidate[f"macro_mAP@{int(ap_k)}"])
            > float(baseline[f"macro_mAP@{int(ap_k)}"])
        ),
        # M2 does not train or transform either feature space; it only reranks
        # a frozen candidate pool.  Mean DeltaCE is therefore exactly retained.
        "mean_delta_ce_preserved": True,
    }


def selection_key(result: Mapping[str, object], checks: Mapping[str, bool], ap_k: int) -> tuple:
    return (
        all(checks.values()),
        sum(bool(value) for value in checks.values()),
        float(result["per_class"][1]["R@1"]),
        float(result["per_class"][1]["R@5"]),
        float(result[f"macro_mAP@{int(ap_k)}"]),
        float(result["per_class"][0]["R@1"]),
        float(result["per_class"][2]["R@1"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M2 frozen multi-view reranking audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", default=",".join(ALLOWED_RERANK_VIEWS))
    parser.add_argument("--lambdas", default="0.10,0.25,0.50,1.0")
    parser.add_argument("--coarse_k", type=int, default=20)
    parser.add_argument("--query_chunk", type=int, default=512)
    parser.add_argument("--ap_k", type=int, default=10)
    parser.add_argument("--minimum_damaged_gain", type=float, default=0.01)
    parser.add_argument("--relation_pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    if not views or any(item not in ALLOWED_RERANK_VIEWS for item in views):
        raise ValueError(f"views must be drawn from {ALLOWED_RERANK_VIEWS}")
    lambdas = tuple(float(item) for item in args.lambdas.split(",") if item.strip())
    if not lambdas or min(lambdas) <= 0:
        raise ValueError("reranking lambdas must be positive")

    train_mean, train_y, train_e, train_event_map, train = load_cmr_cache(
        args.train_cache, "mean", expected_role="train"
    )
    val_mean, val_y, val_e, val_event_map, val = load_cmr_cache(
        args.val_cache, "mean", expected_role="query"
    )
    validate_disjoint_ids(train["image_ids"], val["image_ids"])
    if train_event_map != val_event_map:
        raise RuntimeError("Train/Validation event maps differ")
    if train["metadata"].get("p3_checkpoint_sha256") != val["metadata"].get(
        "p3_checkpoint_sha256"
    ):
        raise RuntimeError("Train/Validation P3 checkpoint hashes differ")
    memory_mean, memory_y, memory_e, memory_indices = frozen_memory_view(
        train_mean, train_y, train_e, train
    )
    memory_views = {
        name: train["key_variants"][name][memory_indices].float()
        for name in views
    }
    train_views = {
        name: train["key_variants"][name].float() for name in views
    }
    val_views = {name: val["key_variants"][name].float() for name in views}

    print("=" * 96)
    print("CMR-M2 Train-LOEO parameter selection")
    print(f"Train regions : {train_mean.shape[0]}")
    print(f"Memory items  : {memory_indices.numel()}")
    print(f"Views         : {views}")
    print(f"Lambdas       : {lambdas}")
    print(f"Coarse K      : {args.coarse_k}")
    train_sweep = multiview_rerank_sweep(
        memory_mean,
        memory_views,
        memory_y,
        memory_e,
        train_mean,
        train_views,
        train_y,
        train_e,
        lambdas=lambdas,
        coarse_k=args.coarse_k,
        ap_k=args.ap_k,
        query_chunk=args.query_chunk,
        device=args.device,
    )
    train_baseline = train_sweep["results"]["mean_only"]
    train_checks = {}
    for name, result in train_sweep["results"].items():
        train_checks[name] = admission_checks(
            result,
            train_baseline,
            ap_k=args.ap_k,
            minimum_damaged_gain=args.minimum_damaged_gain,
        )
    candidates = [name for name in train_sweep["results"] if name != "mean_only"]
    selected = max(
        candidates,
        key=lambda name: selection_key(
            train_sweep["results"][name], train_checks[name], args.ap_k
        ),
    )
    selected_config = train_sweep["configs"][selected]

    # Validation receives only the train-locked candidate and Mean baseline.
    selected_view = str(selected_config["view"])
    selected_lambda = float(selected_config["lambda"])
    validation_sweep = multiview_rerank_sweep(
        memory_mean,
        {selected_view: memory_views[selected_view]},
        memory_y,
        memory_e,
        val_mean,
        {selected_view: val_views[selected_view]},
        val_y,
        val_e,
        lambdas=(selected_lambda,),
        coarse_k=args.coarse_k,
        ap_k=args.ap_k,
        query_chunk=args.query_chunk,
        device=args.device,
    )
    validation_baseline = validation_sweep["results"]["mean_only"]
    validation_name = f"mean+{selected_view}@{selected_lambda:g}"
    validation_candidate = validation_sweep["results"][validation_name]
    validation_checks = admission_checks(
        validation_candidate,
        validation_baseline,
        ap_k=args.ap_k,
        minimum_damaged_gain=args.minimum_damaged_gain,
    )
    passed = all(validation_checks.values())
    mean_relation = relation_statistics(
        train_mean,
        train_y,
        train_e,
        num_pairs=args.relation_pairs,
        seed=args.seed,
    )
    payload = {
        "protocol": "CMR_M2_train_locked_multiview_rerank_v1",
        "train_cache": str(Path(args.train_cache).resolve()),
        "val_cache": str(Path(args.val_cache).resolve()),
        "memory_items": int(memory_indices.numel()),
        "coarse_key": "mean",
        "coarse_k": int(args.coarse_k),
        "selection_split": "natural Train Query with frozen P0 Memory and same-event exclusion",
        "selection_candidates": train_sweep["configs"],
        "selected_config": selected_config,
        "selected_name": selected,
        "train_results": train_sweep["results"],
        "train_checks": train_checks,
        "validation_baseline": validation_baseline,
        "validation_candidate": validation_candidate,
        "validation_checks": validation_checks,
        "mean_relation_statistics": mean_relation,
        "passed": passed,
        "decision": (
            "ADMIT_CMR_M2_TO_REJECTOR_AUDIT_DO_NOT_FUSE_OR_TEST"
            if passed
            else "STOP_MULTIVIEW_RERANK_THEN_TRY_SELECTIVE_DAMAGED_SUPCON"
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print("CMR-M2 train-locked multiview validation")
    print(f"Selected on Train : {selected}")
    for label, result in (
        ("Mean baseline", validation_baseline),
        ("Locked rerank", validation_candidate),
    ):
        print(
            f"{label:<14} | macro-mAP={result[f'macro_mAP@{args.ap_k}']:.4f} | "
            f"I-R1={result['per_class'][0]['R@1']:.4f} | "
            f"D-R1={result['per_class'][1]['R@1']:.4f} | "
            f"D-R5={result['per_class'][1]['R@5']:.4f} | "
            f"X-R1={result['per_class'][2]['R@1']:.4f}"
        )
    print(f"Checks            : {validation_checks}")
    print(f"Decision          : {payload['decision']}")
    print(f"Saved             : {output}")
    print("Test accessed     : False")


if __name__ == "__main__":
    main()

