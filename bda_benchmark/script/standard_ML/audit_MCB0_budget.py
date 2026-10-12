#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-CB0: fixed-budget P3 versus Raw-Mean-kNN complement audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from script.standard_ML.mcb0_budget_runtime import candidate_table, evaluate_budgets
from script.standard_ML.rm_region_retrieval_runtime import (
    class_balanced_knn,
    validate_disjoint_ids,
)


def load_p0(path: str, role: str) -> Mapping[str, object]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError(f"M-CB0 input is not a CMR-P0 cache: {path}")
    expected = "train" if role == "train" else "query"
    if metadata.get("role") != expected:
        raise RuntimeError(f"M-CB0 cache role mismatch: {metadata.get('role')} != {expected}")
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("M-CB0 refuses Test-derived caches")
    if not bool(metadata.get("diagnostic_gt_eligibility", False)):
        raise RuntimeError("M-CB0 requires declared diagnostic P0 eligibility")
    return cache


def parse_budgets(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("M-CB0 budgets must be unique positive integers")
    return result


@torch.no_grad()
def memory_scores(
    table: Mapping[str, object],
    memory_cache: Mapping[str, object],
    *,
    topk: int,
    beta: float,
    max_neighbors_per_event: int,
    chunk_size: int,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    if "memory_indices" not in memory_cache:
        raise KeyError("M-CB0 Train P0 cache has no class-event-balanced memory_indices")
    memory_indices = torch.as_tensor(memory_cache["memory_indices"]).long().cpu()
    memory_keys = torch.as_tensor(
        memory_cache["key_variants"]["mean"]
    ).float()[memory_indices].to(device)
    memory_labels = torch.as_tensor(memory_cache["labels"]).long()[memory_indices]
    memory_events = [
        str(memory_cache["events"][int(index)]) for index in memory_indices
    ]
    retrieval = class_balanced_knn(
        torch.as_tensor(table["keys"]).float().to(device),
        memory_keys,
        memory_labels,
        topk_per_class=topk,
        beta=beta,
        query_events=table["events"],
        memory_events=memory_events,
        exclude_same_event=True,
        max_neighbors_per_event=max_neighbors_per_event,
        chunk_size=chunk_size,
    )
    scores = retrieval["scores"].float().cpu()
    damaged_score = scores[:, 1]
    damaged_margin = damaged_score - torch.maximum(scores[:, 0], scores[:, 2])
    return {
        "damaged_score": damaged_score,
        "damaged_margin_diagnostic": damaged_margin,
        "best_damaged_similarity": retrieval["best_similarity_by_class"][:, 1],
        "damaged_neighbor_event_count": retrieval[
            "unique_neighbor_events_by_class"
        ][:, 1],
        "memory_items": int(memory_indices.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser("M-CB0 fixed review budget audit")
    parser.add_argument("--train_p0_cache", required=True)
    parser.add_argument("--validation_p0_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budgets", default="50,100,200,350")
    parser.add_argument("--primary_budget", type=int, default=200)
    parser.add_argument("--p3_fraction", type=float, default=0.80)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--minimum_combined_damaged_gain", type=int, default=1)
    parser.add_argument("--minimum_unique_damaged", type=int, default=10)
    parser.add_argument("--minimum_unique_damaged_events", type=int, default=5)
    parser.add_argument("--maximum_unique_damaged_event_share", type=float, default=0.40)
    args = parser.parse_args()
    budgets = parse_budgets(args.budgets)
    if args.primary_budget not in budgets:
        raise ValueError("M-CB0 primary budget must be one of the declared budgets")
    if abs(float(args.p3_fraction) - 0.80) > 1e-9:
        raise ValueError("M-CB0 v1 fixes p3_fraction=0.80; do not scan it")

    train_cache = load_p0(args.train_p0_cache, "train")
    val_cache = load_p0(args.validation_p0_cache, "validation")
    if train_cache["metadata"].get("p3_checkpoint_sha256") != val_cache[
        "metadata"
    ].get("p3_checkpoint_sha256"):
        raise RuntimeError("M-CB0 Train/Validation P3 checkpoint SHA differs")
    validate_disjoint_ids(train_cache["image_ids"], val_cache["image_ids"])
    train_table = candidate_table(train_cache)
    val_table = candidate_table(val_cache)
    device = torch.device(args.device)

    train_memory = memory_scores(
        train_table,
        train_cache,
        topk=args.topk,
        beta=args.beta,
        max_neighbors_per_event=args.max_neighbors_per_event,
        chunk_size=args.chunk_size,
        device=device,
    )
    val_memory = memory_scores(
        val_table,
        train_cache,
        topk=args.topk,
        beta=args.beta,
        max_neighbors_per_event=args.max_neighbors_per_event,
        chunk_size=args.chunk_size,
        device=device,
    )
    train_results = evaluate_budgets(
        train_table["p3_scores"],
        train_memory["damaged_score"],
        train_table["truth"],
        train_table["events"],
        budgets=budgets,
        p3_fraction=args.p3_fraction,
    )
    validation_results = evaluate_budgets(
        val_table["p3_scores"],
        val_memory["damaged_score"],
        val_table["truth"],
        val_table["events"],
        budgets=budgets,
        p3_fraction=args.p3_fraction,
    )

    def contract(result: Mapping[str, object]) -> tuple[dict[str, bool], dict[str, int]]:
        p3 = result["methods"]["p3_probability"]
        combined = result["methods"]["p3_80_memory_20"]
        complement = result["combined_vs_p3"]
        damaged_gain = int(combined["damaged_rescued"]) - int(p3["damaged_rescued"])
        affected_gain = int(combined["affected_rescued"]) - int(p3["affected_rescued"])
        checks = {
            "combined_damaged_gain_at_least_declared": (
                damaged_gain >= args.minimum_combined_damaged_gain
            ),
            "combined_affected_rescue_nondecreasing": affected_gain >= 0,
            "memory_contributes_at_least_declared_unique_damaged": (
                int(complement["unique_damaged_rescued"])
                >= args.minimum_unique_damaged
            ),
            "unique_damaged_cover_at_least_declared_events": (
                int(complement["unique_damaged_events"])
                >= args.minimum_unique_damaged_events
            ),
            "unique_damaged_event_share_at_most_declared": (
                float(complement["maximum_unique_damaged_event_share"])
                <= args.maximum_unique_damaged_event_share
            ),
        }
        return checks, {
            "combined_damaged_gain_vs_p3": damaged_gain,
            "combined_affected_gain_vs_p3": affected_gain,
        }

    train_checks, train_diagnostics = contract(
        train_results[str(args.primary_budget)]
    )
    validation_checks, validation_diagnostics = contract(
        validation_results[str(args.primary_budget)]
    )
    checks = {"train": train_checks, "validation": validation_checks}
    passed = all(train_checks.values()) and all(validation_checks.values())

    def unique_case_records(
        result: Mapping[str, object],
        table: Mapping[str, object],
        retrieval: Mapping[str, torch.Tensor],
    ) -> list[dict[str, object]]:
        local_indices = result["combined_vs_p3"]["unique_local_indices"]
        cache_rows = torch.as_tensor(table["indices"]).long().cpu()
        truth = torch.as_tensor(table["truth"]).long().cpu()
        p3_scores = torch.as_tensor(table["p3_scores"]).float().cpu()
        memory_rank = torch.as_tensor(retrieval["damaged_score"]).float().cpu()
        records = []
        for local in local_indices:
            index = int(local)
            records.append({
                "candidate_local_index": index,
                "p0_cache_row": int(cache_rows[index]),
                "image_id": str(table["image_ids"][index]),
                "event": str(table["events"][index]),
                "diagnostic_truth": int(truth[index]),
                "p3_affected_score": float(p3_scores[index]),
                "memory_damaged_score": float(memory_rank[index]),
            })
        return records

    train_primary = train_results[str(args.primary_budget)]
    validation_primary = validation_results[str(args.primary_budget)]
    payload = {
        "protocol": "M_CB0_fixed_budget_p3_memory_complement_v1",
        "setting": "human_review_candidate_generation_not_automatic_classification",
        "selection_contract": (
            "Budgets, primary budget, 80/20 allocation, Raw-Mean kNN configuration, "
            "and admission checks are fixed before Validation; no threshold is selected."
        ),
        "ranking_contract": {
            "p3_probability": "p(Damaged)+p(Destroyed) among frozen-P3 Intact regions",
            "memory_knn": "Raw Mean cross-event class-balanced kNN Damaged affinity",
            "combined": "top 80% P3 plus top 20% not already selected by P3",
        },
        "budgets": list(budgets),
        "primary_budget": int(args.primary_budget),
        "p3_fraction": float(args.p3_fraction),
        "retrieval": {
            "topk_per_class": int(args.topk),
            "beta": float(args.beta),
            "exclude_same_event": True,
            "max_neighbors_per_event": int(args.max_neighbors_per_event),
            "memory_items": int(val_memory["memory_items"]),
        },
        "train_candidate_count": int(train_table["candidate_count"]),
        "validation_candidate_count": int(val_table["candidate_count"]),
        "train_results": train_results,
        "validation_results": validation_results,
        "primary_budget_diagnostics": {
            "train": train_diagnostics,
            "validation": validation_diagnostics,
        },
        "primary_budget_unique_memory_cases": {
            "train": unique_case_records(train_primary, train_table, train_memory),
            "validation": unique_case_records(
                validation_primary, val_table, val_memory
            ),
        },
        "thresholds": {
            "minimum_combined_damaged_gain": args.minimum_combined_damaged_gain,
            "minimum_unique_damaged": args.minimum_unique_damaged,
            "minimum_unique_damaged_events": args.minimum_unique_damaged_events,
            "maximum_unique_damaged_event_share": args.maximum_unique_damaged_event_share,
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_MEMORY_AS_FIXED_BUDGET_COMPLEMENTARY_PROPOSER_NO_AUTOMATIC_FUSION"
            if passed else "STOP_CURRENT_MEMORY_COMPLEMENT_NO_UNIQUE_BUDGET_VALUE"
        ),
        "training": False,
        "fuses_p3": False,
        "human_review_required": True,
        "validation_selects_budget": False,
        "diagnostic_gt_eligibility": True,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 124)
    print("M-CB0 fixed-budget P3 versus Raw-Mean-kNN complement audit")
    print(f"Candidates       : Train={train_table['candidate_count']} | Validation={val_table['candidate_count']}")
    print(f"Memory items     : {val_memory['memory_items']}")
    print(f"Primary budget   : {args.primary_budget} | P3/Memory={args.p3_fraction:.0%}/{1-args.p3_fraction:.0%}")
    for budget in budgets:
        result = validation_results[str(budget)]
        print(f"Budget={budget}")
        for name, method in result["methods"].items():
            print(
                f"  {name:<20} | A={method['affected_rescued']:>3} | "
                f"D={method['damaged_rescued']:>3} | X={method['destroyed_rescued']:>3} | "
                f"precision={100*method['precision_at_budget']:>6.2f}% | "
                f"events={method['rescue_events']:>2}"
            )
        comp = result["combined_vs_p3"]
        print(
            "  combined unique vs P3 | "
            f"D={comp['unique_damaged_rescued']} | "
            f"X={comp['unique_destroyed_rescued']} | "
            f"I={comp['unique_intact_reviewed']} | "
            f"D-events={comp['unique_damaged_events']} | "
            f"max-event-share={comp['maximum_unique_damaged_event_share']:.3f}"
        )
    print(f"Train checks     : {train_checks}")
    print(f"Validation checks: {validation_checks}")
    print(f"Decision         : {payload['decision']}")
    print(f"Saved            : {output}")
    print("Training         : False")
    print("Automatic fusion : False")
    print("Test accessed    : False")


if __name__ == "__main__":
    main()
