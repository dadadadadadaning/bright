#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RM7: train-only locked dual-key retrieval and reject-option audit.

The stage never trains or changes P3.  A P3-Intact region is changed to Damaged
only when independent semantic and scene-relative SAR memories agree, the query
lies near a P3 decision boundary, the support is in-distribution, and neighbours
come from multiple events.  All thresholds are chosen using train-memory LOEO
retrieval with natural-distribution stratum weights; validation is read once.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from script.standard_ML.audit_RM_region_retrieval import load_cache
from script.standard_ML.rm_region_retrieval_runtime import (
    apply_dual_key_reject_rule,
    binary_auroc,
    build_raw_only_keys,
    class_balanced_knn,
    class_event_stratum_weights,
    complement_audit,
    dual_key_reject_signals,
    metrics_from_predictions,
    validate_disjoint_ids,
)


def retrieve_dual(
    memory: Mapping[str, object],
    query: Mapping[str, object],
    *,
    device: torch.device,
    topk: int,
    beta: float,
    max_neighbors_per_event: int,
) -> Dict[str, object]:
    normalizer = memory["metadata"]["relative_raw_stat_normalizer"]
    memory_visual = F.normalize(memory["keys"].float(), dim=1)
    query_visual = F.normalize(query["keys"].float(), dim=1)
    memory_sar = build_raw_only_keys(memory["relative_raw_stats"], normalizer)
    query_sar = build_raw_only_keys(query["relative_raw_stats"], normalizer)
    kwargs = dict(
        memory_labels=memory["labels"],
        topk_per_class=int(topk),
        beta=float(beta),
        query_events=query["events"],
        memory_events=memory["events"],
        exclude_same_event=True,
        max_neighbors_per_event=int(max_neighbors_per_event),
    )
    visual = class_balanced_knn(
        query_visual.to(device), memory_visual.to(device), **kwargs
    )
    sar = class_balanced_knn(query_sar.to(device), memory_sar.to(device), **kwargs)
    signals = dual_key_reject_signals(query["base_probabilities"], visual, sar)
    return {"visual": visual, "sar": sar, "signals": signals}


def _weighted_rule_audit(
    truth: torch.Tensor,
    base_prediction: torch.Tensor,
    accepted: torch.Tensor,
    weights: torch.Tensor,
    events: Sequence[str],
) -> Dict[str, object]:
    truth = truth.long().cpu()
    base = base_prediction.long().cpu()
    accepted = accepted.bool().cpu()
    weights = weights.float().cpu()
    auditable = accepted & ((truth == 0) | (truth == 1))
    rescued_mask = auditable & (truth == 1) & (base == 0)
    harmed_mask = auditable & (truth == 0) & (base == 0)
    rescued = float(weights[rescued_mask].sum())
    harmed = float(weights[harmed_mask].sum())
    all_events = sorted(set(map(str, events)))
    event_net = {event: 0.0 for event in all_events}
    for index in torch.nonzero(auditable, as_tuple=False).flatten().tolist():
        event_net[str(events[index])] += (
            float(weights[index]) if truth[index] == 1 else -float(weights[index])
        )
    selected = int(auditable.sum())
    return {
        "selected": selected,
        "rescued_weight": rescued,
        "harmed_weight": harmed,
        "net_weight": rescued - harmed,
        "precision": rescued / max(rescued + harmed, 1e-12),
        "rescued_regions": int(rescued_mask.sum()),
        "harmed_regions": int(harmed_mask.sum()),
        "nonnegative_events": sum(value >= 0.0 for value in event_net.values()),
        "event_net": event_net,
    }


def _quantiles(values: torch.Tensor, levels: Sequence[float]) -> Sequence[float]:
    vector = values.detach().float().cpu()
    return sorted(set(float(torch.quantile(vector, float(level))) for level in levels))


def select_rule(
    memory: Mapping[str, object],
    signals: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
) -> Dict[str, object]:
    base = memory["base_predictions"].long().cpu()
    truth = memory["labels"].long().cpu()
    proposal = (base == 0) & ((truth == 0) | (truth == 1))
    if int(proposal.sum()) < 100:
        raise RuntimeError("RM7 requires at least 100 train LOEO proposals")
    levels = (0.50, 0.70, 0.85, 0.95)
    v_values = _quantiles(signals["visual_damaged_vs_intact_llr"][proposal], levels)
    s_values = _quantiles(signals["sar_damaged_vs_intact_llr"][proposal], levels)
    u_values = _quantiles(signals["p3_uncertainty"][proposal], (0.50, 0.75, 0.90))
    support_values = _quantiles(signals["damaged_support"][proposal], (0.25, 0.50, 0.75))
    candidates = []
    for visual_threshold in v_values:
        for sar_threshold in s_values:
            for uncertainty_threshold in u_values:
                for support_threshold in support_values:
                    for minimum_events in (2, 3, 4):
                        rule = {
                            "enabled": True,
                            "visual_llr_threshold": visual_threshold,
                            "sar_llr_threshold": sar_threshold,
                            "p3_uncertainty_threshold": uncertainty_threshold,
                            "damaged_support_threshold": support_threshold,
                            "minimum_damaged_events": float(minimum_events),
                        }
                        accepted = apply_dual_key_reject_rule(
                            base, signals, rule
                        )["accepted"]
                        audit = _weighted_rule_audit(
                            truth, base, accepted, weights, memory["events"]
                        )
                        utility = audit["rescued_weight"] - 2.0 * audit["harmed_weight"]
                        candidates.append({"rule": rule, "audit": audit, "utility": utility})
    admissible = [
        item for item in candidates
        if item["audit"]["selected"] >= 10
        and item["audit"]["rescued_regions"] >= 5
        and item["audit"]["precision"] >= 0.67
        and item["audit"]["net_weight"] > 0.0
        and item["audit"]["nonnegative_events"] >= 10
    ]
    if not admissible:
        return {
            "rule": {"enabled": False},
            "audit": _weighted_rule_audit(
                truth,
                base,
                torch.zeros_like(base, dtype=torch.bool),
                weights,
                memory["events"],
            ),
            "utility": 0.0,
            "admissible_rule_found": False,
            "candidate_count": len(candidates),
        }
    winner = max(
        admissible,
        key=lambda item: (
            item["utility"],
            item["audit"]["precision"],
            item["audit"]["rescued_weight"],
        ),
    )
    return {**winner, "admissible_rule_found": True, "candidate_count": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser("RM7 dual-key reject-option audit")
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    args = parser.parse_args()

    memory = load_cache(args.memory, "memory")
    query = load_cache(args.queries, "query")
    validate_disjoint_ids(memory["image_ids"], query["image_ids"])
    device = torch.device(args.device)
    train = retrieve_dual(
        memory,
        memory,
        device=device,
        topk=args.topk,
        beta=args.beta,
        max_neighbors_per_event=args.max_neighbors_per_event,
    )
    metadata = memory["metadata"]
    train_weights = class_event_stratum_weights(
        memory["labels"],
        memory["events"],
        metadata["class_event_counts_before_balance"],
        metadata["class_event_counts"],
    )
    selected = select_rule(memory, train["signals"], train_weights)

    validation = retrieve_dual(
        memory,
        query,
        device=device,
        topk=args.topk,
        beta=args.beta,
        max_neighbors_per_event=args.max_neighbors_per_event,
    )
    applied = apply_dual_key_reject_rule(
        query["base_predictions"], validation["signals"], selected["rule"]
    )
    base_metrics = metrics_from_predictions(
        query["labels"], query["base_predictions"], query["events"]
    )
    candidate_metrics = metrics_from_predictions(
        query["labels"], applied["prediction"], query["events"]
    )
    complement = complement_audit(
        query["labels"], query["base_predictions"], applied["prediction"]
    )
    auditable = (
        (query["base_predictions"].long().cpu() == 0)
        & ((query["labels"].long().cpu() == 0) | (query["labels"].long().cpu() == 1))
    )
    target = query["labels"].long().cpu()[auditable] == 1
    agreement_auc = binary_auroc(
        validation["signals"]["agreement_score"][auditable], target
    )
    event_delta = {
        event: 100.0 * (
            candidate_metrics["events"][event]["macro_f1"]
            - base_metrics["events"][event]["macro_f1"]
        )
        for event in base_metrics["events"]
    }
    accepted = applied["accepted"]
    accepted_auditable = accepted & (
        (query["labels"].long().cpu() == 0) | (query["labels"].long().cpu() == 1)
    )
    rescued = int((accepted_auditable & (query["labels"].long().cpu() == 1)).sum())
    harmed = int((accepted_auditable & (query["labels"].long().cpu() == 0)).sum())
    precision = rescued / max(rescued + harmed, 1)
    checks = {
        "train_only_admissible_rule_found": bool(selected["admissible_rule_found"]),
        "accepted_at_least_10_validation_regions": int(accepted.sum()) >= 10,
        "validation_precision_at_least_0_60": precision >= 0.60,
        "validation_overall_net_positive": complement["net_complement"] > 0,
        "validation_damaged_net_positive": complement["per_class"]["damaged"]["net_complement"] > 0,
        "validation_macro_f1_nondecreasing": candidate_metrics["macro_f1"] >= base_metrics["macro_f1"],
        "ten_events_nondecreasing": sum(value >= 0.0 for value in event_delta.values()) >= 10,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "RM7_dual_key_train_LOEO_reject_v1",
        "key_contract": {
            "semantic": "frozen P3 region feature cosine key",
            "sar": "separate scene-relative SAR mean/std cosine key",
            "fusion": "agreement by rejection; keys are never concatenated",
        },
        "reject_contract": "P3 Intact AND visual D/I LLR AND SAR D/I LLR AND P3 uncertainty AND support AND multi-event consensus",
        "selection_split": "train memory LOEO with natural-distribution stratum weights",
        "selected_train_rule": selected,
        "validation": {
            "accepted": int(accepted.sum()),
            "rescued": rescued,
            "harmed": harmed,
            "precision": precision,
            "agreement_auroc": agreement_auc,
            "base_metrics": base_metrics,
            "candidate_metrics": candidate_metrics,
            "complement": complement,
            "event_macro_f1_delta_pp_vs_p3": event_delta,
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_RM7_TO_MULTI_SEED_VALIDATION_DO_NOT_TEST"
            if passed
            else "STOP_RM7_REJECTOR_DO_NOT_FUSE"
        ),
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 96)
    print("RM stage       : rm7_dual_key_reject")
    print(f"Train rule     : {selected['rule']}")
    print(f"Accepted       : {int(accepted.sum())}")
    print(f"Rescue/Harm    : {rescued}/{harmed}")
    print(f"Precision      : {100*precision:.2f}%")
    print(f"Agreement AUC  : {agreement_auc:.4f}")
    print(f"Macro P3/RM7  : {100*base_metrics['macro_f1']:.4f}/{100*candidate_metrics['macro_f1']:.4f}")
    print(f"Checks         : {checks}")
    print(f"Decision       : {payload['decision']}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
