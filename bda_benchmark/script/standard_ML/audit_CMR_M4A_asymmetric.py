# -*- coding: utf-8 -*-
"""CMR-M4A: no-training, Train-locked asymmetric Damaged rescue audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from model.cmr.pairwise import CMRCandidateConditionalPairwiseScorer
from script.standard_ML.cmr_asymmetric_rescue_utils import (
    metric_bundle,
    one_way_accept_mask,
    promote_selected_to_front,
    rescue_complement,
    select_locked_threshold,
    threshold_checks,
)
from script.standard_ML.train_CMR_M3_pairwise import load_candidates


@torch.no_grad()
def score_split(
    model: CMRCandidateConditionalPairwiseScorer,
    payload: Mapping[str, object],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, torch.Tensor]:
    query = payload["query_features"].float()
    memory = payload["memory_features"].float()
    indices = payload["candidate_indices"].long()
    base = payload["candidate_scores"].float()
    labels = payload["candidate_labels"].long()
    model_orders, margins, best_d_positions = [], [], []
    model.eval()
    for start in range(0, query.shape[0], int(batch_size)):
        end = min(start + int(batch_size), query.shape[0])
        local_labels = labels[start:end].to(device)
        q = query[start:end].to(device)[:, None, :].expand(-1, indices.shape[1], -1)
        c = memory[indices[start:end]].to(device)
        score = base[start:end].to(device) + model(q, c, local_labels)
        order = torch.argsort(score, dim=1, descending=True, stable=True)
        model_orders.append(order.cpu())
        damaged = local_labels == 1
        nondamaged = ~damaged
        best_d_score, best_d_position = score.masked_fill(~damaged, float("-inf")).max(dim=1)
        best_non_d_score = score.masked_fill(~nondamaged, float("-inf")).max(dim=1).values
        # Rows without D (or without Non-D) are valid. Their margin becomes
        # +/-inf, while the one-way proposal contract makes acceptance
        # impossible whenever the required source/target class is absent.
        margins.append((best_d_score - best_non_d_score).cpu())
        best_d_positions.append(best_d_position.cpu())
    order = torch.cat(model_orders)
    model_ranked = torch.gather(labels, 1, order)
    return {
        "raw_ranked": labels,
        "model_ranked": model_ranked,
        "proposal_margin": torch.cat(margins),
        "best_d_position": torch.cat(best_d_positions),
    }


def evaluate_threshold(
    scored: Mapping[str, torch.Tensor],
    payload: Mapping[str, object],
    threshold: float,
) -> Dict[str, object]:
    accept = one_way_accept_mask(
        scored["raw_ranked"],
        scored["model_ranked"],
        scored["proposal_margin"],
        threshold=float(threshold),
    )
    final_ranked = promote_selected_to_front(
        scored["raw_ranked"], scored["best_d_position"], accept
    )
    complement = rescue_complement(
        scored["raw_ranked"],
        final_ranked,
        payload["query_labels"],
        payload["query_events"],
        accept,
    )
    return {
        "threshold": float(threshold),
        "complement": complement,
        "metrics": metric_bundle(final_ranked, payload["query_labels"]),
        "final_ranked": final_ranked,
    }


def validation_checks(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    complement: Mapping[str, object],
    *,
    maximum_destroyed_drop: float,
    minimum_positive_events: int,
) -> Dict[str, bool]:
    return {
        "damaged_r1_improves": candidate["per_class"][1]["R@1"] > baseline["per_class"][1]["R@1"],
        "damaged_r5_nondecreasing": candidate["per_class"][1]["R@5"] >= baseline["per_class"][1]["R@5"],
        "intact_r1_nondecreasing": candidate["per_class"][0]["R@1"] >= baseline["per_class"][0]["R@1"],
        "destroyed_r1_drop_within_limit": candidate["per_class"][2]["R@1"] >= baseline["per_class"][2]["R@1"] - float(maximum_destroyed_drop),
        "overall_net_correction_positive": int(complement["net_correction"]) > 0,
        "positive_events_at_least_declared": int(complement["positive_events"]) >= int(minimum_positive_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M4A asymmetric rescue audit")
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--thresholds", default="0,0.0025,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--minimum_rescued", type=int, default=10)
    parser.add_argument("--minimum_precision", type=float, default=0.55)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    parser.add_argument("--minimum_positive_events", type=int, default=3)
    parser.add_argument("--maximum_destroyed_drop", type=float, default=0.01)
    args = parser.parse_args()

    thresholds = tuple(sorted(set(float(value) for value in args.thresholds.split(",") if value.strip())))
    if not thresholds or min(thresholds) < 0:
        raise ValueError("thresholds must be a non-empty nonnegative list")
    train = load_candidates(args.train_candidates, "train")
    val = load_candidates(args.val_candidates, "validation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "CMR_M3T_candidate_conditional_top1_v1":
        raise RuntimeError("CMR-M4A requires the frozen M3T checkpoint")
    if bool(checkpoint.get("test_accessed", True)):
        raise RuntimeError("M3T checkpoint Test contract violation")
    config = checkpoint["config"]
    model = CMRCandidateConditionalPairwiseScorer(
        train["query_features"].shape[1],
        hidden_dim=config["hidden_dim"],
        maximum_residual=config["maximum_residual"],
        num_candidate_classes=3,
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    train_scored = score_split(model, train, device=device)
    validation_scored = score_split(model, val, device=device)
    train_candidates = []
    for threshold in thresholds:
        result = evaluate_threshold(train_scored, train, threshold)
        checks = threshold_checks(
            result["complement"],
            minimum_rescued=args.minimum_rescued,
            minimum_precision=args.minimum_precision,
            minimum_nonnegative_events=args.minimum_nonnegative_events,
            minimum_positive_events=args.minimum_positive_events,
        )
        train_candidates.append({
            "threshold": float(threshold),
            "complement": result["complement"],
            "metrics": result["metrics"],
            "checks": checks,
            "passed": all(checks.values()),
        })
    selected = select_locked_threshold(train_candidates)
    validation = evaluate_threshold(validation_scored, val, selected["threshold"])
    raw_validation = metric_bundle(validation_scored["raw_ranked"], val["query_labels"])
    val_checks = validation_checks(
        raw_validation,
        validation["metrics"],
        validation["complement"],
        maximum_destroyed_drop=args.maximum_destroyed_drop,
        minimum_positive_events=args.minimum_positive_events,
    )
    passed = bool(selected["passed"]) and all(val_checks.values())
    payload = {
        "protocol": "CMR_M4A_train_locked_one_way_damaged_rescue_v1",
        "policy": "Raw D frozen; only Raw Non-D -> M3T D may be accepted",
        "threshold_candidates": list(thresholds),
        "selected_threshold": float(selected["threshold"]),
        "train_candidates": train_candidates,
        "selected_train": selected,
        "validation": {
            "raw_metrics": raw_validation,
            "candidate_metrics": validation["metrics"],
            "complement": validation["complement"],
            "checks": val_checks,
        },
        "passed": passed,
        "decision": (
            "ADMIT_CMR_M4A_TO_NO_GT_QUERY_PROTOCOL_DO_NOT_TEST_YET"
            if passed else "STOP_ASYMMETRIC_MEMORY_RESCUE_EVIDENCE_INSUFFICIENT"
        ),
        "trains_model": False,
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 112)
    print("CMR-M4A Train-locked asymmetric Damaged rescue audit")
    print("Train threshold candidates:")
    for item in train_candidates:
        complement = item["complement"]
        print(
            f"  tau={item['threshold']:<7g} | accepted={complement['accepted']:>5} | "
            f"D-rescue={complement['damaged_rescued']:>4} | "
            f"harm={complement['harmed']:>4} | net={complement['net_correction']:+5} | "
            f"precision={100*complement['accepted_precision']:>6.2f}% | "
            f"events+={complement['positive_events']:>2} | passed={item['passed']}"
        )
    print(f"Selected threshold : {selected['threshold']}")
    print(f"Train passed       : {selected['passed']}")
    print(f"Train complement   : {selected['complement']}")
    raw, candidate = raw_validation, validation["metrics"]
    print(
        f"Validation Raw      | macro={raw['macro_mAP@10']:.4f} | "
        f"I-R1={raw['per_class'][0]['R@1']:.4f} | D-R1={raw['per_class'][1]['R@1']:.4f} | "
        f"D-R5={raw['per_class'][1]['R@5']:.4f} | X-R1={raw['per_class'][2]['R@1']:.4f}"
    )
    print(
        f"Validation Rescue   | macro={candidate['macro_mAP@10']:.4f} | "
        f"I-R1={candidate['per_class'][0]['R@1']:.4f} | D-R1={candidate['per_class'][1]['R@1']:.4f} | "
        f"D-R5={candidate['per_class'][1]['R@5']:.4f} | X-R1={candidate['per_class'][2]['R@1']:.4f}"
    )
    print(f"Validation complement: {validation['complement']}")
    print(f"Checks               : {val_checks}")
    print(f"Decision             : {payload['decision']}")
    print(f"Saved                : {output}")
    print("Test accessed        : False")


if __name__ == "__main__":
    main()
