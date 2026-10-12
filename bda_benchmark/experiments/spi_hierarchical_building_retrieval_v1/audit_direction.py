from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (
    binary_auroc, direction_name, flat_retrieve, hierarchical_retrieve,
    prepare_cards,
)


def direction_metrics(data, result, mask: torch.Tensor) -> dict:
    target = data.utility_label[mask]
    score = result["probability"][mask]
    valid = result["valid"][mask].any(1)
    event_rows = [name for name, keep in zip(data.event_names, mask.tolist()) if keep]
    event_auc = {}
    event_counts = {}
    for event in sorted(set(event_rows)):
        event_mask = torch.tensor([value == event for value in event_rows])
        event_target = target[event_mask]
        positive = int((event_target == 1).sum())
        negative = int((event_target == 0).sum())
        event_counts[event] = {"positive": positive, "negative": negative}
        value = binary_auroc(event_target, score[event_mask])
        if value == value:
            event_auc[event] = value
    return {
        "clusters": int(mask.sum()),
        "positive": int((target == 1).sum()),
        "negative": int((target == 0).sum()),
        "utility_auroc": binary_auroc(target, score),
        "event_macro_auroc": (
            sum(event_auc.values()) / len(event_auc) if event_auc else float("nan")
        ),
        "correct_events": sum(value > .5 for value in event_auc.values()),
        "evaluable_events": len(event_auc),
        "coverage": float(valid.float().mean()) if len(valid) else 0.0,
        "neighbour_purity": (
            float(result["neighbour_purity"][mask][valid].mean())
            if bool(valid.any()) else 0.0
        ),
        "mean_neighbour_events": (
            float(result["neighbour_event_count"][mask][valid].float().mean())
            if bool(valid.any()) else 0.0
        ),
        "events": event_auc,
        "event_counts": event_counts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Direction detail audit for hierarchical retrieval")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--r0-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coarse-buildings", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--minimum-positive", type=int, default=100)
    parser.add_argument("--minimum-negative", type=int, default=100)
    parser.add_argument("--minimum-evaluable-events", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    r0 = json.loads(Path(args.r0_json).read_text(encoding="utf-8"))
    if r0.get("decision") != "STOP_HBR_R0_NO_CROSS_EVENT_UTILITY_NEIGHBOURS":
        raise RuntimeError("direction audit is intended for the stopped R0 result")
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    if train_cache.get("split") != "meta_train" or calibration_cache.get("split") != "meta_calibration":
        raise RuntimeError("direction audit requires meta_train/meta_calibration")
    train, statistics = prepare_cards(train_cache)
    calibration, _ = prepare_cards(calibration_cache, statistics)

    retrievals = {
        "b0_flat_cluster": flat_retrieve(
            calibration, train, topk=args.topk,
            exact_direction=False, maximum_per_event=args.topk,
        ),
        "b1_building_route": hierarchical_retrieve(
            calibration, train, coarse_buildings=args.coarse_buildings,
            topk=args.topk, exact_direction=False,
            maximum_per_event=args.topk,
        ),
        "b2_direction_route": hierarchical_retrieve(
            calibration, train, coarse_buildings=args.coarse_buildings,
            topk=args.topk, exact_direction=True,
            maximum_per_event=args.topk,
        ),
        "b3_direction_event_diverse": hierarchical_retrieve(
            calibration, train, coarse_buildings=args.coarse_buildings,
            topk=args.topk, exact_direction=True,
            maximum_per_event=args.maximum_per_event,
        ),
    }
    directions = {}
    admitted = []
    for code in sorted(int(value) for value in torch.unique(calibration.direction).tolist()):
        mask = calibration.direction == code
        name = direction_name(code)
        rows = {variant: direction_metrics(calibration, result, mask)
                for variant, result in retrievals.items()}
        best_name = max(rows, key=lambda key: (
            rows[key]["event_macro_auroc"] if rows[key]["event_macro_auroc"] == rows[key]["event_macro_auroc"] else -1,
            rows[key]["utility_auroc"] if rows[key]["utility_auroc"] == rows[key]["utility_auroc"] else -1,
        ))
        best = rows[best_name]
        checks = {
            "positive_ge_minimum": best["positive"] >= args.minimum_positive,
            "negative_ge_minimum": best["negative"] >= args.minimum_negative,
            "evaluable_events_ge_minimum": best["evaluable_events"] >= args.minimum_evaluable_events,
            "utility_auroc_ge_0p65": best["utility_auroc"] >= .65,
            "event_macro_auroc_ge_0p60": best["event_macro_auroc"] >= .60,
            "neighbour_purity_ge_0p60": best["neighbour_purity"] >= .60,
        }
        if all(checks.values()):
            admitted.append(name)
        directions[name] = {
            "code": code, "variants": rows, "selected_variant": best_name,
            "checks": checks, "admitted": all(checks.values()),
        }
        print(json.dumps({
            "direction": name, "selected": best_name,
            "auc": best["utility_auroc"], "event": best["event_macro_auroc"],
            "positive": best["positive"], "negative": best["negative"],
            "purity": best["neighbour_purity"], "admitted": all(checks.values()),
        }, ensure_ascii=False), flush=True)

    decision = (
        "BUILD_DIRECTION_SPECIFIC_COMPLETE_BUILDING_CACHE"
        if admitted else "BUILD_ALL_DIRECTION_COMPLETE_BUILDING_CACHE_NO_D1_SHORTCUT"
    )
    report = {
        "protocol": f"{PROTOCOL}_DIRECTION_DETAIL",
        "configuration": vars(args), "directions": directions,
        "admitted_directions": admitted, "decision": decision,
        "interpretation": (
            "Admitted directions may proceed to a complete-building token cache; "
            "this result does not admit inference-time Memory."
        ),
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"admitted_directions": admitted, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
