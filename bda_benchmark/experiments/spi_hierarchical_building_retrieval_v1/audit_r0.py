from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import binary_auroc, flat_retrieve, hierarchical_retrieve, prepare_cards


def metrics(data, result) -> dict:
    auc = binary_auroc(data.utility_label, result["probability"])
    event_auc = {}
    for name in sorted(set(data.event_names)):
        mask = torch.tensor([value == name for value in data.event_names])
        value = binary_auroc(data.utility_label[mask], result["probability"][mask])
        if value == value:
            event_auc[name] = value
    valid = result["valid"].any(1)
    return {
        "utility_auroc": auc,
        "event_macro_auroc": sum(event_auc.values()) / max(len(event_auc), 1),
        "correct_events": sum(value > .5 for value in event_auc.values()),
        "evaluable_events": len(event_auc),
        "query_coverage": float(valid.float().mean()),
        "mean_neighbour_purity": float(result["neighbour_purity"][valid].mean()) if valid.any() else 0.0,
        "mean_neighbour_events": float(result["neighbour_event_count"][valid].float().mean()) if valid.any() else 0.0,
        "events": event_auc,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Hierarchical building-to-cluster retrieval R0")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coarse-buildings", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    if train_cache.get("split") != "meta_train" or calibration_cache.get("split") != "meta_calibration":
        raise RuntimeError("R0 requires meta_train and meta_calibration caches")
    train, statistics = prepare_cards(train_cache)
    calibration, _ = prepare_cards(calibration_cache, statistics)

    variants = {}
    print("R0 evaluating b0_flat_cluster", flush=True)
    flat = flat_retrieve(
        calibration, train, topk=args.topk, exact_direction=False,
        maximum_per_event=args.topk,
    )
    variants["b0_flat_cluster"] = metrics(calibration, flat)
    print(json.dumps({"b0_flat_cluster": variants["b0_flat_cluster"]}, ensure_ascii=False), flush=True)
    for name, exact_direction, per_event in (
        ("b1_building_route", False, args.topk),
        ("b2_direction_route", True, args.topk),
        ("b3_direction_event_diverse", True, args.maximum_per_event),
    ):
        print(f"R0 evaluating {name}", flush=True)
        result = hierarchical_retrieve(
            calibration, train, coarse_buildings=args.coarse_buildings,
            topk=args.topk, exact_direction=exact_direction,
            maximum_per_event=per_event,
        )
        variants[name] = metrics(calibration, result)
        print(json.dumps({name: variants[name]}, ensure_ascii=False), flush=True)

    selected_name = max(variants, key=lambda key: (
        variants[key]["event_macro_auroc"], variants[key]["utility_auroc"]
    ))
    overall_winner = max(variants, key=lambda key: variants[key]["utility_auroc"])
    selected = variants[selected_name]
    baseline = variants["b0_flat_cluster"]
    checks = {
        "utility_auroc_ge_0p60": selected["utility_auroc"] >= .60,
        "event_macro_auroc_ge_0p58": selected["event_macro_auroc"] >= .58,
        "correct_events_ge_10": selected["correct_events"] >= min(10, selected["evaluable_events"]),
        "coverage_ge_0p80": selected["query_coverage"] >= .80,
        "neighbour_purity_ge_0p60": selected["mean_neighbour_purity"] >= .60,
        "mean_neighbour_events_ge_3": selected["mean_neighbour_events"] >= 3.0,
        "beats_flat_cluster_event_macro_by_0p01": (
            selected["event_macro_auroc"] >= baseline["event_macro_auroc"] + .01
        ),
    }
    decision = (
        "ADMIT_HIERARCHICAL_BUILDING_RETRIEVAL_R1"
        if all(checks.values()) else "STOP_HBR_R0_NO_CROSS_EVENT_UTILITY_NEIGHBOURS"
    )
    report = {
        "protocol": PROTOCOL, "configuration": vars(args),
        "train_clusters": len(train.local_key),
        "calibration_clusters": len(calibration.local_key),
        "train_buildings": len(train.building_key),
        "calibration_buildings": len(calibration.building_key),
        "variants": variants, "selected_variant": selected_name,
        "event_macro_winner": selected_name,
        "overall_auroc_winner": overall_winner,
        "checks": checks, "decision": decision,
        "limitations": [
            "building cards aggregate D1 eligible changed clusters only",
            "this audit does not alter P3 or fixed S2 predictions",
        ],
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_name, "checks": checks, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
