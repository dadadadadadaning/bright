from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (class_similarity_scores, classification_metrics, event_metrics,
                   prepare_cache, probability_from_scores)


def quantiles(value: torch.Tensor) -> dict:
    return {name: float(torch.quantile(value.float(), q)) for name, q in
            (("q10", .10), ("q25", .25), ("q50", .50), ("q75", .75), ("q90", .90))}


def retrieval_row(train, calibration, exclude_same_event, purity_threshold,
                  shuffled, topk, seed):
    labels = train.hard_label.clone()
    if shuffled:
        labels = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))]
    scores, availability = class_similarity_scores(
        calibration.x, train.x, labels, calibration.event, train.event,
        topk=topk, exclude_same_event=exclude_same_event,
        purity=train.purity, minimum_purity=purity_threshold,
    )
    probability = probability_from_scores(scores)
    return {
        "overall": classification_metrics(calibration.hard_label, probability),
        "event": event_metrics(calibration, probability),
        "availability_min": availability.min(0).values.tolist(),
        "availability_median": availability.float().median(0).values.tolist(),
        "availability_lt_topk_fraction": (availability < topk).float().mean(0).tolist(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Memory V2 D0 label, coverage and naked-retrieval audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, split in ((train_cache, "meta_train"), (calibration_cache, "meta_calibration")):
        if cache.get("split") != split:
            raise RuntimeError(f"D0 expected {split}")
        if cache.get("meta_confirm_accessed", False) or cache.get("val450_accessed", False) or cache.get("test_accessed", False):
            raise RuntimeError("D0 received a forbidden cache")
    train, mean, std = prepare_cache(train_cache)
    calibration, _, _ = prepare_cache(calibration_cache, mean=mean, std=std)

    class_names = ("intact", "damaged", "destroyed")
    class_audit = {}
    for label, name in enumerate(class_names):
        mask = train.hard_label == label
        event_count = len(torch.unique(train.event[mask]))
        image_count = len(torch.unique(train.image[mask]))
        class_audit[name] = {
            "count": int(mask.sum()), "events": event_count, "images": image_count,
            "purity": quantiles(train.purity[mask]),
            "purity_ge_0p8_fraction": float((train.purity[mask] >= .8).float().mean()),
            "foreground_fraction": quantiles(train.foreground_fraction[mask]),
        }
    retrieval = {
        "same_event_allowed": retrieval_row(train, calibration, False, 0.0, False, args.topk, args.seed),
        "cross_event_all": retrieval_row(train, calibration, True, 0.0, False, args.topk, args.seed),
        "cross_event_pure80": retrieval_row(train, calibration, True, 0.8, False, args.topk, args.seed),
        "cross_event_shuffled": retrieval_row(train, calibration, True, 0.0, True, args.topk, args.seed + 99),
    }
    same = retrieval["same_event_allowed"]["overall"]["macro_f1"]
    cross = retrieval["cross_event_all"]["overall"]["macro_f1"]
    shuffled = retrieval["cross_event_shuffled"]["overall"]["macro_f1"]
    checks = {
        "all_classes_ge_200": all(row["count"] >= 200 for row in class_audit.values()),
        "all_classes_span_ge_3_events": all(row["events"] >= 3 for row in class_audit.values()),
        "cross_event_has_full_topk": max(retrieval["cross_event_all"]["availability_lt_topk_fraction"]) <= .05,
        "real_memory_beats_shuffled": cross >= shuffled + .01,
        "same_cross_gap_le_0p05": same - cross <= .05,
    }
    decision = "ADMIT_CROSS_EVENT_METRIC_M1" if checks["all_classes_ge_200"] and checks["all_classes_span_ge_3_events"] else "STOP_MEMORY_LABEL_OR_COVERAGE_FATAL"
    report = {
        "protocol": PROTOCOL, "stage": "D0", "configuration": vars(args),
        "train_samples": len(train.x), "calibration_samples": len(calibration.x),
        "class_audit": class_audit, "retrieval": retrieval,
        "same_minus_cross_macro_f1": same - cross,
        "cross_minus_shuffled_macro_f1": cross - shuffled,
        "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"class_audit": class_audit, "same_macro_f1": same,
                      "cross_macro_f1": cross, "shuffled_macro_f1": shuffled,
                      "checks": checks, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
