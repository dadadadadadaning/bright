from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence import average_precision
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PROTOCOL


GROUPS = ("baseline", "topology", "distance", "direction", "probability", "shuffled_topology", "shuffled_distance")
SPATIAL_GROUPS = ("topology", "distance", "direction")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utility_labels(cache: dict, margin: float) -> torch.Tensor:
    ratio = cache["utility"]["utility_ratio"].float()
    labels = torch.full(ratio.shape, -1, dtype=torch.long)
    labels[ratio >= margin] = 1
    labels[ratio <= -margin] = 0
    return labels


def select_scalar(cache: dict, labels: torch.Tensor, group: str) -> dict:
    valid = labels >= 0
    values, target = cache["features"][group].float()[valid], labels[valid]
    rows = []
    for index, name in enumerate(cache["feature_names"][group]):
        raw = binary_auroc(target, values[:, index])
        rows.append({"index": index, "name": name, "orientation": 1.0 if raw >= 0.5 else -1.0,
                     "train_auroc": max(raw, 1.0 - raw)})
    return max(rows, key=lambda row: (row["train_auroc"], -row["index"]))


def evaluate(cache: dict, labels: torch.Tensor, group: str, selected: dict) -> tuple[dict, torch.Tensor, torch.Tensor]:
    valid = labels >= 0
    scores = cache["features"][group].float()[:, selected["index"]] * selected["orientation"]
    target, score = labels[valid], scores[valid]
    row_images = cache["image_index"].long()[valid]
    events = {}
    for event in sorted(set(cache["image_events"])):
        ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        mask = torch.isin(row_images, ids)
        current = target[mask]
        if int((current == 0).sum()) and int((current == 1).sum()):
            events[event] = binary_auroc(current, score[mask])
    aucs = list(events.values())
    return ({
        **selected, "calibration_auroc": binary_auroc(target, score),
        "average_precision": average_precision(target, score),
        "count": int(valid.sum()), "positive": int((target == 1).sum()),
        "negative": int((target == 0).sum()),
        "event_macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "correct_events": sum(value > 0.5 for value in aucs), "event_total": len(aucs),
        "events": events,
    }, target, score)


def permutation_p(labels: torch.Tensor, scores: torch.Tensor, observed: float, iterations: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    exceed = 1
    for _ in range(iterations):
        shuffled = labels[torch.randperm(len(labels), generator=generator)]
        raw = binary_auroc(shuffled, scores)
        exceed += int(max(raw, 1.0 - raw) >= observed)
    return exceed / (iterations + 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("S2 modification-pattern parameter-free audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--minimum-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-event-macro", type=float, default=0.58)
    parser.add_argument("--minimum-correct-events", type=int, default=10)
    parser.add_argument("--minimum-control-gain", type=float, default=0.02)
    parser.add_argument("--minimum-class-count", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, split in ((train, "meta_train"), (calibration, "meta_calibration")):
        if cache.get("protocol") != PROTOCOL or cache.get("split") != split:
            raise RuntimeError(f"expected {PROTOCOL} {split}")
        if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True) or cache.get("official_instances_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError("cache accessed a forbidden split")
    if train["p3_sha256"] != calibration["p3_sha256"] or set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("cache provenance mismatch or split overlap")
    train_labels = utility_labels(train, args.utility_margin)
    calibration_labels = utility_labels(calibration, args.utility_margin)
    results, score_rows = {}, {}
    for group in GROUPS:
        result, target, score = evaluate(calibration, calibration_labels, group, select_scalar(train, train_labels, group))
        result["permutation_p"] = permutation_p(target, score, result["calibration_auroc"], args.permutations, args.seed + GROUPS.index(group))
        results[group], score_rows[group] = result, score
    baseline_auc = results["baseline"]["calibration_auroc"]
    shuffled_auc = max(results["shuffled_topology"]["calibration_auroc"], results["shuffled_distance"]["calibration_auroc"])
    checks, admitted = {}, []
    for group in SPATIAL_GROUPS:
        row = results[group]
        group_checks = {
            "auroc_ge_threshold": row["calibration_auroc"] >= args.minimum_auroc,
            "event_macro_ge_threshold": row["event_macro_auroc"] >= args.minimum_event_macro,
            "correct_events_ge_threshold": row["correct_events"] >= args.minimum_correct_events,
            "at_least_ten_evaluable_events": row["event_total"] >= args.minimum_correct_events,
            "positive_count_ge_threshold": row["positive"] >= args.minimum_class_count,
            "negative_count_ge_threshold": row["negative"] >= args.minimum_class_count,
            "beats_baseline_by_margin": row["calibration_auroc"] >= baseline_auc + args.minimum_control_gain,
            "beats_spatial_shuffle_by_margin": row["calibration_auroc"] >= shuffled_auc + args.minimum_control_gain,
            "permutation_p_lt_0_05": row["permutation_p"] < 0.05,
        }
        checks[group] = group_checks
        if all(group_checks.values()):
            admitted.append(group)
    decision = "ADMIT_SMP_E1_SINGLE_VERIFIER" if admitted else "STOP_SMP_NO_SPATIAL_UTILITY_SIGNAL"
    payload = {
        "protocol": "SPI_S2_MODIFICATION_PATTERN_E0_AUDIT_v1", "configuration": vars(args),
        "train_cache_sha256": _sha256(args.train_cache), "calibration_cache_sha256": _sha256(args.calibration_cache),
        "results": results, "baseline_auroc": baseline_auc, "strongest_shuffled_auroc": shuffled_auc,
        "checks": checks, "admitted_groups": admitted, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for group, row in results.items():
        print(f"SMP-E0 {group:20s} AUC={row['calibration_auroc']:.4f} event={row['event_macro_auroc']:.4f} correct={row['correct_events']}/{row['event_total']} p={row['permutation_p']:.5f} channel={row['name']}")
    print(f"Admitted: {admitted}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

