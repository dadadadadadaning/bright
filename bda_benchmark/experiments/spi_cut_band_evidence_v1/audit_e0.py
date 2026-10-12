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


GROUPS = ("geometry", "rgb_edge", "sar_edge", "same_position", "local_matching", "shuffled_matching")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels(cache: dict, margin: float) -> torch.Tensor:
    ratio = cache["utility"]["utility_ratio"].float()
    output = torch.full(ratio.shape, -1, dtype=torch.long)
    output[ratio >= margin] = 1
    output[ratio <= -margin] = 0
    return output


def select_channel(cache: dict, target: torch.Tensor, group: str) -> dict:
    valid = target >= 0
    values = cache["features"][group].float()[valid]
    target = target[valid]
    rows = []
    candidate_indices = range(values.shape[1])
    if group in {"local_matching", "shuffled_matching"}:
        # Index 0--4 are summaries of colocated cosine distance.  Excluding
        # them prevents the local matcher from winning with a direct feature.
        candidate_indices = range(5, values.shape[1])
    for index in candidate_indices:
        name = cache["feature_names"][group][index]
        raw = binary_auroc(target, values[:, index])
        rows.append({"index": index, "name": name, "orientation": 1.0 if raw >= 0.5 else -1.0,
                     "train_auroc": max(raw, 1.0 - raw)})
    return max(rows, key=lambda row: (row["train_auroc"], -row["index"]))


def evaluate(cache: dict, target: torch.Tensor, group: str, selected: dict) -> dict:
    valid = target >= 0
    scores = cache["features"][group].float()[:, selected["index"]] * selected["orientation"]
    row_target, row_scores = target[valid], scores[valid]
    row_images = cache["image_index"].long()[valid]
    event_rows = {}
    for event in sorted(set(cache["image_events"])):
        image_ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        mask = torch.isin(row_images, image_ids)
        current = row_target[mask]
        if int((current == 0).sum()) and int((current == 1).sum()):
            event_rows[event] = binary_auroc(current, row_scores[mask])
    aucs = list(event_rows.values())
    return {
        **selected, "calibration_auroc": binary_auroc(row_target, row_scores),
        "average_precision": average_precision(row_target, row_scores),
        "count": int(valid.sum()), "positive": int((row_target == 1).sum()),
        "negative": int((row_target == 0).sum()),
        "event_macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "correct_events": sum(value > 0.5 for value in aucs), "event_total": len(aucs),
        "events": event_rows,
    }


def permutation_p(target: torch.Tensor, score: torch.Tensor, observed: float, iterations: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    exceed = 1
    for _ in range(iterations):
        permuted = target[torch.randperm(len(target), generator=generator)]
        auc = binary_auroc(permuted, score)
        exceed += int(max(auc, 1.0 - auc) >= observed)
    return exceed / (iterations + 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Parameter-free cut-band evidence admission")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.05)
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
            raise RuntimeError(f"expected {PROTOCOL} {split} cache")
        if cache.get("val450_accessed", True) or cache.get("official_instances_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError("cache accessed a forbidden evaluation source")
    if train["p3_sha256"] != calibration["p3_sha256"] or train["rgid_sha256"] != calibration["rgid_sha256"]:
        raise RuntimeError("train/calibration provenance mismatch")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta_train and meta_calibration overlap")
    train_y, cal_y = labels(train, args.utility_margin), labels(calibration, args.utility_margin)
    results = {group: evaluate(calibration, cal_y, group, select_channel(train, train_y, group)) for group in GROUPS}
    local = results["local_matching"]
    controls = max(results[name]["calibration_auroc"] for name in ("geometry", "rgb_edge", "sar_edge", "same_position"))
    valid = cal_y >= 0
    local_scores = calibration["features"]["local_matching"].float()[valid, local["index"]] * local["orientation"]
    p_value = permutation_p(cal_y[valid], local_scores, local["calibration_auroc"], args.permutations, args.seed)
    checks = {
        "local_auroc_ge_threshold": local["calibration_auroc"] >= args.minimum_auroc,
        "local_event_macro_ge_threshold": local["event_macro_auroc"] >= args.minimum_event_macro,
        "local_correct_events_ge_threshold": local["correct_events"] >= args.minimum_correct_events,
        "at_least_ten_evaluable_events": local["event_total"] >= args.minimum_correct_events,
        "positive_count_ge_threshold": local["positive"] >= args.minimum_class_count,
        "negative_count_ge_threshold": local["negative"] >= args.minimum_class_count,
        "beats_geometry_rgb_colocated_by_margin": local["calibration_auroc"] >= controls + args.minimum_control_gain,
        "beats_spatial_shuffle_by_margin": local["calibration_auroc"] >= results["shuffled_matching"]["calibration_auroc"] + args.minimum_control_gain,
        "permutation_p_lt_0_05": p_value < 0.05,
    }
    decision = "ADMIT_CUT_BAND_VERIFIER_E1" if all(checks.values()) else "STOP_CUT_BAND_EVIDENCE_NO_LOCAL_SIGNAL"
    payload = {
        "protocol": "SPI_CUT_BAND_E0_AUDIT_v1", "configuration": vars(args),
        "train_cache_sha256": sha256(args.train_cache),
        "calibration_cache_sha256": sha256(args.calibration_cache),
        "results": results, "strongest_control_auroc": controls,
        "local_matching_permutation_p": p_value, "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for group, row in results.items():
        print(f"CB-E0 {group:18s} AUC={row['calibration_auroc']:.4f} event={row['event_macro_auroc']:.4f} correct={row['correct_events']}/{row['event_total']} channel={row['name']}")
    print(f"Permutation p: {p_value:.6f}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
