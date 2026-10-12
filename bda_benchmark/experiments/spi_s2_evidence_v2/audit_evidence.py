from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc


PROTOCOL = "SPI_S2_PARAMETER_FREE_EVIDENCE_AUDIT_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool().flatten()
    scores = scores.float().flatten()
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order]
    precision = ranked.cumsum(0).float() / torch.arange(1, ranked.numel() + 1).float()
    return float(precision[ranked].mean())


def select_scalar(train: dict, group: str) -> dict[str, object]:
    labels = train["selector_label"].long()
    valid = labels >= 0
    values = train["evidence"][group].float()[valid]
    labels = labels[valid]
    names = tuple(train["evidence_channel_names"][group])
    candidates = []
    for index, name in enumerate(names):
        raw_auc = binary_auroc(labels, values[:, index])
        orientation = 1.0 if raw_auc >= 0.5 else -1.0
        candidates.append(
            {
                "channel_index": index,
                "channel_name": name,
                "orientation": orientation,
                "train_auroc": max(raw_auc, 1.0 - raw_auc),
                "raw_train_auroc": raw_auc,
            }
        )
    return max(candidates, key=lambda row: (row["train_auroc"], -row["channel_index"]))


def evaluate_scalar(calibration: dict, group: str, selected: dict) -> dict[str, object]:
    labels_all = calibration["selector_label"].long()
    valid = labels_all >= 0
    labels = labels_all[valid]
    score_all = (
        calibration["evidence"][group].float()[:, int(selected["channel_index"])]
        * float(selected["orientation"])
    )
    score = score_all[valid]
    image_indices = calibration["image_index"].long()[valid]
    events = calibration["image_events"]
    event_rows = {}
    for event in sorted(set(events)):
        image_ids = torch.tensor(
            [index for index, value in enumerate(events) if value == event], dtype=torch.long
        )
        event_mask = torch.isin(image_indices, image_ids)
        event_labels = labels[event_mask]
        if int((event_labels == 1).sum()) == 0 or int((event_labels == 0).sum()) == 0:
            continue
        event_rows[event] = {
            "auroc": binary_auroc(event_labels, score[event_mask]),
            "count": int(event_mask.sum()),
        }
    event_aucs = [row["auroc"] for row in event_rows.values()]
    return {
        **selected,
        "calibration_auroc": binary_auroc(labels, score),
        "calibration_average_precision": average_precision(labels, score),
        "decisive_count": int(valid.sum()),
        "positives": int((labels == 1).sum()),
        "negatives": int((labels == 0).sum()),
        "event_macro_auroc": float(np.mean(event_aucs)) if event_aucs else float("nan"),
        "event_direction_correct": sum(value >= 0.5 for value in event_aucs),
        "event_total": len(event_aucs),
        "events": event_rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train-oriented, calibration-evaluated evidence audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, expected in ((train, "train"), (calibration, "internal_calibration")):
        if cache.get("protocol") != "SPI_S2_EVIDENCE_SAMPLE_CACHE_v2":
            raise RuntimeError("unexpected evidence cache protocol")
        if cache.get("split") != expected:
            raise RuntimeError(f"expected {expected} cache")
        if cache.get("validation_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError("evidence cache accessed a forbidden split")
    if train.get("p3_sha256") != calibration.get("p3_sha256"):
        raise RuntimeError("train/calibration P3 provenance mismatch")
    groups = tuple(name for name in ("direct", "relation", "matching", "grid", "slic") if name in train["evidence"] and name in calibration["evidence"])
    results = {}
    for group in groups:
        results[group] = evaluate_scalar(calibration, group, select_scalar(train, group))

    direct_auc = results["direct"]["calibration_auroc"]
    checks = {}
    admitted = []
    for group in ("relation", "matching"):
        if group not in results:
            continue
        result = results[group]
        group_checks = {
            "auroc_ge_0_60": result["calibration_auroc"] >= 0.60,
            "event_macro_ge_0_58": result["event_macro_auroc"] >= 0.58,
            "ten_events_direction_correct": result["event_direction_correct"] >= min(10, result["event_total"]),
            "beats_direct_by_0_02": result["calibration_auroc"] >= direct_auc + 0.02,
        }
        checks[group] = group_checks
        if all(group_checks.values()):
            admitted.append(group)
    if "slic" in results and "grid" in results:
        slic = results["slic"]
        grid = results["grid"]
        slic_checks = {
            "auroc_ge_0_60": slic["calibration_auroc"] >= 0.60,
            "event_macro_ge_0_58": slic["event_macro_auroc"] >= 0.58,
            "ten_events_direction_correct": slic["event_direction_correct"] >= min(10, slic["event_total"]),
            "beats_equal_budget_grid_by_0_01": slic["calibration_auroc"] >= grid["calibration_auroc"] + 0.01,
        }
        checks["slic"] = slic_checks
        if all(slic_checks.values()):
            admitted.append("slic")

    base_hist = calibration["base_histograms"].sum(0)
    fixed_hist = calibration["fixed_histograms"].sum(0)
    labels = calibration["selector_label"].long()
    e0 = {
        "p3": metrics_from_histogram(base_hist),
        "fixed_s2": metrics_from_histogram(fixed_hist),
        "changed": int(labels.numel()),
        "corrected": int((labels == 1).sum()),
        "harmed": int((labels == 0).sum()),
        "neutral": int((labels < 0).sum()),
    }
    payload = {
        "protocol": PROTOCOL,
        "scope": "Train-selected scalar orientation; Train2918 internal calibration; no Val450",
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_cache_sha256": sha256_file(args.train_cache),
        "calibration_cache": str(Path(args.calibration_cache).resolve()),
        "calibration_cache_sha256": sha256_file(args.calibration_cache),
        "e0_fixed_s2_lock": e0,
        "results": results,
        "checks": checks,
        "admitted_groups": admitted,
        "decision": "ADMIT_SINGLE_EVIDENCE_VERIFIER" if admitted else "STOP_EVIDENCE_VERIFIER_TRAINING",
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 112)
    print(
        f"E0 P3={100*e0['p3']['valid_miou']:.4f} | "
        f"fixed-S2={100*e0['fixed_s2']['valid_miou']:.4f} | "
        f"corrected/harmed/neutral={e0['corrected']}/{e0['harmed']}/{e0['neutral']}"
    )
    for group, row in results.items():
        print(
            f"{group:9s} channel={row['channel_name']} sign={row['orientation']:+.0f} "
            f"AUC={row['calibration_auroc']:.4f} AP={row['calibration_average_precision']:.4f} "
            f"event={row['event_macro_auroc']:.4f} correct={row['event_direction_correct']}/{row['event_total']}"
        )
    print(f"Checks: {checks}")
    print(f"Admitted: {admitted}")
    print(f"Decision: {payload['decision']}")
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
