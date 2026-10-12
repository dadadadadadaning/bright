from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence import average_precision
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc


PROTOCOL = "SPI_BUILDING_UTILITY_E0_AUDIT_v1"
BASE_GROUPS = ("shape", "semantic", "s2")
ADVANCED_GROUPS = ("direct", "relation", "matching", "grid", "slic")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_from_cache(cache: dict, margin: float) -> torch.Tensor:
    ratio = cache["utility"]["utility_ratio"].float()
    labels = torch.full_like(ratio, -1, dtype=torch.long)
    labels[ratio >= margin] = 1
    labels[ratio <= -margin] = 0
    return labels


def select_scalar(cache: dict, labels: torch.Tensor, group: str) -> dict[str, object]:
    valid = labels >= 0
    values = cache["features"][group].float()[valid]
    selected_labels = labels[valid]
    candidates = []
    for index, name in enumerate(cache["feature_names"][group]):
        raw_auc = binary_auroc(selected_labels, values[:, index])
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


def evaluate_scalar(cache: dict, labels: torch.Tensor, group: str, selected: dict) -> dict[str, object]:
    valid = labels >= 0
    score_all = cache["features"][group].float()[:, int(selected["channel_index"])]
    score_all = score_all * float(selected["orientation"])
    score = score_all[valid]
    selected_labels = labels[valid]
    image_indices = cache["image_index"].long()[valid]
    event_rows = {}
    for event in sorted(set(cache["image_events"])):
        event_image_ids = torch.tensor(
            [index for index, value in enumerate(cache["image_events"]) if value == event],
            dtype=torch.long,
        )
        event_mask = torch.isin(image_indices, event_image_ids)
        event_labels = selected_labels[event_mask]
        if int((event_labels == 1).sum()) == 0 or int((event_labels == 0).sum()) == 0:
            continue
        event_rows[event] = {
            "auroc": binary_auroc(event_labels, score[event_mask]),
            "count": int(event_mask.sum()),
            "positive": int((event_labels == 1).sum()),
            "negative": int((event_labels == 0).sum()),
        }
    event_aucs = [row["auroc"] for row in event_rows.values()]
    return {
        **selected,
        "calibration_auroc": binary_auroc(selected_labels, score),
        "calibration_average_precision": average_precision(selected_labels, score),
        "count": int(valid.sum()),
        "positive": int((selected_labels == 1).sum()),
        "negative": int((selected_labels == 0).sum()),
        "ambiguous": int((labels < 0).sum()),
        "event_macro_auroc": float(np.mean(event_aucs)) if event_aucs else float("nan"),
        "event_direction_correct": sum(value >= 0.5 for value in event_aucs),
        "event_total": len(event_aucs),
        "events": event_rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("BU-E0 building oracle and parameter-free learnability audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--minimum-oracle-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-oracle-nonnegative-events", type=int, default=10)
    parser.add_argument("--minimum-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-event-macro", type=float, default=0.58)
    parser.add_argument("--minimum-correct-events", type=int, default=10)
    parser.add_argument("--minimum-class-count", type=int, default=200)
    parser.add_argument("--minimum-control-gain", type=float, default=0.02)
    args = parser.parse_args(argv)
    if not 0 < args.utility_margin < 1:
        raise ValueError("utility margin must be in (0,1)")

    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, split in ((train, "meta_train"), (calibration, "meta_calibration")):
        if cache.get("protocol") != "SPI_BUILDING_UTILITY_CACHE_v1" or cache.get("split") != split:
            raise RuntimeError(f"expected {split} BU cache")
        if (
            cache.get("validation_accessed", True)
            or cache.get("official_instances_accessed", True)
            or cache.get("test_accessed", True)
        ):
            raise RuntimeError("BU cache accessed a forbidden split")
    if train.get("p3_sha256") != calibration.get("p3_sha256"):
        raise RuntimeError("P3 provenance mismatch")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta_train and meta_calibration overlap")
    for key in ("minimum_area", "matching_max_side", "matching_window", "cell_size", "evidence_groups"):
        if train["configuration"].get(key) != calibration["configuration"].get(key):
            raise RuntimeError(f"BU cache configuration mismatch: {key}")
    if train.get("s2_configuration") != calibration.get("s2_configuration"):
        raise RuntimeError("fixed-S2 configuration mismatch")
    groups = tuple(
        name for name in (*BASE_GROUPS, *ADVANCED_GROUPS)
        if name in train["features"] and name in calibration["features"]
    )
    train_labels = labels_from_cache(train, args.utility_margin)
    calibration_labels = labels_from_cache(calibration, args.utility_margin)
    if min(int((train_labels == value).sum()) for value in (0, 1)) == 0:
        raise RuntimeError("meta_train lacks one building-utility class")
    if min(int((calibration_labels == value).sum()) for value in (0, 1)) == 0:
        raise RuntimeError("meta_calibration lacks one building-utility class")

    evidence_results = {
        group: evaluate_scalar(calibration, calibration_labels, group, select_scalar(train, train_labels, group))
        for group in groups
    }
    reports = {
        name: metrics_from_histogram(rows.sum(0))
        for name, rows in calibration["histograms"].items()
    }
    oracle_gain_pp = 100.0 * (
        reports["building_oracle"]["valid_miou"] - reports["fixed_s2"]["valid_miou"]
    )
    pixel_oracle_gain_pp = 100.0 * (
        reports["pixel_oracle"]["valid_miou"] - reports["fixed_s2"]["valid_miou"]
    )
    event_reports = {}
    for event in sorted(set(calibration["image_events"])):
        indices = [
            index for index, value in enumerate(calibration["image_events"]) if value == event
        ]
        event_reports[event] = {
            name: metrics_from_histogram(rows[indices].sum(0))
            for name, rows in calibration["histograms"].items()
        }
        event_reports[event]["building_oracle_gain_vs_fixed_s2_miou_pp"] = 100.0 * (
            event_reports[event]["building_oracle"]["valid_miou"]
            - event_reports[event]["fixed_s2"]["valid_miou"]
        )
    oracle_nonnegative_events = sum(
        rows["building_oracle_gain_vs_fixed_s2_miou_pp"] >= 0.0
        for rows in event_reports.values()
    )
    oracle_checks = {
        "gain_vs_fixed_s2_ge_threshold": oracle_gain_pp >= args.minimum_oracle_gain_pp,
        "nonnegative_events_ge_threshold": (
            oracle_nonnegative_events >= args.minimum_oracle_nonnegative_events
        ),
    }
    oracle_check = all(oracle_checks.values())

    absolute_checks = {}
    for group, result in evidence_results.items():
        absolute_checks[group] = {
            "auroc_ge_threshold": result["calibration_auroc"] >= args.minimum_auroc,
            "event_macro_ge_threshold": result["event_macro_auroc"] >= args.minimum_event_macro,
            "at_least_ten_evaluable_events": result["event_total"] >= args.minimum_correct_events,
            "correct_events_ge_threshold": result["event_direction_correct"] >= args.minimum_correct_events,
            "positive_count_ge_threshold": result["positive"] >= args.minimum_class_count,
            "negative_count_ge_threshold": result["negative"] >= args.minimum_class_count,
        }
    simple_groups = [name for name in BASE_GROUPS if name in evidence_results]
    simple_control_auc = max(evidence_results[name]["calibration_auroc"] for name in simple_groups)
    admitted_base = [name for name in simple_groups if all(absolute_checks[name].values())]
    advanced_checks, admitted_advanced = {}, []
    for group in ADVANCED_GROUPS:
        if group not in evidence_results:
            continue
        reference_auc = simple_control_auc
        if group in ("relation", "matching") and "direct" in evidence_results:
            reference_auc = max(reference_auc, evidence_results["direct"]["calibration_auroc"])
        group_checks = {
            **absolute_checks[group],
            "beats_simple_reference_by_margin": evidence_results[group]["calibration_auroc"] >= reference_auc + args.minimum_control_gain,
        }
        if group == "slic" and "grid" in evidence_results:
            group_checks["beats_equal_budget_grid_by_0_01"] = (
                evidence_results["slic"]["calibration_auroc"]
                >= evidence_results["grid"]["calibration_auroc"] + 0.01
            )
        advanced_checks[group] = group_checks
        if all(group_checks.values()):
            admitted_advanced.append(group)

    admitted = admitted_base + admitted_advanced
    if not oracle_check:
        decision = "STOP_BUILDING_UTILITY_ORACLE_CEILING_TOO_LOW"
    elif not admitted:
        decision = "STOP_BUILDING_UTILITY_NO_LEARNABLE_EVIDENCE"
    else:
        decision = "ADMIT_BU_E1_SINGLE_BUILDING_VERIFIER"
    label_distribution = {
        "meta_train": {
            "positive": int((train_labels == 1).sum()),
            "negative": int((train_labels == 0).sum()),
            "ambiguous": int((train_labels < 0).sum()),
            "building_oracle_accept_net_gt_zero": int((train["utility"]["net"] > 0).sum()),
        },
        "meta_calibration": {
            "positive": int((calibration_labels == 1).sum()),
            "negative": int((calibration_labels == 0).sum()),
            "ambiguous": int((calibration_labels < 0).sum()),
            "building_oracle_accept_net_gt_zero": int((calibration["utility"]["net"] > 0).sum()),
        },
    }
    payload = {
        "protocol": PROTOCOL,
        "utility_label": {
            "positive": f"net/changed >= +{args.utility_margin}",
            "negative": f"net/changed <= -{args.utility_margin}",
            "ambiguous": "excluded from AUROC",
        },
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_cache_sha256": sha256(args.train_cache),
        "calibration_cache": str(Path(args.calibration_cache).resolve()),
        "calibration_cache_sha256": sha256(args.calibration_cache),
        "reports": reports,
        "label_distribution": label_distribution,
        "event_reports": event_reports,
        "building_oracle_gain_vs_fixed_s2_miou_pp": oracle_gain_pp,
        "pixel_oracle_gain_vs_fixed_s2_miou_pp": pixel_oracle_gain_pp,
        "building_oracle_nonnegative_events": oracle_nonnegative_events,
        "building_oracle_event_total": len(event_reports),
        "oracle_checks": oracle_checks,
        "oracle_passed": oracle_check,
        "evidence_results": evidence_results,
        "absolute_checks": absolute_checks,
        "simple_control_auroc": simple_control_auc,
        "advanced_checks": advanced_checks,
        "admitted_base_groups": admitted_base,
        "admitted_advanced_groups": admitted_advanced,
        "admitted_groups": admitted,
        "decision": decision,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 116)
    for name, report in reports.items():
        print(f"{name:16s} mIoU={100*report['valid_miou']:.4f} D-F1={100*report['damaged_f1']:.4f}")
    print(f"Building Oracle gain vs fixed S2: {oracle_gain_pp:+.4f}pp")
    print(f"Pixel Oracle gain vs fixed S2   : {pixel_oracle_gain_pp:+.4f}pp")
    print(f"Building Oracle nonnegative events: {oracle_nonnegative_events}/{len(event_reports)}")
    print(f"Building labels: {label_distribution}")
    for group, result in evidence_results.items():
        print(
            f"{group:9s} channel={result['channel_name']} sign={result['orientation']:+.0f} "
            f"AUC={result['calibration_auroc']:.4f} AP={result['calibration_average_precision']:.4f} "
            f"event={result['event_macro_auroc']:.4f} "
            f"correct={result['event_direction_correct']}/{result['event_total']}"
        )
    print(f"Admitted base: {admitted_base}")
    print(f"Admitted advanced: {admitted_advanced}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
