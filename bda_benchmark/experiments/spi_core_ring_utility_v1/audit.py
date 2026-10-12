from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.experiments.spi_pseudo_instance_memory_v1.core import permutation_null
from bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence import average_precision
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from .core import PROTOCOL


AUDIT_PROTOCOL = "SPI_CORE_RING_UTILITY_E0_AUDIT_v1"
FEATURE_GROUPS = ("zone_geometry", "zone_semantic", "zone_s2")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache(cache: dict, split: str) -> None:
    if cache.get("protocol") != PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"expected {split} Core-Ring cache")
    if any(bool(cache.get(key, True)) for key in (
        "validation_accessed", "official_instances_accessed", "test_accessed"
    )):
        raise RuntimeError("Core-Ring cache accessed a forbidden split")


def utility_labels(cache: dict, margin: float) -> torch.Tensor:
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
        candidates.append({
            "channel_index": index,
            "channel_name": name,
            "orientation": orientation,
            "train_auroc": max(raw_auc, 1.0 - raw_auc),
            "raw_train_auroc": raw_auc,
        })
    return max(candidates, key=lambda row: (row["train_auroc"], -row["channel_index"]))


def evaluate_scalar(cache: dict, labels: torch.Tensor, group: str, selected: dict) -> tuple[dict, torch.Tensor]:
    valid = labels >= 0
    scores = cache["features"][group].float()[valid, int(selected["channel_index"])]
    scores = scores * float(selected["orientation"])
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
        if int((event_labels == 0).sum()) == 0 or int((event_labels == 1).sum()) == 0:
            continue
        event_rows[event] = {
            "auroc": binary_auroc(event_labels, scores[event_mask]),
            "count": int(event_mask.sum()),
        }
    event_aucs = [row["auroc"] for row in event_rows.values()]
    report = {
        **selected,
        "calibration_auroc": binary_auroc(selected_labels, scores),
        "calibration_average_precision": average_precision(selected_labels, scores),
        "event_macro_auroc": float(np.mean(event_aucs)) if event_aucs else float("nan"),
        "event_direction_correct": sum(value >= 0.5 for value in event_aucs),
        "event_total": len(event_aucs),
        "count": int(valid.sum()),
        "positive": int((selected_labels == 1).sum()),
        "negative": int((selected_labels == 0).sum()),
        "ambiguous": int((labels < 0).sum()),
        "events": event_rows,
    }
    return report, scores


def metric_reports(cache: dict) -> dict[str, dict]:
    return {
        name: metrics_from_histogram(rows.sum(0))
        for name, rows in cache["histograms"].items()
    }


def miou_gain(reports: dict, method: str, reference: str = "fixed_s2") -> float:
    return 100.0 * (
        reports[method]["valid_miou"] - reports[reference]["valid_miou"]
    )


def event_policy_report(cache: dict, method: str, reference: str = "fixed_s2") -> dict:
    gains = {}
    for event in sorted(set(cache["image_events"])):
        indices = [index for index, value in enumerate(cache["image_events"]) if value == event]
        method_report = metrics_from_histogram(cache["histograms"][method][indices].sum(0))
        reference_report = metrics_from_histogram(cache["histograms"][reference][indices].sum(0))
        gains[event] = 100.0 * (method_report["valid_miou"] - reference_report["valid_miou"])
    return {
        "gains_miou_pp": gains,
        "nonnegative": sum(value >= 0.0 for value in gains.values()),
        "total": len(gains),
    }


def zone_summary(cache: dict) -> dict[str, dict[str, float | int]]:
    output = {}
    for zone, row in cache["zone_totals"].items():
        changed = int(row["changed"])
        decisive = int(row["corrected"]) + int(row["harmed"])
        output[zone] = {
            **{key: int(value) for key, value in row.items()},
            "changed_fraction_of_area": changed / max(int(row["area"]), 1),
            "correction_rate_on_changed": int(row["corrected"]) / max(changed, 1),
            "harm_rate_on_changed": int(row["harmed"]) / max(changed, 1),
            "decisive_precision": int(row["corrected"]) / max(decisive, 1),
            "utility_on_changed": int(row["net"]) / max(changed, 1),
        }
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Core-Ring Utility E0 parameter-free audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--minimum-fixed-gain-pp", type=float, default=0.10)
    parser.add_argument("--minimum-random-control-gain-pp", type=float, default=0.05)
    parser.add_argument("--minimum-oracle-gain-vs-building-pp", type=float, default=0.20)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--minimum-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-event-macro", type=float, default=0.58)
    parser.add_argument("--minimum-class-count", type=int, default=200)
    args = parser.parse_args(argv)

    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    validate_cache(train, "meta_train")
    validate_cache(calibration, "meta_calibration")
    if train.get("p3_sha256") != calibration.get("p3_sha256"):
        raise RuntimeError("P3 provenance mismatch")
    if train.get("s2_configuration") != calibration.get("s2_configuration"):
        raise RuntimeError("fixed-S2 configuration mismatch")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta_train and meta_calibration overlap")
    for key in ("minimum_area", "core_radius", "minimum_core_area"):
        if train["configuration"].get(key) != calibration["configuration"].get(key):
            raise RuntimeError(f"cache configuration mismatch: {key}")

    train_reports = metric_reports(train)
    calibration_reports = metric_reports(calibration)
    candidates = ("core_only", "ring_only")
    selected_policy = max(
        candidates,
        key=lambda name: (miou_gain(train_reports, name), -candidates.index(name)),
    )
    random_control = "random_core_only" if selected_policy == "core_only" else "random_ring_only"
    selected_gain = miou_gain(calibration_reports, selected_policy)
    random_gain = miou_gain(calibration_reports, random_control)
    policy_events = event_policy_report(calibration, selected_policy)
    fixed_policy_checks = {
        "gain_vs_fixed_s2_ge_threshold": selected_gain >= args.minimum_fixed_gain_pp,
        "beats_random_same_area_by_margin": (
            selected_gain >= random_gain + args.minimum_random_control_gain_pp
        ),
        "nonnegative_events_ge_threshold": (
            policy_events["nonnegative"] >= args.minimum_nonnegative_events
        ),
    }

    core_ring_oracle_gain = miou_gain(
        calibration_reports, "core_ring_oracle", "building_oracle"
    )
    oracle_events = event_policy_report(calibration, "core_ring_oracle", "building_oracle")
    oracle_checks = {
        "gain_vs_building_oracle_ge_threshold": (
            core_ring_oracle_gain >= args.minimum_oracle_gain_vs_building_pp
        ),
        "nonnegative_events_ge_threshold": (
            oracle_events["nonnegative"] >= args.minimum_nonnegative_events
        ),
    }

    train_labels = utility_labels(train, args.utility_margin)
    calibration_labels = utility_labels(calibration, args.utility_margin)
    evidence_results = {}
    for group in FEATURE_GROUPS:
        selected = select_scalar(train, train_labels, group)
        report, scores = evaluate_scalar(calibration, calibration_labels, group, selected)
        report["permutation"] = permutation_null(
            calibration_labels[calibration_labels >= 0], scores,
            args.permutations, args.seed + 101 + FEATURE_GROUPS.index(group),
        )
        evidence_results[group] = report
    evidence_checks = {
        group: {
            "auroc_ge_threshold": report["calibration_auroc"] >= args.minimum_auroc,
            "event_macro_ge_threshold": report["event_macro_auroc"] >= args.minimum_event_macro,
            "correct_events_ge_threshold": (
                report["event_direction_correct"] >= args.minimum_nonnegative_events
            ),
            "positive_count_ge_threshold": report["positive"] >= args.minimum_class_count,
            "negative_count_ge_threshold": report["negative"] >= args.minimum_class_count,
            "permutation_p_le_0_05": report["permutation"]["one_sided_p"] <= 0.05,
        }
        for group, report in evidence_results.items()
    }
    admitted_evidence = [
        group for group, checks in evidence_checks.items() if all(checks.values())
    ]

    if all(fixed_policy_checks.values()):
        decision = "ADMIT_FIXED_CORE_RING_POLICY_CONFIRM"
    elif all(oracle_checks.values()) and admitted_evidence:
        decision = "ADMIT_CORE_RING_E1_LOW_CAPACITY_VERIFIER"
    elif all(oracle_checks.values()):
        decision = "STOP_CORE_RING_NO_PREDICTIVE_SIGNAL_TRY_RGID_E0"
    else:
        decision = "STOP_CORE_RING_CEILING_LOW_TRY_RGID_E0"

    payload = {
        "protocol": AUDIT_PROTOCOL,
        "selection_contract": "core_only versus ring_only selected on meta_train; meta_calibration evaluated once",
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_cache_sha256": sha256(args.train_cache),
        "calibration_cache": str(Path(args.calibration_cache).resolve()),
        "calibration_cache_sha256": sha256(args.calibration_cache),
        "settings": vars(args),
        "selected_fixed_policy_train_only": selected_policy,
        "selected_fixed_policy_gain_vs_s2_miou_pp": selected_gain,
        "random_same_area_control": random_control,
        "random_control_gain_vs_s2_miou_pp": random_gain,
        "fixed_policy_events": policy_events,
        "fixed_policy_checks": fixed_policy_checks,
        "core_ring_oracle_gain_vs_building_oracle_miou_pp": core_ring_oracle_gain,
        "core_ring_oracle_events": oracle_events,
        "oracle_checks": oracle_checks,
        "zone_summary": zone_summary(calibration),
        "reports": calibration_reports,
        "evidence_results": evidence_results,
        "evidence_checks": evidence_checks,
        "admitted_evidence": admitted_evidence,
        "decision": decision,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 112)
    for name, report in calibration_reports.items():
        print(f"{name:18s} mIoU={100*report['valid_miou']:.4f} D-F1={100*report['damaged_f1']:.4f}")
    print(f"Selected fixed policy : {selected_policy}")
    print(f"Policy/random gain    : {selected_gain:+.4f}pp / {random_gain:+.4f}pp")
    print(f"Policy events         : {policy_events['nonnegative']}/{policy_events['total']}")
    print(f"Core-Ring Oracle gain : {core_ring_oracle_gain:+.4f}pp vs Building Oracle")
    print(f"Zone summary          : {payload['zone_summary']}")
    for group, report in evidence_results.items():
        print(
            f"{group:14s} channel={report['channel_name']} sign={report['orientation']:+.0f} "
            f"AUC={report['calibration_auroc']:.4f} event={report['event_macro_auroc']:.4f} "
            f"correct={report['event_direction_correct']}/{report['event_total']} "
            f"p={report['permutation']['one_sided_p']:.4g}"
        )
    print(f"Checks: fixed={fixed_policy_checks} oracle={oracle_checks}")
    print(f"Admitted evidence: {admitted_evidence}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

