from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.audit_e0 import (
    evaluate_cross_event, permutation_p, utility_labels,
)

from . import PROTOCOL


MAIN_GROUPS = (
    "spatial", "stage2_role", "rgb_edge", "sar_texture", "stage2_direct",
    "stage2_matching", "structure_relation",
)
CONTROL_GROUPS = (
    "shuffled_rgb_edge", "shuffled_sar_texture", "shuffled_stage2_matching",
    "shuffled_structure_relation",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"expected {PROTOCOL} {split}")
    if cache.get("meta_confirm_accessed", True):
        raise RuntimeError("D1 E0 may not access meta_confirm")
    if cache.get("val450_accessed", True) or cache.get("official_instances_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError("D1 cache accessed a forbidden split")
    if int(cache["configuration"]["minimum_cluster_area"]) != 33:
        raise RuntimeError("D1 preregistration requires area > 32 (minimum 33)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("D1 parameter-free high-resolution cluster audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--d0-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--minimum-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-weighted-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-event-macro", type=float, default=0.58)
    parser.add_argument("--minimum-correct-events", type=int, default=10)
    parser.add_argument("--minimum-control-gain", type=float, default=0.02)
    parser.add_argument("--minimum-class-count", type=int, default=100)
    parser.add_argument("--minimum-direction-class-count", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    d0 = json.loads(Path(args.d0_json).read_text(encoding="utf-8"))
    if d0.get("decision") != "ADMIT_CLUSTER_DETAIL_D1_STAGE2_EDGE_EVIDENCE":
        raise RuntimeError("D1 refused: D0 did not admit high-resolution evidence")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("train/calibration P3 fingerprints differ")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("train/calibration images overlap")

    train_labels = utility_labels(train, args.utility_margin)
    labels = utility_labels(calibration, args.utility_margin)
    results, score_rows = {}, {}
    for index, group in enumerate(MAIN_GROUPS + CONTROL_GROUPS):
        row, target, scores = evaluate_cross_event(
            train, train_labels, calibration, labels, group,
            args.minimum_direction_class_count,
        )
        row["permutation_p"] = None
        if group in ("stage2_role", "rgb_edge", "sar_texture", "stage2_matching", "structure_relation"):
            row["permutation_p"] = permutation_p(
                target, scores, row["calibration_auroc"], args.permutations,
                args.seed + index,
            )
        results[group], score_rows[group] = row, scores

    spatial_auc = results["spatial"]["calibration_auroc"]
    direct_auc = results["stage2_direct"]["calibration_auroc"]
    controls = {
        "stage2_role": spatial_auc,
        "rgb_edge": max(spatial_auc, results["shuffled_rgb_edge"]["calibration_auroc"]),
        "sar_texture": max(spatial_auc, results["shuffled_sar_texture"]["calibration_auroc"]),
        "stage2_matching": max(direct_auc, results["shuffled_stage2_matching"]["calibration_auroc"]),
        "structure_relation": max(direct_auc, results["shuffled_structure_relation"]["calibration_auroc"]),
    }
    checks, admitted = {}, []
    for group, control_auc in controls.items():
        row = results[group]
        current = {
            "auroc_ge_threshold": row["calibration_auroc"] >= args.minimum_auroc,
            "pixel_weighted_auroc_ge_threshold": row["pixel_weighted_auroc"] >= args.minimum_weighted_auroc,
            "event_macro_ge_threshold": row["event_macro_auroc"] >= args.minimum_event_macro,
            "correct_events_ge_threshold": row["correct_events"] >= args.minimum_correct_events,
            "at_least_ten_evaluable_events": row["event_total"] >= args.minimum_correct_events,
            "positive_count_ge_threshold": row["positive"] >= args.minimum_class_count,
            "negative_count_ge_threshold": row["negative"] >= args.minimum_class_count,
            "beats_control_by_margin": row["calibration_auroc"] >= control_auc + args.minimum_control_gain,
            "permutation_p_lt_0_05": row["permutation_p"] < 0.05,
        }
        checks[group] = current
        if all(current.values()):
            admitted.append(group)
    structural = {"rgb_edge", "sar_texture", "stage2_matching", "structure_relation"}
    structural_admitted = [name for name in admitted if name in structural]
    selected_group = (
        max(
            structural_admitted,
            key=lambda name: (
                results[name]["event_macro_auroc"],
                results[name]["pixel_weighted_auroc"],
                results[name]["calibration_auroc"],
            ),
        )
        if structural_admitted else None
    )
    decision = (
        "ADMIT_CLUSTER_DETAIL_D1_E1_SINGLE_VERIFIER"
        if selected_group is not None
        else "STOP_CLUSTER_DETAIL_D1_NO_HIGH_RESOLUTION_SIGNAL"
    )
    payload = {
        "protocol": "SPI_CLUSTER_DETAIL_D1_E0_AUDIT_v1", "configuration": vars(args),
        "train_cache_sha256": _sha256(args.train_cache),
        "calibration_cache_sha256": _sha256(args.calibration_cache),
        "d0_json": str(Path(args.d0_json).resolve()), "results": results,
        "controls": controls, "checks": checks, "admitted_groups": admitted,
        "selected_group": selected_group,
        "decision": decision, "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, row in results.items():
        print(f"CD-D1 {name:28s} AUC={row['calibration_auroc']:.4f} weighted={row['pixel_weighted_auroc']:.4f} event={row['event_macro_auroc']:.4f} correct={row['correct_events']}/{row['event_total']} p={row['permutation_p']}", flush=True)
    print(f"Admitted: {admitted}")
    print(f"Selected single evidence group: {selected_group}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
