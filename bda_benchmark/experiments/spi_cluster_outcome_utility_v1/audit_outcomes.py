from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import bootstrap, event_rows
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import per_image_selective_histograms
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import sha256
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import AUDIT_PROTOCOL
from .core import OUTCOMES, outcome_counts


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"unexpected {split} D1 cache")
    for name in ("meta_confirm_accessed", "val450_accessed", "official_instances_accessed", "test_accessed"):
        if cache.get(name, True):
            raise RuntimeError(f"{split} cache provenance is contaminated")


def _distribution_rows(cache: dict) -> dict:
    counts = outcome_counts(cache)
    cluster_events = [cache["image_events"][int(i)] for i in cache["image_index"]]
    result = {}
    for event in sorted(set(cluster_events)):
        selected = torch.tensor([value == event for value in cluster_events])
        total = counts[selected].sum(0)
        result[event] = {
            **{name: int(total[index]) for index, name in enumerate(OUTCOMES)},
            **{f"{name}_fraction": float(total[index] / total.sum().clamp_min(1.0))
               for index, name in enumerate(OUTCOMES)},
            "clusters": int(selected.sum()),
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("E2.1 local outcome target and Oracle audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--e2-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-oracle-gain-pp", type=float, default=.30)
    parser.add_argument("--minimum-outcome-pixels", type=int, default=1000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("refusing repeated E2.1 outcome audit")
    previous = json.loads(Path(args.e2_json).read_text(encoding="utf-8"))
    if previous.get("decision") != "STOP_DIRECTION_IMPACT_E2_AFTER_LOEO":
        raise RuntimeError("outcome audit requires the frozen failed E2 result")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("P3 provenance mismatch")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta split leakage")
    counts = outcome_counts(calibration)
    # Ties retain fixed S2. Only a strictly harmful cluster is reverted to P3.
    accept = counts[:, 0] >= counts[:, 1]
    selected_by_image = per_image_selective_histograms(calibration, accept)
    fixed = metrics_from_histogram(calibration["fixed_s2_histograms"].sum(0))
    oracle = metrics_from_histogram(selected_by_image.sum(0))
    gain = 100 * (oracle["valid_miou"] - fixed["valid_miou"])
    damaged = 100 * (oracle["damaged_f1"] - fixed["damaged_f1"])
    events = event_rows(calibration, selected_by_image)
    nonnegative = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in events.values())
    bootstrap_row = bootstrap(calibration, selected_by_image, args.bootstrap_iterations, args.seed)
    totals = counts.sum(0)
    checks = {
        "local_outcomes_partition_changed_pixels": True,
        "each_outcome_has_minimum_pixels": bool((totals >= args.minimum_outcome_pixels).all()),
        "oracle_gain_ge_threshold": gain >= args.minimum_oracle_gain_pp,
        "oracle_damaged_f1_not_below_fixed_s2": damaged >= 0,
        "oracle_nonnegative_events_ge_10": nonnegative >= 10,
        "oracle_bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
    }
    decision = "ADMIT_CLUSTER_OUTCOME_E21_LOEO" if all(checks.values()) else "STOP_CLUSTER_OUTCOME_E21_ORACLE_LOW"
    payload = {
        "protocol": AUDIT_PROTOCOL, "configuration": vars(args),
        "outcome_pixel_totals": {name: int(totals[i]) for i, name in enumerate(OUTCOMES)},
        "train_event_distributions": _distribution_rows(train),
        "calibration_event_distributions": _distribution_rows(calibration),
        "fixed_s2": fixed, "local_outcome_oracle": oracle,
        "oracle_gain_vs_fixed_s2_miou_pp": gain,
        "oracle_damaged_f1_change_pp": damaged,
        "oracle_event_nonnegative": nonnegative, "events": events,
        "bootstrap": bootstrap_row, "checks": checks, "decision": decision,
        "provenance": {"train_cache_sha256": sha256(args.train_cache),
                       "calibration_cache_sha256": sha256(args.calibration_cache),
                       "e2_json_sha256": sha256(args.e2_json)},
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Outcome pixels: {payload['outcome_pixel_totals']}")
    print(f"Local outcome Oracle gain: {gain:+.4f}pp D-F1={damaged:+.4f}pp events={nonnegative}/14")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
