from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import PROTOCOL


METHODS = ("p3", "fixed_s2", "building_oracle", "cluster_oracle", "pixel_oracle")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gain(reports: dict, method: str, reference: str) -> float:
    return 100.0 * (reports[method]["valid_miou"] - reports[reference]["valid_miou"])


def _reports(cache: dict) -> dict:
    return {
        name: metrics_from_histogram(cache["histograms"][name].sum(0))
        for name in METHODS
    }


def _event_reports(cache: dict) -> dict:
    output = {}
    for event in sorted(set(cache["image_events"])):
        indices = [index for index, value in enumerate(cache["image_events"]) if value == event]
        reports = {
            name: metrics_from_histogram(cache["histograms"][name][indices].sum(0))
            for name in METHODS
        }
        reports["cluster_minus_building_miou_pp"] = _gain(reports, "cluster_oracle", "building_oracle")
        reports["cluster_minus_fixed_s2_miou_pp"] = _gain(reports, "cluster_oracle", "fixed_s2")
        output[event] = reports
    return output


def _bootstrap(cache: dict, iterations: int, seed: int) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(seed)
    building = cache["histograms"]["building_oracle"].double()
    cluster = cache["histograms"]["cluster_oracle"].double()
    gains = []
    for _ in range(iterations):
        indices = torch.randint(0, len(building), (len(building),), generator=generator)
        reports = {
            "building_oracle": metrics_from_histogram(building[indices].sum(0)),
            "cluster_oracle": metrics_from_histogram(cluster[indices].sum(0)),
        }
        gains.append(_gain(reports, "cluster_oracle", "building_oracle"))
    values = np.asarray(gains)
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": iterations,
    }


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"expected {PROTOCOL} {split}")
    forbidden = ("meta_confirm_accessed", "val450_accessed", "official_instances_accessed", "test_accessed")
    if any(cache.get(key, True) for key in forbidden):
        raise RuntimeError("cluster-oracle cache accessed a forbidden split")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("S2 modification-cluster oracle ceiling audit")
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gain-vs-building-pp", type=float, default=0.10)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--minimum-accepted-clusters", type=int, default=200)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(cache, "meta_calibration")
    reports = _reports(cache)
    events = _event_reports(cache)
    bootstrap = _bootstrap(cache, args.bootstrap_iterations, args.seed)
    cluster_vs_building = _gain(reports, "cluster_oracle", "building_oracle")
    cluster_vs_s2 = _gain(reports, "cluster_oracle", "fixed_s2")
    pixel_vs_building = _gain(reports, "pixel_oracle", "building_oracle")
    nonnegative_events = sum(row["cluster_minus_building_miou_pp"] >= 0 for row in events.values())
    accepted = int(cache["counts"].get("cluster_oracle_accepted", 0))
    checks = {
        "cluster_gain_vs_building_ge_threshold": cluster_vs_building >= args.minimum_gain_vs_building_pp,
        "cluster_nonnegative_events_ge_threshold": nonnegative_events >= args.minimum_nonnegative_events,
        "accepted_clusters_ge_threshold": accepted >= args.minimum_accepted_clusters,
        "bootstrap_ci_low_gt_zero": bootstrap["ci95_low_pp"] > 0,
    }
    decision = (
        "ADMIT_S2_CLUSTER_EVIDENCE_E0"
        if all(checks.values())
        else "STOP_S2_CLUSTER_GRANULARITY_CEILING_LOW"
    )
    denominator = max(pixel_vs_building, 1e-12)
    payload = {
        "protocol": "SPI_S2_CLUSTER_ORACLE_E0_AUDIT_v1",
        "configuration": vars(args),
        "cache": str(Path(args.calibration_cache).resolve()),
        "cache_sha256": _sha256(args.calibration_cache),
        "reports": reports,
        "event_reports": events,
        "counts": cache["counts"],
        "transition_counts": cache["transition_counts"],
        "gains_miou_pp": {
            "cluster_minus_building": cluster_vs_building,
            "cluster_minus_fixed_s2": cluster_vs_s2,
            "pixel_minus_building": pixel_vs_building,
        },
        "additional_pixel_oracle_space_recovered": cluster_vs_building / denominator,
        "cluster_nonnegative_events": nonnegative_events,
        "event_total": len(events),
        "bootstrap_cluster_minus_building": bootstrap,
        "checks": checks,
        "decision": decision,
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=" * 104)
    for name, report in reports.items():
        print(f"{name:18s} mIoU={100*report['valid_miou']:.4f} D-F1={100*report['damaged_f1']:.4f}")
    print(f"Cluster Oracle gain vs Building Oracle: {cluster_vs_building:+.4f}pp")
    print(f"Cluster Oracle gain vs fixed S2       : {cluster_vs_s2:+.4f}pp")
    print(f"Remaining Pixel-vs-Building space     : {pixel_vs_building:+.4f}pp")
    print(f"Nonnegative events                    : {nonnegative_events}/{len(events)}")
    print(f"Bootstrap                             : {bootstrap}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

