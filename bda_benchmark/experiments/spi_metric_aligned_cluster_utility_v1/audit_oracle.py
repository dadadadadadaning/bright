from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import AUDIT_PROTOCOL, TARGET_PROTOCOL
from .core import per_image_selective_histograms, selective_histogram


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_rows(cache: dict, selected: torch.Tensor) -> dict:
    rows = {}
    fixed = cache["fixed_s2_histograms"].double()
    for event in sorted(set(cache["image_events"])):
        ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        a = metrics_from_histogram(selected[ids].sum(0))
        b = metrics_from_histogram(fixed[ids].sum(0))
        rows[event] = {"gain_vs_fixed_s2_miou_pp": 100.0 * (a["valid_miou"] - b["valid_miou"])}
    return rows


def bootstrap(cache: dict, selected: torch.Tensor, iterations: int, seed: int) -> dict:
    fixed = cache["fixed_s2_histograms"].double()
    generator = torch.Generator().manual_seed(seed)
    gains = []
    for _ in range(iterations):
        ids = torch.randint(len(fixed), (len(fixed),), generator=generator)
        a = metrics_from_histogram(selected[ids].sum(0))["valid_miou"]
        b = metrics_from_histogram(fixed[ids].sum(0))["valid_miou"]
        gains.append(100.0 * (a - b))
    return {"mean_pp": float(np.mean(gains)), "ci95_low_pp": float(np.quantile(gains, .025)),
            "ci95_high_pp": float(np.quantile(gains, .975)), "iterations": iterations}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Metric-aligned cluster Oracle audit")
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--calibration-targets", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-oracle-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    targets = torch.load(args.calibration_targets, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(args.target_manifest).read_text(encoding="utf-8"))
    if targets.get("protocol") != TARGET_PROTOCOL or targets.get("split") != "meta_calibration":
        raise RuntimeError("unexpected metric target cache")
    if targets["source_cache_sha256"] != sha256(args.calibration_cache):
        raise RuntimeError("metric targets/calibration cache fingerprint mismatch")
    if manifest.get("protocol") != TARGET_PROTOCOL:
        raise RuntimeError("unexpected target manifest")
    fixed_hist = cache["fixed_s2_histograms"].sum(0)
    p3 = metrics_from_histogram(cache["p3_histograms"].sum(0))
    fixed = metrics_from_histogram(fixed_hist)
    pixel_accept = cache["utility"]["net"].long() > 0
    metric_accept = targets["metric_utility_pp"].double() > 0
    exact_accept = targets["exact_marginal_utility_pp"].double() > 0
    variants = {}
    for name, accept in (
        ("pixel_net_oracle", pixel_accept),
        ("metric_linear_oracle", metric_accept),
        ("metric_exact_oracle", exact_accept),
    ):
        by_image = per_image_selective_histograms(cache, accept)
        report = metrics_from_histogram(by_image.sum(0))
        events = event_rows(cache, by_image)
        variants[name] = {
            "report": report,
            "gain_vs_fixed_s2_miou_pp": 100.0 * (report["valid_miou"] - fixed["valid_miou"]),
            "accepted_clusters": int(accept.sum()), "rejected_clusters": int((~accept).sum()),
            "event_nonnegative": sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in events.values()),
            "event_total": len(events), "events": events,
            "bootstrap": bootstrap(cache, by_image, args.bootstrap_iterations, args.seed + len(variants)),
        }
    metric = variants["metric_linear_oracle"]
    checks = {
        "metric_oracle_gain_ge_threshold": metric["gain_vs_fixed_s2_miou_pp"] >= args.minimum_oracle_gain_pp,
        "metric_oracle_nonnegative_events_ge_threshold": metric["event_nonnegative"] >= min(args.minimum_nonnegative_events, metric["event_total"]),
        "metric_oracle_bootstrap_ci_low_gt_zero": metric["bootstrap"]["ci95_low_pp"] > 0,
        "linear_exact_sign_agreement_ge_0_95": manifest["summaries"]["meta_calibration"]["linear_exact_sign_agreement"] >= .95,
        "both_classes_present": int(metric_accept.sum()) >= 100 and int((~metric_accept).sum()) >= 100,
    }
    decision = "ADMIT_METRIC_ALIGNED_LINEAR_VERIFIER_E1" if all(checks.values()) else "STOP_METRIC_ALIGNED_CLUSTER_ORACLE_LOW"
    payload = {"protocol": AUDIT_PROTOCOL, "configuration": vars(args), "p3": p3,
               "fixed_s2": fixed, "variants": variants, "checks": checks,
               "decision": decision, "meta_confirm_accessed": False,
               "val450_accessed": False, "official_instances_accessed": False,
               "test_accessed": False}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, row in variants.items():
        print(f"MACU-M0 {name:22s} gain={row['gain_vs_fixed_s2_miou_pp']:+.4f}pp events={row['event_nonnegative']}/{row['event_total']} CI=[{row['bootstrap']['ci95_low_pp']:+.4f},{row['bootstrap']['ci95_high_pp']:+.4f}]", flush=True)
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
