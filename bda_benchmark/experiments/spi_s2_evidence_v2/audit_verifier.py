from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from .audit_evidence import average_precision
from .verifier import SingleEvidenceVerifier, standardized_inputs


PROTOCOL = "SPI_S2_SINGLE_EVIDENCE_VERIFIER_INTERNAL_CALIBRATION_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjusted_image_histograms(cache: dict, accept: torch.Tensor) -> torch.Tensor:
    histograms = cache["base_histograms"].clone().long()
    image_index = cache["image_index"].long()
    target = cache["target_class"].long()
    base = cache["base_class"].long()
    candidate = cache["candidate_class"].long()
    selected = torch.nonzero(accept, as_tuple=False).flatten()
    if selected.numel():
        before_code = image_index[selected] * 16 + target[selected] * 4 + base[selected]
        after_code = image_index[selected] * 16 + target[selected] * 4 + candidate[selected]
        size = histograms.shape[0] * 16
        delta = torch.bincount(after_code, minlength=size) - torch.bincount(
            before_code, minlength=size
        )
        histograms += delta.reshape_as(histograms)
    if int(histograms.min()) < 0:
        raise RuntimeError("histogram reconstruction produced a negative count")
    return histograms


def bootstrap(image_histograms, candidate_name, reference_name, events, iterations, seed):
    grouped = defaultdict(list)
    for index, event in enumerate(events):
        grouped[event].append(index)
    rng = np.random.default_rng(seed)
    values = np.zeros(iterations, dtype=np.float64)

    def miou(hist):
        intersection = np.diag(hist)
        union = hist.sum(0) + hist.sum(1) - intersection
        return float((intersection / np.maximum(union, 1.0))[1:4].mean())

    for iteration in range(iterations):
        candidate_total = np.zeros((4, 4), dtype=np.float64)
        reference_total = np.zeros((4, 4), dtype=np.float64)
        for indices in grouped.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            candidate_total += image_histograms[candidate_name][sampled].sum(0).numpy()
            reference_total += image_histograms[reference_name][sampled].sum(0).numpy()
        values[iteration] = 100.0 * (miou(candidate_total) - miou(reference_total))
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": iterations,
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Audit one admitted Evidence Verifier V2")
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-steps", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)

    cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    if cache.get("split") != "internal_calibration":
        raise RuntimeError("verifier audit requires internal calibration cache")
    if cache.get("validation_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError("calibration cache accessed a forbidden split")
    saved = torch.load(args.verifier_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("protocol") != "SPI_S2_SINGLE_EVIDENCE_VERIFIER_TRAIN_ONLY_v2":
        raise RuntimeError("unexpected verifier checkpoint protocol")
    if int(saved.get("optimizer_step", 0)) != args.expected_steps:
        raise RuntimeError("verifier optimizer-step mismatch")
    threshold_record = saved.get("threshold_selection")
    if not threshold_record:
        raise RuntimeError("verifier lacks a Train-selected threshold")
    group = str(saved["group"])
    device = torch.device(args.device)
    inputs = standardized_inputs(cache, group)
    model = SingleEvidenceVerifier(int(saved["input_channels"]), int(saved["hidden"])).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval().requires_grad_(False)
    mean = saved["normalizer_mean"].float().to(device)
    std = saved["normalizer_std"].float().to(device)
    scores = []
    for start in range(0, inputs.shape[0], 65536):
        batch = inputs[start:start + 65536].to(device)
        scores.append(torch.sigmoid(model((batch - mean) / std)).cpu())
    scores = torch.cat(scores)
    accept = scores >= float(threshold_record["threshold"])
    gate_histograms = adjusted_image_histograms(cache, accept)
    image_histograms = {
        "p3": cache["base_histograms"].long(),
        "fixed_s2": cache["fixed_histograms"].long(),
        "verifier": gate_histograms,
    }
    reports = {
        name: metrics_from_histogram(rows.sum(0)) for name, rows in image_histograms.items()
    }
    events = cache["image_events"]
    event_reports = {}
    for event in sorted(set(events)):
        indices = [index for index, value in enumerate(events) if value == event]
        event_reports[event] = {
            name: metrics_from_histogram(rows[indices].sum(0))
            for name, rows in image_histograms.items()
        }
    labels = cache["selector_label"].long()
    decisive = labels >= 0
    selector = {
        "auroc": binary_auroc(labels[decisive], scores[decisive]),
        "average_precision": average_precision(labels[decisive], scores[decisive]),
        "count": int(decisive.sum()),
    }
    changes = {
        "accepted": int(accept.sum()),
        "corrected": int((accept & (labels == 1)).sum()),
        "harmed": int((accept & (labels == 0)).sum()),
        "neutral": int((accept & (labels < 0)).sum()),
    }
    fixed_harmed = int((labels == 0).sum())
    gain_miou = 100.0 * (reports["verifier"]["valid_miou"] - reports["fixed_s2"]["valid_miou"])
    foreground_deltas = {
        key: 100.0 * (reports["verifier"][key] - reports["fixed_s2"][key])
        for key in ("intact_iou", "damaged_iou", "destroyed_iou")
    }
    event_nonnegative = sum(
        rows["verifier"]["valid_miou"] >= rows["fixed_s2"]["valid_miou"]
        for rows in event_reports.values()
    )
    paired = bootstrap(
        image_histograms,
        "verifier",
        "fixed_s2",
        events,
        args.bootstrap_iterations,
        args.seed,
    )
    checks = {
        "beats_fixed_s2_ge_0_10pp": gain_miou >= 0.10,
        "harm_reduction_ge_10pct": changes["harmed"] <= 0.90 * max(fixed_harmed, 1),
        "selector_auroc_ge_0_60": selector["auroc"] >= 0.60,
        "event_nonnegative_ge_60pct": event_nonnegative >= math.ceil(0.60 * len(event_reports)),
        "no_foreground_drop_gt_0_50pp": min(foreground_deltas.values()) >= -0.50,
        "bootstrap_ci_low_gt_zero": paired["ci95_low_pp"] > 0.0,
    }
    decision = "FREEZE_METHOD_THEN_ALLOW_ONE_VAL450_RUN" if all(checks.values()) else "STOP_EVIDENCE_VERIFIER_AFTER_INTERNAL_CALIBRATION"
    payload = {
        "protocol": PROTOCOL,
        "group": group,
        "threshold": threshold_record,
        "reports": reports,
        "event_reports": event_reports,
        "selector": selector,
        "changes": changes,
        "gain_vs_fixed_s2_miou_pp": gain_miou,
        "foreground_iou_deltas_pp": foreground_deltas,
        "event_nonnegative": event_nonnegative,
        "event_total": len(event_reports),
        "bootstrap": paired,
        "checks": checks,
        "decision": decision,
        "calibration_cache": str(Path(args.calibration_cache).resolve()),
        "calibration_cache_sha256": sha256_file(args.calibration_cache),
        "verifier_checkpoint": str(Path(args.verifier_checkpoint).resolve()),
        "verifier_checkpoint_sha256": sha256_file(args.verifier_checkpoint),
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 116)
    for name, report in reports.items():
        print(f"{name:10s} mIoU={100*report['valid_miou']:.4f} D-F1={100*report['damaged_f1']:.4f}")
    print(f"Selector: {selector}")
    print(f"Changes: {changes}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
