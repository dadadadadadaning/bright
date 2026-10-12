from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1 import TARGET_PROTOCOL
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import (
    bootstrap, event_rows,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    metric_labels, metric_utilities, per_image_selective_histograms,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import (
    MetricUtilityVerifier, matrix, sha256,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PROTOCOL
from .core import (
    area_masks, class_f1_gradient, direction_name, linear_utility,
    mechanism_counts,
)


def _scores(cache: dict, checkpoint: dict, split: str) -> torch.Tensor:
    values = matrix(cache, tuple(checkpoint["groups"]), split)
    features = (
        (values - checkpoint["feature_mean"].float())
        / checkpoint["feature_std"].float()
    ).clamp(-8, 8)
    model = MetricUtilityVerifier(
        checkpoint["input_dim"], checkpoint["model_type"], checkpoint["hidden"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        scores = model(features).sigmoid()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError(f"{split} scores contain non-finite values")
    return scores


def _score_summary(
    cache: dict, scores: torch.Tensor, threshold: float,
    metric_gradient: torch.Tensor,
) -> dict:
    utility = metric_utilities(cache["delta_hist"], metric_gradient)
    labels = metric_labels(utility)
    valid = labels >= 0
    quantiles = torch.quantile(
        scores.float(), torch.tensor((0.05, 0.25, 0.5, 0.75, 0.95))
    )
    return {
        "clusters": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std(unbiased=False)),
        "q05_q25_q50_q75_q95": [float(value) for value in quantiles],
        "accepted_fraction": float((scores >= threshold).float().mean()),
        "metric_utility_auroc": binary_auroc(labels[valid], scores[valid]),
        "metric_positive": int((labels == 1).sum()),
        "metric_negative": int((labels == 0).sum()),
    }


def _numeric_differences(first: dict, second: dict) -> dict:
    result = {}
    for key in sorted(set(first) & set(second)):
        a, b = first[key], second[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            result[key] = float(b - a)
    return result


def _group_summary(
    cache: dict, mask: torch.Tensor, accept: torch.Tensor, scores: torch.Tensor,
    metric_utility: torch.Tensor, damaged_utility: torch.Tensor,
) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {"clusters": 0}
    utility = cache["utility"]
    local_accept = accept[mask]
    return {
        "clusters": count,
        "changed_pixels": int(utility["changed"][mask].long().sum()),
        "corrected_pixels": int(utility["corrected"][mask].long().sum()),
        "harmed_pixels": int(utility["harmed"][mask].long().sum()),
        "pixel_net": int(utility["net"][mask].long().sum()),
        "accept_fraction": float(local_accept.float().mean()),
        "score_mean": float(scores[mask].mean()),
        "metric_utility_mean_pp": float(metric_utility[mask].mean()),
        "damaged_utility_mean_pp": float(damaged_utility[mask].mean()),
        **mechanism_counts(local_accept, metric_utility[mask], damaged_utility[mask]),
    }


def _event_diagnostics(
    cache: dict, selected_by_image: torch.Tensor, scores: torch.Tensor,
    threshold: float, metric_utility: torch.Tensor,
    damaged_utility: torch.Tensor,
) -> dict:
    result = {}
    fixed_by_image = cache["fixed_s2_histograms"].double()
    image_index = cache["image_index"].long()
    for event in sorted(set(cache["image_events"])):
        image_ids = torch.tensor([
            index for index, value in enumerate(cache["image_events"]) if value == event
        ])
        cluster_mask = torch.zeros(len(scores), dtype=torch.bool)
        for index in image_ids.tolist():
            cluster_mask |= image_index == index
        fixed = metrics_from_histogram(fixed_by_image[image_ids].sum(0))
        selected = metrics_from_histogram(selected_by_image[image_ids].sum(0))
        labels = metric_labels(metric_utility[cluster_mask])
        valid = labels >= 0
        result[event] = {
            "images": len(image_ids),
            "clusters": int(cluster_mask.sum()),
            "accepted_fraction": float((scores[cluster_mask] >= threshold).float().mean())
            if bool(cluster_mask.any()) else 0.0,
            "gain_vs_fixed_s2_miou_pp": 100.0 * (
                selected["valid_miou"] - fixed["valid_miou"]
            ),
            "damaged_f1_change_pp": 100.0 * (
                selected["damaged_f1"] - fixed["damaged_f1"]
            ),
            "metric_utility_auroc": binary_auroc(
                labels[valid], scores[cluster_mask][valid]
            ) if int(valid.sum()) > 1 and int(labels[valid].unique().numel()) == 2 else None,
            **mechanism_counts(
                scores[cluster_mask] >= threshold,
                metric_utility[cluster_mask], damaged_utility[cluster_mask],
            ),
        }
    return result


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Read-only MACU-E1 failure mechanism audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--confirm-cache", required=True)
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--confirm-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated D0 audit: {output}")
    caches = {
        "meta_train": torch.load(args.train_cache, map_location="cpu", weights_only=False),
        "meta_calibration": torch.load(args.calibration_cache, map_location="cpu", weights_only=False),
        "meta_confirm": torch.load(args.confirm_cache, map_location="cpu", weights_only=False),
    }
    targets = torch.load(args.train_targets, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.verifier_checkpoint, map_location="cpu", weights_only=False)
    confirm_result = json.loads(Path(args.confirm_result).read_text(encoding="utf-8"))

    for split, cache in caches.items():
        if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
            raise RuntimeError(f"unexpected {split} D1 cache")
        if cache.get("val450_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError(f"{split} cache accessed a forbidden split")
    if not caches["meta_confirm"].get("meta_confirm_accessed", False):
        raise RuntimeError("this diagnostic requires the already-consumed confirm cache")
    if confirm_result.get("decision") != "STOP_MACU_E1_AFTER_META_CONFIRM":
        raise RuntimeError("D0 is only valid after the frozen MACU confirm failure")
    if checkpoint.get("protocol") != "SPI_METRIC_ALIGNED_CLUSTER_VERIFIER_E1_v1":
        raise RuntimeError("unexpected verifier checkpoint")
    if targets.get("protocol") != TARGET_PROTOCOL or targets.get("split") != "meta_train":
        raise RuntimeError("unexpected train-frozen target cache")
    if targets["source_cache_sha256"] != sha256(args.train_cache):
        raise RuntimeError("target/train cache fingerprint mismatch")
    expected_verifier_sha = confirm_result.get("provenance", {}).get("verifier_sha256")
    if expected_verifier_sha != sha256(args.verifier_checkpoint):
        raise RuntimeError("confirm result and verifier fingerprint mismatch")
    if confirm_result.get("val450_accessed", True) or confirm_result.get("test_accessed", True):
        raise RuntimeError("confirm result provenance is contaminated")

    metric_gradient = targets["train_frozen_miou_gradient"].double()
    damaged_gradient = class_f1_gradient(
        caches["meta_train"]["fixed_s2_histograms"].sum(0), 2
    )
    threshold = float(checkpoint["threshold"])
    split_scores = {
        split: _scores(cache, checkpoint, split) for split, cache in caches.items()
    }
    score_drift = {
        split: _score_summary(cache, split_scores[split], threshold, metric_gradient)
        for split, cache in caches.items()
    }

    confirm = caches["meta_confirm"]
    scores = split_scores["meta_confirm"]
    accept = scores >= threshold
    metric_utility = metric_utilities(confirm["delta_hist"], metric_gradient)
    damaged_utility = linear_utility(confirm["delta_hist"], damaged_gradient)
    selected_by_image = per_image_selective_histograms(confirm, accept)
    fixed = metrics_from_histogram(confirm["fixed_s2_histograms"].sum(0))
    selected = metrics_from_histogram(selected_by_image.sum(0))

    dual_accept = (metric_utility > 0) & (damaged_utility >= 0)
    dual_by_image = per_image_selective_histograms(confirm, dual_accept)
    dual = metrics_from_histogram(dual_by_image.sum(0))
    dual_gain_pp = 100.0 * (dual["valid_miou"] - fixed["valid_miou"])
    dual_events = event_rows(confirm, dual_by_image)
    dual_nonnegative = sum(
        row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in dual_events.values()
    )
    dual_bootstrap = bootstrap(
        confirm, dual_by_image, args.bootstrap_iterations, args.seed
    )
    dual_checks = {
        "gain_vs_fixed_s2_ge_0_30pp": dual_gain_pp >= 0.30,
        "damaged_f1_not_below_fixed_s2": dual["damaged_f1"] >= fixed["damaged_f1"],
        "nonnegative_events_ge_10": dual_nonnegative >= min(10, len(dual_events)),
        "bootstrap_ci_low_gt_zero": dual_bootstrap["ci95_low_pp"] > 0,
        "both_damaged_risk_classes_ge_100": int((damaged_utility >= 0).sum()) >= 100
        and int((damaged_utility < 0).sum()) >= 100,
    }

    directions = {}
    for code in sorted(confirm["direction_code"].long().unique().tolist()):
        mask = confirm["direction_code"].long() == code
        directions[direction_name(code)] = _group_summary(
            confirm, mask, accept, scores, metric_utility, damaged_utility
        )
    areas = {
        name: _group_summary(
            confirm, mask, accept, scores, metric_utility, damaged_utility
        )
        for name, mask in area_masks(confirm["cluster_area"]).items()
    }
    mechanism = mechanism_counts(accept, metric_utility, damaged_utility)
    if mechanism["accepted_damaged_harm"] > mechanism["rejected_damaged_benefit"]:
        dominant = "ACCEPTED_DAMAGED_HARM"
    elif mechanism["rejected_damaged_benefit"] > mechanism["accepted_damaged_harm"]:
        dominant = "REJECTED_DAMAGED_BENEFIT"
    else:
        dominant = "BALANCED_DAMAGED_ERRORS"
    decision = (
        "ADMIT_CS_MACU_V2_DEVELOPMENT_ONLY"
        if all(dual_checks.values())
        else "STOP_CLASS_SAFE_MACU_CEILING_OR_STABILITY_LOW"
    )
    payload = {
        "protocol": PROTOCOL,
        "configuration": vars(args),
        "provenance": {
            "train_cache_sha256": sha256(args.train_cache),
            "calibration_cache_sha256": sha256(args.calibration_cache),
            "confirm_cache_sha256": sha256(args.confirm_cache),
            "train_targets_sha256": sha256(args.train_targets),
            "verifier_sha256": sha256(args.verifier_checkpoint),
            "confirm_result_sha256": sha256(args.confirm_result),
        },
        "warning": "Post-confirm diagnostic only; it must not be reported as an independent confirmation of a revised method.",
        "threshold": threshold,
        "fixed_s2": fixed,
        "macu": selected,
        "macu_metric_differences": _numeric_differences(fixed, selected),
        "score_drift": score_drift,
        "mechanism_counts": mechanism,
        "dominant_damaged_failure": dominant,
        "direction_diagnostics": directions,
        "area_diagnostics": areas,
        "event_diagnostics": _event_diagnostics(
            confirm, selected_by_image, scores, threshold,
            metric_utility, damaged_utility,
        ),
        "class_safe_metric_oracle": {
            "accepted_clusters": int(dual_accept.sum()),
            "rejected_clusters": int((~dual_accept).sum()),
            "report": dual,
            "gain_vs_fixed_s2_miou_pp": dual_gain_pp,
            "damaged_f1_change_pp": 100.0 * (
                dual["damaged_f1"] - fixed["damaged_f1"]
            ),
            "event_nonnegative": dual_nonnegative,
            "event_total": len(dual_events),
            "events": dual_events,
            "bootstrap": dual_bootstrap,
            "checks": dual_checks,
        },
        "decision": decision,
        "meta_confirm_already_consumed_for_diagnosis": True,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"MACU-D0 dominant damaged failure: {dominant}")
    print(f"MACU-D0 current mIoU gain: {100.0 * (selected['valid_miou'] - fixed['valid_miou']):+.4f}pp")
    print(f"MACU-D0 current Damaged F1 change: {100.0 * (selected['damaged_f1'] - fixed['damaged_f1']):+.4f}pp")
    print(f"CS Oracle mIoU gain: {dual_gain_pp:+.4f}pp")
    print(f"CS Oracle Damaged F1 change: {100.0 * (dual['damaged_f1'] - fixed['damaged_f1']):+.4f}pp")
    print(f"CS Oracle events: {dual_nonnegative}/{len(dual_events)}")
    print(f"CS Oracle bootstrap: {dual_bootstrap}")
    print(f"Checks: {dual_checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Read cached consumed meta_confirm only. Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
