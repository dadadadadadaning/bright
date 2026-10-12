from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import sha256
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import PROTOCOL, REPLAY_PROTOCOL
from .core import (
    average_precision, direction_name, exact_keep_utilities, finite, quantiles,
    safe_ratio, sign_agreement, weighted_binary_auroc,
)


COVERAGE_GRID = (0.0, .005, .01, .02, .05, .10, .20, .50)


def _combined(train: dict, calibration: dict, replay: dict) -> dict:
    train_images = len(train["image_names"])
    return {
        "delta_hist": torch.cat((train["delta_hist"], calibration["delta_hist"])),
        "direction_code": torch.cat((train["direction_code"], calibration["direction_code"])),
        "cluster_area": torch.cat((train["cluster_area"], calibration["cluster_area"])),
        "image_index": torch.cat((
            train["image_index"].long(),
            calibration["image_index"].long() + train_images,
        )),
        "image_names": list(train["image_names"]) + list(calibration["image_names"]),
        "image_events": list(train["image_events"]) + list(calibration["image_events"]),
        "p3_histograms": torch.cat((train["p3_histograms"], calibration["p3_histograms"])),
        "fixed_s2_histograms": torch.cat((train["fixed_s2_histograms"], calibration["fixed_s2_histograms"])),
        **{
            name: torch.cat((replay["rows"]["meta_train"][name], replay["rows"]["meta_calibration"][name]))
            for name in replay["rows"]["meta_train"]
        },
    }


def _cluster_events(data: dict) -> list[str]:
    return [data["image_events"][int(index)] for index in data["image_index"]]


def _selected_histograms(data: dict, accepted: torch.Tensor) -> torch.Tensor:
    rows = []
    for image_index in range(len(data["image_names"])):
        current = data["image_index"] == image_index
        rejected = current & ~accepted.bool()
        histogram = data["fixed_s2_histograms"][image_index].double()
        if bool(rejected.any()):
            histogram = histogram - data["delta_hist"][rejected].sum(0).double()
        rows.append(histogram)
    return torch.stack(rows)


def _metric_row(fixed: torch.Tensor, selected: torch.Tensor) -> dict:
    fixed_report = metrics_from_histogram(fixed.sum(0))
    selected_report = metrics_from_histogram(selected.sum(0))
    return {
        "miou": selected_report["valid_miou"],
        "damaged_f1": selected_report["damaged_f1"],
        "gain_vs_fixed_s2_miou_pp": 100.0 * (selected_report["valid_miou"] - fixed_report["valid_miou"]),
        "damaged_f1_change_pp": 100.0 * (selected_report["damaged_f1"] - fixed_report["damaged_f1"]),
    }


def _per_event_exact(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_events = _cluster_events(data)
    metric = torch.empty(len(cluster_events), dtype=torch.float64)
    damaged = torch.empty_like(metric)
    for event in sorted(set(data["image_events"])):
        image_mask = torch.tensor([value == event for value in data["image_events"]])
        cluster_mask = torch.tensor([value == event for value in cluster_events])
        values = exact_keep_utilities(
            data["fixed_s2_histograms"][image_mask].sum(0),
            data["delta_hist"][cluster_mask],
        )
        metric[cluster_mask], damaged[cluster_mask] = values
    return metric, damaged


def _event_rows(data: dict, accepted: torch.Tensor, exact_metric: torch.Tensor, exact_damaged: torch.Tensor) -> dict:
    selected = _selected_histograms(data, accepted)
    cluster_events = _cluster_events(data)
    rows = {}
    for event in sorted(set(data["image_events"])):
        image_mask = torch.tensor([value == event for value in data["image_events"]])
        cluster_mask = torch.tensor([value == event for value in cluster_events])
        fixed = data["fixed_s2_histograms"][image_mask]
        p3 = data["p3_histograms"][image_mask]
        row = _metric_row(fixed, selected[image_mask])
        p3_report = metrics_from_histogram(p3.sum(0))
        fixed_report = metrics_from_histogram(fixed.sum(0))
        oracle_accept = exact_metric[cluster_mask] >= 0
        local_data = {
            "image_names": [name for name, keep in zip(data["image_names"], image_mask) if keep],
            "image_index": data["image_index"][cluster_mask],
            "fixed_s2_histograms": fixed,
            "delta_hist": data["delta_hist"][cluster_mask],
        }
        # Remap global image ids to this event's contiguous local indices.
        global_ids = torch.nonzero(image_mask, as_tuple=False).flatten()
        remap = {int(value): index for index, value in enumerate(global_ids.tolist())}
        local_data["image_index"] = torch.tensor([remap[int(value)] for value in local_data["image_index"]])
        oracle_hist = _selected_histograms(local_data, oracle_accept)
        oracle = _metric_row(fixed, oracle_hist)
        headroom = oracle["gain_vs_fixed_s2_miou_pp"]
        row.update({
            "p3_miou": p3_report["valid_miou"],
            "fixed_s2_miou": fixed_report["valid_miou"],
            "fixed_s2_gain_vs_p3_pp": 100.0 * (fixed_report["valid_miou"] - p3_report["valid_miou"]),
            "exact_metric_oracle_gain_pp": headroom,
            "b1_oracle_recovery_ratio": safe_ratio(row["gain_vs_fixed_s2_miou_pp"], headroom),
            "clusters": int(cluster_mask.sum()),
            "changed_pixels": int(data["cluster_area"][cluster_mask].sum()),
            "accepted_cluster_fraction": float(accepted[cluster_mask].float().mean()),
            "accepted_pixel_fraction": float(
                data["cluster_area"][cluster_mask & accepted].sum()
                / data["cluster_area"][cluster_mask].sum().clamp_min(1)
            ),
            "metric_score_quantiles": quantiles(data["metric_score"][cluster_mask]),
            "benefit_auroc_exact": finite(weighted_binary_auroc(exact_metric[cluster_mask] > 0, data["metric_score"][cluster_mask])),
            "benefit_auprc_exact": finite(average_precision(exact_metric[cluster_mask] > 0, data["metric_score"][cluster_mask])),
            "benefit_utility_weighted_auroc": finite(weighted_binary_auroc(
                exact_metric[cluster_mask] > 0, data["metric_score"][cluster_mask], exact_metric[cluster_mask].abs()
            )),
            "harm_auroc_exact": finite(weighted_binary_auroc(exact_damaged[cluster_mask] < 0, data["harm_score"][cluster_mask])),
            "harm_utility_weighted_auroc": finite(weighted_binary_auroc(
                exact_damaged[cluster_mask] < 0, data["harm_score"][cluster_mask], exact_damaged[cluster_mask].abs()
            )),
        })
        rows[event] = row
    return rows


def _direction_rows(data: dict, accepted: torch.Tensor, exact_metric: torch.Tensor, exact_damaged: torch.Tensor) -> dict:
    rows = {}
    for code in sorted(torch.unique(data["direction_code"]).tolist()):
        mask = data["direction_code"] == int(code)
        rows[direction_name(code)] = {
            "clusters": int(mask.sum()),
            "pixels": int(data["cluster_area"][mask].sum()),
            "accepted_fraction": float(accepted[mask].float().mean()),
            "exact_metric_positive_fraction": float((exact_metric[mask] > 0).float().mean()),
            "exact_damaged_harm_fraction": float((exact_damaged[mask] < 0).float().mean()),
            "exact_metric_utility_sum_pp": float(exact_metric[mask].sum()),
            "exact_damaged_utility_sum_pp": float(exact_damaged[mask].sum()),
            "benefit_auroc_exact": finite(weighted_binary_auroc(exact_metric[mask] > 0, data["metric_score"][mask])),
            "benefit_utility_weighted_auroc": finite(weighted_binary_auroc(
                exact_metric[mask] > 0, data["metric_score"][mask], exact_metric[mask].abs()
            )),
        }
    return rows


def _coverage_rows(data: dict) -> dict:
    cluster_events = _cluster_events(data)
    rows = {}
    for fraction in COVERAGE_GRID:
        accepted = torch.ones(len(cluster_events), dtype=torch.bool)
        for event in sorted(set(cluster_events)):
            ids = torch.tensor([index for index, value in enumerate(cluster_events) if value == event])
            reject_count = int(math.floor(float(fraction) * len(ids)))
            if reject_count:
                ranked = ids[torch.argsort(data["harm_score"][ids], descending=True)]
                accepted[ranked[:reject_count]] = False
        report = _metric_row(data["fixed_s2_histograms"], _selected_histograms(data, accepted))
        report.update({
            "rejected_cluster_fraction": float((~accepted).float().mean()),
            "rejected_pixel_fraction": float(
                data["cluster_area"][~accepted].sum() / data["cluster_area"].sum().clamp_min(1)
            ),
        })
        rows[f"reject_top_{100*fraction:g}pct"] = report
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("B1 event-shift and exact utility audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--v21-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated direction-impact audit: {output}")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    replay = torch.load(args.replay, map_location="cpu", weights_only=False)
    previous = json.loads(Path(args.v21_json).read_text(encoding="utf-8"))
    if replay.get("protocol") != REPLAY_PROTOCOL:
        raise RuntimeError("unexpected replay artifact")
    data = _combined(train, calibration, replay)
    exact_metric, exact_damaged = _per_event_exact(data)
    accepted = data["accepted_b1"].bool()
    selected = _selected_histograms(data, accepted)
    reproduced = _metric_row(data["fixed_s2_histograms"], selected)
    expected = previous["arms"]["b1_benefit_only"]["gain_vs_fixed_s2_miou_pp"]
    reproduction_error = reproduced["gain_vs_fixed_s2_miou_pp"] - expected
    if abs(reproduction_error) > 1e-6:
        raise RuntimeError(f"B1 replay mismatch: {reproduction_error:+.8f}pp")

    exact_dual_accept = (exact_metric >= 0) & (exact_damaged >= 0)
    exact_dual = _metric_row(
        data["fixed_s2_histograms"], _selected_histograms(data, exact_dual_accept)
    )
    event_rows = _event_rows(data, accepted, exact_metric, exact_damaged)
    dual_event_rows = _event_rows(data, exact_dual_accept, exact_metric, exact_damaged)
    dual_nonnegative = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in dual_event_rows.values())
    checks = {
        "b1_reproduced": abs(reproduction_error) <= 1e-6,
        "exact_dual_oracle_gain_ge_0_30pp": exact_dual["gain_vs_fixed_s2_miou_pp"] >= .30,
        "exact_dual_oracle_damaged_not_below_fixed": exact_dual["damaged_f1_change_pp"] >= 0,
        "exact_dual_oracle_nonnegative_events_ge_10": dual_nonnegative >= 10,
        "both_exact_harm_classes_present": bool((exact_damaged < 0).any() and (exact_damaged >= 0).any()),
    }
    decision = (
        "ADMIT_DIRECTION_IMPACT_E2_DEVELOPMENT"
        if all(checks.values()) else "STOP_DIRECTION_IMPACT_UTILITY_CEILING_LOW"
    )
    payload = {
        "protocol": PROTOCOL,
        "configuration": vars(args),
        "b1_reproduction": {"expected_gain_pp": expected, "observed": reproduced, "error_pp": reproduction_error},
        "global_signal": {
            "linear_vs_exact_metric_sign_agreement": sign_agreement(data["linear_metric_utility"], exact_metric),
            "linear_vs_exact_damaged_sign_agreement": sign_agreement(data["linear_damaged_utility"], exact_damaged),
            "benefit_auroc_exact": finite(weighted_binary_auroc(exact_metric > 0, data["metric_score"])),
            "benefit_auprc_exact": finite(average_precision(exact_metric > 0, data["metric_score"])),
            "benefit_utility_weighted_auroc": finite(weighted_binary_auroc(exact_metric > 0, data["metric_score"], exact_metric.abs())),
            "harm_auroc_exact": finite(weighted_binary_auroc(exact_damaged < 0, data["harm_score"])),
            "harm_utility_weighted_auroc": finite(weighted_binary_auroc(exact_damaged < 0, data["harm_score"], exact_damaged.abs())),
        },
        "event_shift": event_rows,
        "direction_audit": _direction_rows(data, accepted, exact_metric, exact_damaged),
        "risk_coverage": _coverage_rows(data),
        "exact_dual_oracle": {
            **exact_dual,
            "accepted_fraction": float(exact_dual_accept.float().mean()),
            "nonnegative_events": dual_nonnegative,
            "events": dual_event_rows,
        },
        "checks": checks,
        "decision": decision,
        "provenance": {
            "train_cache_sha256": sha256(args.train_cache),
            "calibration_cache_sha256": sha256(args.calibration_cache),
            "replay_sha256": sha256(args.replay),
            "v21_json_sha256": sha256(args.v21_json),
        },
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"B1 reproduced gain: {reproduced['gain_vs_fixed_s2_miou_pp']:+.4f}pp")
    print(f"Linear/exact sign agreement: metric={payload['global_signal']['linear_vs_exact_metric_sign_agreement']:.4f} damaged={payload['global_signal']['linear_vs_exact_damaged_sign_agreement']:.4f}")
    print(f"Benefit AUC ordinary/impact: {payload['global_signal']['benefit_auroc_exact']:.4f}/{payload['global_signal']['benefit_utility_weighted_auroc']:.4f}")
    print(f"Harm AUC ordinary/impact: {payload['global_signal']['harm_auroc_exact']:.4f}/{payload['global_signal']['harm_utility_weighted_auroc']:.4f}")
    print(f"Exact dual oracle gain: {exact_dual['gain_vs_fixed_s2_miou_pp']:+.4f}pp D-F1={exact_dual['damaged_f1_change_pp']:+.4f}pp events={dual_nonnegative}/14")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

