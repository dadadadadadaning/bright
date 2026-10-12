from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.audit_e0 import (
    policy_scores, utility_labels, weighted_auroc,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PROTOCOL
from .core import area_bin_masks, duplicate_fraction, reconstruct_oracle_histograms


def _report_group(cache: dict, mask: torch.Tensor, fixed_hist: torch.Tensor, full_gain: float) -> dict:
    utility = cache["utility"]
    net = utility["net"].long()
    rejected = mask & (net <= 0)
    selective = fixed_hist - cache["delta_hist"][rejected].sum(0).double()
    fixed_report = metrics_from_histogram(fixed_hist)
    selective_report = metrics_from_histogram(selective)
    gain = 100.0 * (selective_report["valid_miou"] - fixed_report["valid_miou"])
    return {
        "clusters": int(mask.sum()),
        "fraction": float(mask.float().mean()),
        "changed": int(utility["changed"][mask].sum()),
        "corrected": int(utility["corrected"][mask].sum()),
        "harmed": int(utility["harmed"][mask].sum()),
        "net": int(utility["net"][mask].sum()),
        "oracle_rejected": int(rejected.sum()),
        "oracle_gain_vs_fixed_s2_miou_pp": gain,
        "oracle_recovery_ratio": gain / full_gain if full_gain > 1e-12 else float("nan"),
    }


def _heldout_scores(cache: dict, evidence_json: dict, group: str) -> torch.Tensor:
    policies = evidence_json["results"][group]["direction_policy_by_heldout_event"]
    events = [cache["image_events"][int(index)] for index in cache["image_index"].long()]
    output = torch.empty(len(events))
    for event in sorted(set(events)):
        mask = torch.tensor([value == event for value in events])
        output[mask] = policy_scores(cache, group, policies[event])[mask]
    return output


def _auc_rows(cache: dict, scores: torch.Tensor, masks: dict[str, torch.Tensor], margin: float) -> dict:
    labels = utility_labels(cache, margin)
    changed = cache["utility"]["changed"].float()
    rows = {}
    for name, group in masks.items():
        valid = group & (labels >= 0)
        target = labels[valid]
        if int((target == 0).sum()) == 0 or int((target == 1).sum()) == 0:
            continue
        rows[name] = {
            "auroc": binary_auroc(target, scores[valid]),
            "pixel_weighted_auroc": weighted_auroc(target, scores[valid], changed[valid]),
            "positive": int((target == 1).sum()), "negative": int((target == 0).sum()),
        }
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Cluster Detail D0 concentration/resolution audit")
    parser.add_argument("--cluster-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--cluster-evidence-json", required=True)
    parser.add_argument("--cluster-oracle-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--single-stage3-threshold", type=float, default=0.70)
    parser.add_argument("--at-most-two-stage3-threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    cache = torch.load(args.cluster_cache, map_location="cpu", weights_only=False)
    geometry = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    evidence = json.loads(Path(args.cluster_evidence_json).read_text())
    oracle = json.loads(Path(args.cluster_oracle_json).read_text())
    if cache.get("protocol") != "SPI_S2_CLUSTER_EVIDENCE_CACHE_v2" or cache.get("split") != "meta_calibration":
        raise RuntimeError("D0 requires the complete Cluster-E0 v2 calibration cache")
    for name, payload in (("cluster", cache), ("geometry", geometry), ("evidence", evidence)):
        if payload.get("meta_confirm_accessed", True) or payload.get("val450_accessed", True) or payload.get("test_accessed", True):
            raise RuntimeError(f"{name} input violated the D0 data boundary")
    if geometry.get("protocol") != PROTOCOL or geometry.get("split") != "meta_calibration":
        raise RuntimeError("unexpected D0 geometry cache")
    if evidence.get("decision") != "STOP_CLUSTER_EVIDENCE_NO_CROSS_EVENT_SIGNAL":
        raise RuntimeError("D0 is only registered after coarse Cluster-E0 stops")
    if oracle.get("decision") != "ADMIT_S2_CLUSTER_EVIDENCE_E0":
        raise RuntimeError("Cluster Oracle prerequisite did not pass")
    if cache["p3_sha256"] != geometry["p3_sha256"] or len(cache["cluster_area"]) != len(geometry["rows"]["area"]):
        raise RuntimeError("cache provenance/alignment mismatch")

    p3_hist, fixed_hist, cluster_hist = reconstruct_oracle_histograms(cache)
    p3_report = metrics_from_histogram(p3_hist)
    fixed_report = metrics_from_histogram(fixed_hist)
    cluster_report = metrics_from_histogram(cluster_hist)
    full_gain = 100.0 * (cluster_report["valid_miou"] - fixed_report["valid_miou"])
    areas = cache["cluster_area"].long()
    bins = area_bin_masks(areas)
    area_rows = {name: _report_group(cache, mask, fixed_hist, full_gain) for name, mask in bins.items()}
    cumulative = {}
    for threshold in (1, 2, 4, 8, 16, 32):
        mask = areas >= threshold
        cumulative[f"area_ge_{threshold}"] = _report_group(cache, mask, fixed_hist, full_gain)
    directions = {}
    for code in sorted(torch.unique(cache["direction_code"].long()).tolist()):
        before, after = divmod(code, 4)
        directions[f"{before}_to_{after}"] = _report_group(
            cache, cache["direction_code"].long() == code, fixed_hist, full_gain
        )

    rgb_scores = _heldout_scores(cache, evidence, "rgb_role")
    rgb_auc_by_area = _auc_rows(cache, rgb_scores, bins, args.utility_margin)
    token_rows = geometry["rows"]
    resolution = {}
    for stage in ("stage2", "stage3", "stage4"):
        tokens = token_rows[f"{stage}_tokens"].long()
        resolution[stage] = {
            "mean_tokens": float(tokens.float().mean()),
            "median_tokens": float(tokens.float().median()),
            "single_token_fraction": float((tokens <= 1).float().mean()),
            "at_most_two_tokens_fraction": float((tokens <= 2).float().mean()),
            "at_most_four_tokens_fraction": float((tokens <= 4).float().mean()),
        }
    building_keys = cache["image_index"].long() * 1000000 + cache["region_label"].long()
    role_features = torch.cat((cache["features"]["rgb_role"], cache["features"]["sar_role"]), 1)
    collisions = {
        "stage3_rgb_sar_role_duplicate_fraction_q1e3": duplicate_fraction(
            building_keys, role_features, 3
        )
    }
    for group in ("rgb_role", "sar_role", "rgb_role_stage4", "sar_role_stage4"):
        names = tuple(cache["feature_names"][group])
        key = next((i for i, name in enumerate(names) if "change_anchor_cosine_distance" in name), None)
        if key is not None:
            collisions[f"{group}_change_anchor_cosine_distance_mean"] = float(
                cache["features"][group][:, key].float().mean()
            )

    stage3_mismatch = (
        resolution["stage3"]["single_token_fraction"] >= args.single_stage3_threshold
        or resolution["stage3"]["at_most_two_tokens_fraction"] >= args.at_most_two_stage3_threshold
    )
    useful_large = max(
        cumulative["area_ge_8"]["oracle_recovery_ratio"],
        cumulative["area_ge_16"]["oracle_recovery_ratio"],
    )
    checks = {
        "cluster_oracle_gain_positive": full_gain >= 0.10,
        "stage3_resolution_mismatch": stage3_mismatch,
        "stage2_has_more_support_than_stage3": resolution["stage2"]["mean_tokens"] > resolution["stage3"]["mean_tokens"],
        "large_cluster_oracle_recovery_reported": useful_large == useful_large,
    }
    decision = (
        "ADMIT_CLUSTER_DETAIL_D1_STAGE2_EDGE_EVIDENCE"
        if checks["cluster_oracle_gain_positive"] and checks["stage3_resolution_mismatch"]
        else "STOP_CLUSTER_DETAIL_NO_SCALE_MISMATCH"
    )
    payload = {
        "protocol": "SPI_CLUSTER_DETAIL_D0_AUDIT_v1", "configuration": vars(args),
        "metrics": {"p3": p3_report, "fixed_s2": fixed_report, "cluster_oracle": cluster_report},
        "cluster_oracle_gain_vs_fixed_s2_miou_pp": full_gain,
        "area_bins": area_rows, "area_cumulative": cumulative,
        "directions": directions, "rgb_auc_by_area": rgb_auc_by_area,
        "resolution": resolution, "feature_collisions": collisions,
        "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "cluster_oracle_gain_pp": full_gain, "resolution": resolution,
        "area_ge_8": cumulative["area_ge_8"], "area_ge_16": cumulative["area_ge_16"],
        "checks": checks, "decision": decision,
    }, indent=2), flush=True)
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
