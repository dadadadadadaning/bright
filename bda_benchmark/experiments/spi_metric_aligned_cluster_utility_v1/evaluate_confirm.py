from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import TARGET_PROTOCOL
from .audit_oracle import bootstrap, event_rows
from .core import metric_utilities, per_image_selective_histograms
from .train_verifier import MetricUtilityVerifier, matrix, sha256


FROZEN_CONFIGURATION_KEYS = (
    "minimum_building_area",
    "minimum_cluster_area",
    "feature_channels",
    "matching_window",
    "halo_radius",
    "seed",
)


def confirmation_checks(
    *, gain_pp: float, oracle_recovery: float, ci_low_pp: float,
    damaged_f1: float, fixed_damaged_f1: float, nonnegative_events: int,
    event_total: int, minimum_gain_pp: float = 0.30,
    minimum_oracle_recovery: float = 0.25,
    minimum_nonnegative_events: int = 10,
) -> dict[str, bool]:
    return {
        "gain_vs_fixed_s2_ge_threshold": gain_pp >= minimum_gain_pp,
        "oracle_recovery_ge_threshold": oracle_recovery >= minimum_oracle_recovery,
        "bootstrap_ci_low_gt_zero": ci_low_pp > 0,
        "damaged_f1_not_below_fixed_s2": damaged_f1 >= fixed_damaged_f1,
        "nonnegative_events_ge_threshold": nonnegative_events
        >= min(minimum_nonnegative_events, event_total),
    }


def _validate_feature_contract(train: dict, confirm: dict, groups: tuple[str, ...]) -> None:
    train_configuration = train.get("configuration", {})
    confirm_configuration = confirm.get("configuration", {})
    differences = {
        key: (train_configuration.get(key), confirm_configuration.get(key))
        for key in FROZEN_CONFIGURATION_KEYS
        if train_configuration.get(key) != confirm_configuration.get(key)
    }
    if differences:
        raise RuntimeError(f"meta_confirm D1 settings differ from training: {differences}")
    for group in groups:
        if group not in train["features"] or group not in confirm["features"]:
            raise RuntimeError(f"missing frozen feature group: {group}")
        if train["feature_names"][group] != confirm["feature_names"][group]:
            raise RuntimeError(f"feature schema differs for group {group}")
        if train["features"][group].shape[1] != confirm["features"][group].shape[1]:
            raise RuntimeError(f"feature dimension differs for group {group}")


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("One-time frozen MACU meta_confirm evaluation")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--confirm-cache", required=True)
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--calibration-screen", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-oracle-recovery", type=float, default=0.25)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated meta_confirm evaluation: {output}")
    screen = json.loads(Path(args.calibration_screen).read_text(encoding="utf-8"))
    if screen.get("decision") != "FREEZE_MACU_E1_FOR_META_CONFIRM":
        raise RuntimeError("confirm refused: MACU-E1 did not pass calibration screening")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    confirm = torch.load(args.confirm_cache, map_location="cpu", weights_only=False)
    targets = torch.load(args.train_targets, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.verifier_checkpoint, map_location="cpu", weights_only=False)

    if train.get("protocol") != D1_PROTOCOL or train.get("split") != "meta_train":
        raise RuntimeError("unexpected D1 meta_train cache")
    if confirm.get("protocol") != D1_PROTOCOL or confirm.get("split") != "meta_confirm":
        raise RuntimeError("unexpected D1 meta_confirm cache")
    if not confirm.get("meta_confirm_accessed", False):
        raise RuntimeError("meta_confirm cache provenance is incomplete")
    for name, payload in (("train", train), ("confirm", confirm)):
        if payload.get("val450_accessed", True) or payload.get("official_instances_accessed", True) or payload.get("test_accessed", True):
            raise RuntimeError(f"{name} cache accessed a forbidden split")
    if set(train["image_names"]) & set(confirm["image_names"]):
        raise RuntimeError("meta_train/meta_confirm image overlap")
    if checkpoint.get("protocol") != "SPI_METRIC_ALIGNED_CLUSTER_VERIFIER_E1_v1":
        raise RuntimeError("unexpected MACU verifier protocol")
    if checkpoint.get("meta_confirm_accessed", True):
        raise RuntimeError("verifier provenance is contaminated")
    if checkpoint["p3_sha256"] != train["p3_sha256"] or checkpoint["p3_sha256"] != confirm["p3_sha256"]:
        raise RuntimeError("P3 fingerprint differs across train, verifier, and confirm")
    if targets.get("protocol") != TARGET_PROTOCOL or targets.get("split") != "meta_train":
        raise RuntimeError("unexpected train-frozen metric targets")
    if targets["source_cache_sha256"] != sha256(args.train_cache):
        raise RuntimeError("train target/source cache fingerprint mismatch")
    if checkpoint["oracle_sha256"] != sha256(screen["configuration"]["oracle_json"]):
        raise RuntimeError("calibration screen and frozen verifier use different Oracle files")

    groups = tuple(checkpoint["groups"])
    if groups != tuple(screen["groups"]):
        raise RuntimeError("calibration screen and verifier feature groups differ")
    threshold = float(checkpoint["threshold"])
    if abs(threshold - float(screen["selected"]["threshold"])) > 1e-8:
        raise RuntimeError("calibration threshold changed after freezing")
    _validate_feature_contract(train, confirm, groups)

    values = matrix(confirm, groups, "meta_confirm")
    mean = checkpoint["feature_mean"].float()
    std = checkpoint["feature_std"].float()
    if values.shape[1] != checkpoint["input_dim"] or mean.numel() != values.shape[1] or std.numel() != values.shape[1]:
        raise RuntimeError("frozen verifier feature dimension mismatch")
    features = ((values - mean) / std).clamp(-8, 8)
    model = MetricUtilityVerifier(
        checkpoint["input_dim"], checkpoint["model_type"], checkpoint["hidden"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores = model(features).sigmoid()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("meta_confirm scores contain non-finite values")

    accept = scores >= threshold
    selected_by_image = per_image_selective_histograms(confirm, accept)
    metric_accept = metric_utilities(
        confirm["delta_hist"], targets["train_frozen_miou_gradient"]
    ) > 0
    oracle_by_image = per_image_selective_histograms(confirm, metric_accept)
    p3 = metrics_from_histogram(confirm["p3_histograms"].sum(0))
    fixed = metrics_from_histogram(confirm["fixed_s2_histograms"].sum(0))
    selected = metrics_from_histogram(selected_by_image.sum(0))
    metric_oracle = metrics_from_histogram(oracle_by_image.sum(0))
    gain_pp = 100.0 * (selected["valid_miou"] - fixed["valid_miou"])
    oracle_gain_pp = 100.0 * (metric_oracle["valid_miou"] - fixed["valid_miou"])
    oracle_recovery = gain_pp / max(oracle_gain_pp, 1e-9)
    events = event_rows(confirm, selected_by_image)
    nonnegative = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in events.values())
    bootstrap_row = bootstrap(
        confirm, selected_by_image, args.bootstrap_iterations, args.seed
    )
    checks = confirmation_checks(
        gain_pp=gain_pp,
        oracle_recovery=oracle_recovery,
        ci_low_pp=bootstrap_row["ci95_low_pp"],
        damaged_f1=selected["damaged_f1"],
        fixed_damaged_f1=fixed["damaged_f1"],
        nonnegative_events=nonnegative,
        event_total=len(events),
        minimum_gain_pp=args.minimum_gain_pp,
        minimum_oracle_recovery=args.minimum_oracle_recovery,
        minimum_nonnegative_events=args.minimum_nonnegative_events,
    )
    decision = (
        "ADMIT_MACU_E1_TO_VAL450"
        if all(checks.values())
        else "STOP_MACU_E1_AFTER_META_CONFIRM"
    )
    payload = {
        "protocol": "SPI_METRIC_ALIGNED_CLUSTER_VERIFIER_E1_CONFIRM_v1",
        "configuration": vars(args),
        "provenance": {
            "train_cache_sha256": sha256(args.train_cache),
            "confirm_cache_sha256": sha256(args.confirm_cache),
            "train_targets_sha256": sha256(args.train_targets),
            "verifier_sha256": sha256(args.verifier_checkpoint),
            "calibration_screen_sha256": sha256(args.calibration_screen),
            "p3_sha256": confirm["p3_sha256"],
        },
        "groups": groups,
        "threshold": threshold,
        "accepted_fraction": float(accept.float().mean()),
        "rejected_fraction": float((~accept).float().mean()),
        "p3": p3,
        "fixed_s2": fixed,
        "selective": selected,
        "metric_linear_oracle": metric_oracle,
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "metric_oracle_gain_pp": oracle_gain_pp,
        "oracle_recovery_ratio": oracle_recovery,
        "events": events,
        "event_nonnegative": nonnegative,
        "event_total": len(events),
        "bootstrap": bootstrap_row,
        "checks": checks,
        "decision": decision,
        "meta_confirm_accessed": True,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "threshold": threshold,
        "accepted_fraction": payload["accepted_fraction"],
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "metric_oracle_gain_pp": oracle_gain_pp,
        "oracle_recovery_ratio": oracle_recovery,
        "event_nonnegative": f"{nonnegative}/{len(events)}",
        "bootstrap": bootstrap_row,
        "checks": checks,
        "decision": decision,
    }, indent=2), flush=True)
    print(f"Saved: {output}")
    print("Meta-confirm accessed: ONCE. Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
