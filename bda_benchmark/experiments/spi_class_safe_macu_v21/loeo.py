from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_class_safe_macu_v2.core import (
    cluster_event_mask, dual_targets, frozen_gradients, image_event_mask,
    predict, train_dual_linear,
)
from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import bootstrap
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import per_image_selective_histograms
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import matrix, sha256
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PRIMARY_ARM, PROTOCOL
from .core import ARMS, apply_arm_policy, select_arm_policy


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"unexpected {split} D1 cache")
    if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError(f"{split} cache provenance is contaminated")


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.long()
    return binary_auroc(labels, scores) if len(labels) and int(labels.unique().numel()) == 2 else float("nan")


def _held_histograms(
    cache: dict, cluster_mask: torch.Tensor, image_mask: torch.Tensor,
    local_accept: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accept = torch.ones(len(cache["image_index"]), dtype=torch.bool)
    accept[cluster_mask] = local_accept
    selected = per_image_selective_histograms(cache, accept)
    return (
        cache["p3_histograms"][image_mask].double(),
        cache["fixed_s2_histograms"][image_mask].double(),
        selected[image_mask].double(),
    )


def _event_metrics(fixed_rows: torch.Tensor, selected_rows: torch.Tensor, events: list[str]) -> dict:
    result = {}
    for event in sorted(set(events)):
        ids = torch.tensor([index for index, value in enumerate(events) if value == event])
        fixed = metrics_from_histogram(fixed_rows[ids].sum(0))
        selected = metrics_from_histogram(selected_rows[ids].sum(0))
        result[event] = {
            "gain_vs_fixed_s2_miou_pp": 100.0 * (
                selected["valid_miou"] - fixed["valid_miou"]
            ),
            "damaged_f1_change_pp": 100.0 * (
                selected["damaged_f1"] - fixed["damaged_f1"]
            ),
        }
    return result


def _summarize_arm(
    arm: str, p3_rows: torch.Tensor, fixed_rows: torch.Tensor,
    selected_rows: torch.Tensor, events: list[str], accepted: int,
    clusters: int, bootstrap_iterations: int, seed: int,
) -> dict:
    p3 = metrics_from_histogram(p3_rows.sum(0))
    fixed = metrics_from_histogram(fixed_rows.sum(0))
    selected = metrics_from_histogram(selected_rows.sum(0))
    event_rows = _event_metrics(fixed_rows, selected_rows, events)
    miou_nonnegative = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in event_rows.values())
    damaged_nonnegative = sum(row["damaged_f1_change_pp"] >= 0 for row in event_rows.values())
    temporary_cache = {"fixed_s2_histograms": fixed_rows, "image_events": events}
    bootstrap_row = bootstrap(temporary_cache, selected_rows, bootstrap_iterations, seed)
    gain_pp = 100.0 * (selected["valid_miou"] - fixed["valid_miou"])
    damaged_pp = 100.0 * (selected["damaged_f1"] - fixed["damaged_f1"])
    if arm == PRIMARY_ARM:
        checks = {
            "gain_vs_fixed_s2_ge_0_20pp": gain_pp >= 0.20,
            "damaged_f1_strictly_above_fixed_s2": damaged_pp > 0,
            "miou_nonnegative_events_ge_10": miou_nonnegative >= 10,
            "damaged_nonnegative_events_ge_10": damaged_nonnegative >= 10,
            "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
            "nontrivial_acceptance_below_one": accepted < clusters,
        }
    elif arm == "b0_fixed_s2":
        checks = {"identity_control": gain_pp == 0 and damaged_pp == 0 and accepted == clusters}
    else:
        checks = {
            "gain_vs_fixed_s2_ge_0_30pp": gain_pp >= 0.30,
            "damaged_f1_not_below_fixed_s2": damaged_pp >= 0,
            "miou_nonnegative_events_ge_10": miou_nonnegative >= 10,
            "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
            "nontrivial_acceptance_below_one": accepted < clusters,
        }
    return {
        "p3": p3,
        "fixed_s2": fixed,
        "selected": selected,
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "damaged_f1_change_pp": damaged_pp,
        "accepted_fraction": accepted / max(clusters, 1),
        "event_miou_nonnegative": miou_nonnegative,
        "event_damaged_nonnegative": damaged_nonnegative,
        "event_total": len(event_rows),
        "events": event_rows,
        "bootstrap": bootstrap_row,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("CS-MACU V2.1 three-arm 14-fold LOEO")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--v2-json", required=True)
    parser.add_argument("--d0-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated V2.1 LOEO: {output}")
    d0 = json.loads(Path(args.d0_json).read_text(encoding="utf-8"))
    old_v2 = json.loads(Path(args.v2_json).read_text(encoding="utf-8"))
    if d0.get("decision") != "ADMIT_CS_MACU_V2_DEVELOPMENT_ONLY":
        raise RuntimeError("V2.1 refused: Class-Safe Oracle did not pass")
    if old_v2.get("decision") != "STOP_CS_MACU_V2_AFTER_LOEO" or old_v2.get("accepted_fraction") != 1.0:
        raise RuntimeError("V2.1 requires the documented all-accept V2 negative control")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train")
    _validate(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"] or set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("development cache provenance mismatch")
    groups = tuple(dict.fromkeys(name.strip() for name in args.groups.split(",") if name.strip()))
    for group in groups:
        if train["feature_names"][group] != calibration["feature_names"][group]:
            raise RuntimeError(f"feature contract differs for {group}")
    raw_train = matrix(train, groups, "meta_train")
    raw_calibration = matrix(calibration, groups, "meta_calibration")
    events = sorted(set(train["image_events"]) | set(calibration["image_events"]))
    if len(events) != 14:
        raise RuntimeError(f"expected 14 events, observed {len(events)}")

    arm_rows = {
        arm: {"p3": [], "fixed": [], "selected": [], "accepted": 0, "clusters": 0}
        for arm in ARMS
    }
    oof_events, oof_names, folds = [], [], {}
    for fold_index, event in enumerate(events):
        train_cluster = cluster_event_mask(train, event, equal=False)
        train_images = image_event_mask(train, event, equal=False)
        inner_cluster = cluster_event_mask(calibration, event, equal=False)
        inner_images = image_event_mask(calibration, event, equal=False)
        held_train_cluster, held_train_images = ~train_cluster, ~train_images
        held_cal_cluster, held_cal_images = ~inner_cluster, ~inner_images
        metric_gradient, damaged_gradient = frozen_gradients(train, train_images)
        labels = dual_targets(train, metric_gradient, damaged_gradient)
        mean = raw_train[train_cluster].mean(0)
        std = raw_train[train_cluster].std(0, unbiased=False).clamp_min(1e-4)
        train_features = ((raw_train - mean) / std).clamp(-8, 8)
        calibration_features = ((raw_calibration - mean) / std).clamp(-8, 8)
        print(f"CS-MACU V2.1 {fold_index+1}/14 {event}: training", flush=True)
        model = train_dual_linear(
            train_features, train, train_cluster, *labels,
            steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + fold_index * 1009,
        )
        train_metric, train_harm = predict(model, train_features)
        cal_metric, cal_harm = predict(model, calibration_features)
        fold = {
            "metric_head_auroc": _auc(
                torch.cat((labels[0][held_train_cluster], dual_targets(calibration, metric_gradient, damaged_gradient)[0][held_cal_cluster])),
                torch.cat((train_metric[held_train_cluster], cal_metric[held_cal_cluster])),
            ),
            "harm_head_auroc": _auc(
                torch.cat((labels[1][held_train_cluster], dual_targets(calibration, metric_gradient, damaged_gradient)[1][held_cal_cluster])),
                torch.cat((train_harm[held_train_cluster], cal_harm[held_cal_cluster])),
            ),
            "arms": {},
        }
        policies = {}
        for arm in ARMS:
            policy, diagnostics = select_arm_policy(
                arm, calibration, cal_metric, cal_harm,
                inner_cluster, inner_images,
                maximum_reject_fraction=args.maximum_reject_fraction,
            )
            policies[arm] = policy
            held_train_accept = apply_arm_policy(
                arm, train_metric[held_train_cluster], train_harm[held_train_cluster], policy
            )
            held_cal_accept = apply_arm_policy(
                arm, cal_metric[held_cal_cluster], cal_harm[held_cal_cluster], policy
            )
            train_hist = _held_histograms(
                train, held_train_cluster, held_train_images, held_train_accept
            )
            cal_hist = _held_histograms(
                calibration, held_cal_cluster, held_cal_images, held_cal_accept
            )
            p3_rows = torch.cat((train_hist[0], cal_hist[0]))
            fixed_rows = torch.cat((train_hist[1], cal_hist[1]))
            selected_rows = torch.cat((train_hist[2], cal_hist[2]))
            fixed_report = metrics_from_histogram(fixed_rows.sum(0))
            selected_report = metrics_from_histogram(selected_rows.sum(0))
            held_accept = torch.cat((held_train_accept, held_cal_accept))
            fold["arms"][arm] = {
                "policy": policy,
                "threshold_diagnostics": diagnostics,
                "accepted_fraction": float(held_accept.float().mean()) if len(held_accept) else 0.0,
                "gain_vs_fixed_s2_miou_pp": 100.0 * (
                    selected_report["valid_miou"] - fixed_report["valid_miou"]
                ),
                "damaged_f1_change_pp": 100.0 * (
                    selected_report["damaged_f1"] - fixed_report["damaged_f1"]
                ),
            }
            arm_rows[arm]["p3"].extend(p3_rows)
            arm_rows[arm]["fixed"].extend(fixed_rows)
            arm_rows[arm]["selected"].extend(selected_rows)
            arm_rows[arm]["accepted"] += int(held_accept.sum())
            arm_rows[arm]["clusters"] += len(held_accept)
        fold["inner_calibration_best_arm_diagnostic_only"] = max(
            (arm for arm in ARMS if arm != "b0_fixed_s2"),
            key=lambda arm: (policies[arm]["miou"], policies[arm]["damaged_f1"]),
        )
        folds[event] = fold
        held_train_ids = torch.nonzero(held_train_images, as_tuple=False).flatten().tolist()
        held_cal_ids = torch.nonzero(held_cal_images, as_tuple=False).flatten().tolist()
        names = [train["image_names"][index] for index in held_train_ids]
        names += [calibration["image_names"][index] for index in held_cal_ids]
        oof_names.extend(names)
        oof_events.extend([event] * len(names))
        primary = fold["arms"][PRIMARY_ARM]
        print(
            f"CS-MACU V2.1 {fold_index+1}/14 {event}: harm-only "
            f"gain={primary['gain_vs_fixed_s2_miou_pp']:+.4f}pp "
            f"D-F1={primary['damaged_f1_change_pp']:+.4f}pp "
            f"accept={primary['accepted_fraction']:.3f}",
            flush=True,
        )

    all_images_once = len(oof_names) == len(train["image_names"]) + len(calibration["image_names"]) and len(set(oof_names)) == len(oof_names)
    summaries = {}
    for arm_index, arm in enumerate(ARMS):
        rows = arm_rows[arm]
        summaries[arm] = _summarize_arm(
            arm, torch.stack(rows["p3"]), torch.stack(rows["fixed"]),
            torch.stack(rows["selected"]), oof_events,
            rows["accepted"], rows["clusters"],
            args.bootstrap_iterations, args.seed + 5000 + arm_index,
        )
    primary_passed = summaries[PRIMARY_ARM]["passed"] and all_images_once
    decision = "FREEZE_CS_MACU_V21_HARM_ONLY_FOR_VAL450" if primary_passed else "STOP_CS_MACU_V21_AFTER_LOEO"
    payload = {
        "protocol": PROTOCOL,
        "configuration": vars(args),
        "provenance": {
            "train_cache_sha256": sha256(args.train_cache),
            "calibration_cache_sha256": sha256(args.calibration_cache),
            "d0_sha256": sha256(args.d0_json),
            "v2_negative_control_sha256": sha256(args.v2_json),
        },
        "primary_arm": PRIMARY_ARM,
        "arm_selection_rule": "B2 Harm-only is preregistered primary; B1/B3 cannot replace it based on OOF results.",
        "folds": folds,
        "arms": summaries,
        "all_images_predicted_once": all_images_once,
        "decision": decision,
        "meta_confirm_used_for_training_threshold_or_loeo": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for arm in ARMS:
        row = summaries[arm]
        print(
            f"V2.1 {arm:18s} gain={row['gain_vs_fixed_s2_miou_pp']:+.4f}pp "
            f"D-F1={row['damaged_f1_change_pp']:+.4f}pp "
            f"events={row['event_miou_nonnegative']}/14 "
            f"accept={row['accepted_fraction']:.3f} passed={row['passed']}",
            flush=True,
        )
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test used: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
