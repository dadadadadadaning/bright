from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import (
    bootstrap, event_rows,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    per_image_selective_histograms,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import (
    matrix, sha256,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import LOEO_PROTOCOL
from .core import (
    apply_thresholds, choose_thresholds, cluster_event_mask, dual_targets,
    frozen_gradients, image_event_mask, predict, train_dual_linear,
)


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def _validate_cache(cache: dict, split: str) -> None:
    if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"unexpected {split} D1 cache")
    if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError(f"{split} cache provenance is contaminated")


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    if len(labels) < 2 or int(labels.unique().numel()) < 2:
        return float("nan")
    return binary_auroc(labels.long(), scores)


def _held_rows(
    cache: dict, held_cluster_mask: torch.Tensor, held_image_mask: torch.Tensor,
    accept_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accept = torch.ones(len(cache["image_index"]), dtype=torch.bool)
    accept[held_cluster_mask] = accept_local
    selected = per_image_selective_histograms(cache, accept)
    return (
        cache["p3_histograms"][held_image_mask].double(),
        cache["fixed_s2_histograms"][held_image_mask].double(),
        selected[held_image_mask].double(),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("14-fold event-held-out Class-Safe MACU V2")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--d0-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated LOEO experiment: {output}")
    d0 = json.loads(Path(args.d0_json).read_text(encoding="utf-8"))
    if d0.get("decision") != "ADMIT_CS_MACU_V2_DEVELOPMENT_ONLY":
        raise RuntimeError("CS-MACU V2 refused: D0 Class-Safe Oracle did not pass")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate_cache(train, "meta_train")
    _validate_cache(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("P3 fingerprint differs between development caches")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta_train/meta_calibration overlap")
    groups = tuple(dict.fromkeys(name.strip() for name in args.groups.split(",") if name.strip()))
    for group in groups:
        if train["feature_names"][group] != calibration["feature_names"][group]:
            raise RuntimeError(f"feature contract differs for {group}")
    raw_train = matrix(train, groups, "meta_train")
    raw_calibration = matrix(calibration, groups, "meta_calibration")
    events = sorted(set(train["image_events"]) | set(calibration["image_events"]))
    if len(events) != 14:
        raise RuntimeError(f"expected 14 events for LOEO, observed {len(events)}")

    oof_p3, oof_fixed, oof_selected = [], [], []
    oof_names, oof_events = [], []
    fold_rows = {}
    accepted_total = cluster_total = 0
    for fold_index, event in enumerate(events):
        train_cluster = cluster_event_mask(train, event, equal=False)
        train_images = image_event_mask(train, event, equal=False)
        inner_cal_cluster = cluster_event_mask(calibration, event, equal=False)
        inner_cal_images = image_event_mask(calibration, event, equal=False)
        held_train_cluster = ~train_cluster
        held_train_images = ~train_images
        held_cal_cluster = ~inner_cal_cluster
        held_cal_images = ~inner_cal_images

        metric_gradient, damaged_gradient = frozen_gradients(train, train_images)
        metric_label, harm_label, metric_utility, damaged_utility = dual_targets(
            train, metric_gradient, damaged_gradient
        )
        mean = raw_train[train_cluster].mean(0)
        std = raw_train[train_cluster].std(0, unbiased=False).clamp_min(1e-4)
        train_features = ((raw_train - mean) / std).clamp(-8, 8)
        cal_features = ((raw_calibration - mean) / std).clamp(-8, 8)
        print(
            f"CS-MACU LOEO {fold_index+1}/14 {event}: training on "
            f"{int(train_cluster.sum())} clusters",
            flush=True,
        )
        model = train_dual_linear(
            train_features, train, train_cluster,
            metric_label, harm_label, metric_utility, damaged_utility,
            steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + 1009 * fold_index,
        )
        train_metric_score, train_harm_score = predict(model, train_features)
        cal_metric_score, cal_harm_score = predict(model, cal_features)
        policy = choose_thresholds(
            calibration, cal_metric_score, cal_harm_score,
            inner_cal_cluster, inner_cal_images,
            maximum_reject_fraction=args.maximum_reject_fraction,
        )
        held_train_accept = apply_thresholds(
            train_metric_score[held_train_cluster],
            train_harm_score[held_train_cluster], policy,
        )
        held_cal_accept = apply_thresholds(
            cal_metric_score[held_cal_cluster],
            cal_harm_score[held_cal_cluster], policy,
        )
        train_rows = _held_rows(
            train, held_train_cluster, held_train_images, held_train_accept
        )
        cal_rows = _held_rows(
            calibration, held_cal_cluster, held_cal_images, held_cal_accept
        )
        held_p3 = torch.cat((train_rows[0], cal_rows[0]))
        held_fixed = torch.cat((train_rows[1], cal_rows[1]))
        held_selected = torch.cat((train_rows[2], cal_rows[2]))
        fixed_report = metrics_from_histogram(held_fixed.sum(0))
        selected_report = metrics_from_histogram(held_selected.sum(0))

        cal_metric_label, cal_harm_label, _, _ = dual_targets(
            calibration, metric_gradient, damaged_gradient
        )
        held_metric_labels = torch.cat((
            metric_label[held_train_cluster], cal_metric_label[held_cal_cluster]
        )).long()
        held_harm_labels = torch.cat((
            harm_label[held_train_cluster], cal_harm_label[held_cal_cluster]
        )).long()
        held_metric_scores = torch.cat((
            train_metric_score[held_train_cluster], cal_metric_score[held_cal_cluster]
        ))
        held_harm_scores = torch.cat((
            train_harm_score[held_train_cluster], cal_harm_score[held_cal_cluster]
        ))
        held_accept = torch.cat((held_train_accept, held_cal_accept))
        fold_rows[event] = {
            "policy": policy,
            "images": len(held_p3),
            "clusters": len(held_accept),
            "accepted_fraction": float(held_accept.float().mean()) if len(held_accept) else 0.0,
            "metric_head_auroc": _auc(held_metric_labels, held_metric_scores),
            "harm_head_auroc": _auc(held_harm_labels, held_harm_scores),
            "gain_vs_fixed_s2_miou_pp": 100.0 * (
                selected_report["valid_miou"] - fixed_report["valid_miou"]
            ),
            "damaged_f1_change_pp": 100.0 * (
                selected_report["damaged_f1"] - fixed_report["damaged_f1"]
            ),
        }
        accepted_total += int(held_accept.sum())
        cluster_total += len(held_accept)
        oof_p3.extend(held_p3)
        oof_fixed.extend(held_fixed)
        oof_selected.extend(held_selected)
        train_image_ids = torch.nonzero(held_train_images, as_tuple=False).flatten().tolist()
        cal_image_ids = torch.nonzero(held_cal_images, as_tuple=False).flatten().tolist()
        oof_names.extend([train["image_names"][index] for index in train_image_ids])
        oof_names.extend([calibration["image_names"][index] for index in cal_image_ids])
        oof_events.extend([event] * len(held_p3))
        print(
            f"CS-MACU LOEO {fold_index+1}/14 {event}: "
            f"gain={fold_rows[event]['gain_vs_fixed_s2_miou_pp']:+.4f}pp "
            f"D-F1={fold_rows[event]['damaged_f1_change_pp']:+.4f}pp",
            flush=True,
        )

    p3_rows = torch.stack(oof_p3)
    fixed_rows = torch.stack(oof_fixed)
    selected_rows = torch.stack(oof_selected)
    aggregate_cache = {
        "fixed_s2_histograms": fixed_rows,
        "image_events": oof_events,
    }
    p3_report = metrics_from_histogram(p3_rows.sum(0))
    fixed_report = metrics_from_histogram(fixed_rows.sum(0))
    selected_report = metrics_from_histogram(selected_rows.sum(0))
    gain_pp = 100.0 * (selected_report["valid_miou"] - fixed_report["valid_miou"])
    damaged_change_pp = 100.0 * (
        selected_report["damaged_f1"] - fixed_report["damaged_f1"]
    )
    event_results = event_rows(aggregate_cache, selected_rows)
    nonnegative = sum(
        row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in event_results.values()
    )
    bootstrap_row = bootstrap(
        aggregate_cache, selected_rows, args.bootstrap_iterations, args.seed + 9001
    )
    checks = {
        "gain_vs_fixed_s2_ge_threshold": gain_pp >= args.minimum_gain_pp,
        "damaged_f1_not_below_fixed_s2": damaged_change_pp >= 0,
        "nonnegative_held_events_ge_threshold": nonnegative
        >= min(args.minimum_nonnegative_events, len(events)),
        "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
        "all_images_predicted_once": len(oof_names)
        == len(train["image_names"]) + len(calibration["image_names"])
        and len(set(oof_names)) == len(oof_names),
    }
    decision = (
        "FREEZE_CS_MACU_V2_FOR_VAL450"
        if all(checks.values())
        else "STOP_CS_MACU_V2_AFTER_LOEO"
    )
    payload = {
        "protocol": LOEO_PROTOCOL,
        "configuration": vars(args),
        "provenance": {
            "train_cache_sha256": sha256(args.train_cache),
            "calibration_cache_sha256": sha256(args.calibration_cache),
            "d0_sha256": sha256(args.d0_json),
        },
        "groups": groups,
        "p3": p3_report,
        "fixed_s2": fixed_report,
        "class_safe_macu_v2_oof": selected_report,
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "damaged_f1_change_pp": damaged_change_pp,
        "accepted_fraction": accepted_total / max(cluster_total, 1),
        "folds": fold_rows,
        "events": event_results,
        "event_nonnegative": nonnegative,
        "event_total": len(events),
        "bootstrap": bootstrap_row,
        "checks": checks,
        "decision": decision,
        "meta_confirm_used_for_training_or_loeo": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "gain_vs_fixed_s2_miou_pp": gain_pp,
        "damaged_f1_change_pp": damaged_change_pp,
        "event_nonnegative": f"{nonnegative}/{len(events)}",
        "accepted_fraction": payload["accepted_fraction"],
        "bootstrap": bootstrap_row,
        "checks": checks,
        "decision": decision,
    }, indent=2), flush=True)
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test used: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
