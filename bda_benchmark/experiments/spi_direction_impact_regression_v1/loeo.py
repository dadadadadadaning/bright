from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_class_safe_macu_v2.core import (
    cluster_event_mask, image_event_mask,
)
from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_direction_impact_utility_v1.core import (
    exact_keep_utilities, weighted_binary_auroc,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import bootstrap
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import per_image_selective_histograms
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import matrix, sha256
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import PRIMARY_VARIANT, PROTOCOL
from .core import (
    VARIANTS, apply_policy, choose_conservative_policy, direction_one_hot,
    predict, risk_score, signed_log_transform, target_scale, train_regressor,
    uses_direction, uses_impact,
)


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"unexpected {split} D1 cache")
    if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError(f"{split} D1 cache provenance is contaminated")


def _exact_by_event(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_events = [cache["image_events"][int(index)] for index in cache["image_index"]]
    metric = torch.empty(len(cluster_events), dtype=torch.float64)
    damaged = torch.empty_like(metric)
    for event in sorted(set(cache["image_events"])):
        image_mask = torch.tensor([value == event for value in cache["image_events"]])
        cluster_mask = torch.tensor([value == event for value in cluster_events])
        metric[cluster_mask], damaged[cluster_mask] = exact_keep_utilities(
            cache["fixed_s2_histograms"][image_mask].sum(0),
            cache["delta_hist"][cluster_mask],
        )
    return metric, damaged


def _held_histograms(cache: dict, cluster_mask: torch.Tensor, image_mask: torch.Tensor, local_accept: torch.Tensor):
    accepted = torch.ones(len(cache["image_index"]), dtype=torch.bool)
    accepted[cluster_mask] = local_accept
    selected = per_image_selective_histograms(cache, accepted)
    return (
        cache["p3_histograms"][image_mask].double(),
        cache["fixed_s2_histograms"][image_mask].double(),
        selected[image_mask].double(),
    )


def _event_metrics(fixed_rows: torch.Tensor, selected_rows: torch.Tensor, events: list[str]) -> dict:
    rows = {}
    for event in sorted(set(events)):
        ids = torch.tensor([index for index, value in enumerate(events) if value == event])
        fixed = metrics_from_histogram(fixed_rows[ids].sum(0))
        selected = metrics_from_histogram(selected_rows[ids].sum(0))
        rows[event] = {
            "gain_vs_fixed_s2_miou_pp": 100.0 * (selected["valid_miou"] - fixed["valid_miou"]),
            "damaged_f1_change_pp": 100.0 * (selected["damaged_f1"] - fixed["damaged_f1"]),
        }
    return rows


def _rank_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if len(prediction) < 2:
        return float("nan")
    left = torch.argsort(torch.argsort(prediction.double(), stable=True), stable=True).double()
    right = torch.argsort(torch.argsort(target.double(), stable=True), stable=True).double()
    left -= left.mean(); right -= right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-12))


def _summarize(name: str, rows: dict, events: list[str], iterations: int, seed: int) -> dict:
    p3 = torch.stack(rows["p3"])
    fixed = torch.stack(rows["fixed"])
    selected = torch.stack(rows["selected"])
    fixed_report = metrics_from_histogram(fixed.sum(0))
    selected_report = metrics_from_histogram(selected.sum(0))
    gain = 100.0 * (selected_report["valid_miou"] - fixed_report["valid_miou"])
    damaged = 100.0 * (selected_report["damaged_f1"] - fixed_report["damaged_f1"])
    event_rows = _event_metrics(fixed, selected, events)
    event_miou = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in event_rows.values())
    event_damaged = sum(row["damaged_f1_change_pp"] >= 0 for row in event_rows.values())
    metric_prediction = torch.cat(rows["metric_prediction"])
    damaged_prediction = torch.cat(rows["damaged_prediction"])
    metric_target = torch.cat(rows["metric_utility"])
    damaged_target = torch.cat(rows["damaged_utility"])
    temporary = {"fixed_s2_histograms": fixed, "image_events": events}
    bootstrap_row = bootstrap(temporary, selected, iterations, seed)
    checks = {
        "gain_vs_fixed_s2_ge_0_30pp": gain >= .30,
        "damaged_f1_not_below_fixed_s2": damaged >= 0,
        "miou_nonnegative_events_ge_10": event_miou >= 10,
        "damaged_nonnegative_events_ge_10": event_damaged >= 10,
        "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
        "nontrivial_acceptance": rows["accepted"] < rows["clusters"],
        "harm_impact_weighted_auroc_ge_0_60": weighted_binary_auroc(
            damaged_target < 0, -damaged_prediction, damaged_target.abs()
        ) >= .60,
    }
    return {
        "p3": metrics_from_histogram(p3.sum(0)), "fixed_s2": fixed_report,
        "selected": selected_report,
        "gain_vs_fixed_s2_miou_pp": gain, "damaged_f1_change_pp": damaged,
        "accepted_fraction": rows["accepted"] / max(rows["clusters"], 1),
        "event_miou_nonnegative": event_miou,
        "event_damaged_nonnegative": event_damaged,
        "events": event_rows, "bootstrap": bootstrap_row,
        "prediction_quality": {
            "metric_spearman": _rank_correlation(metric_prediction, metric_target),
            "damaged_spearman": _rank_correlation(damaged_prediction, damaged_target),
            "benefit_auroc": weighted_binary_auroc(metric_target > 0, metric_prediction),
            "benefit_impact_weighted_auroc": weighted_binary_auroc(
                metric_target > 0, metric_prediction, metric_target.abs()
            ),
            "harm_auroc": weighted_binary_auroc(damaged_target < 0, -damaged_prediction),
            "harm_impact_weighted_auroc": weighted_binary_auroc(
                damaged_target < 0, -damaged_prediction, damaged_target.abs()
            ),
        },
        "checks": checks, "passed": all(checks.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("E2 direction/impact continuous utility 14-fold LOEO")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--e1-json", required=True)
    parser.add_argument("--v21-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof-output", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--impact-cap", type=float, default=8.0)
    parser.add_argument("--maximum-reject-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    output, oof_output = Path(args.output), Path(args.oof_output)
    if output.exists() or oof_output.exists():
        raise RuntimeError("refusing repeated E2 LOEO output")
    e1 = json.loads(Path(args.e1_json).read_text(encoding="utf-8"))
    previous = json.loads(Path(args.v21_json).read_text(encoding="utf-8"))
    if e1.get("decision") != "ADMIT_DIRECTION_IMPACT_E2_DEVELOPMENT":
        raise RuntimeError("E2 refused: E1 exact utility audit did not pass")
    if previous.get("decision") != "STOP_CS_MACU_V21_AFTER_LOEO":
        raise RuntimeError("E2 requires the frozen V2.1 negative control")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    if args.maximum_reject_fraction > .1000001:
        raise RuntimeError("E2 preregistration caps rejection at 10%")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("P3 provenance differs between E2 caches")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta_train/meta_calibration image leakage")
    raw_groups = args.groups.strip().lower()
    groups = DEFAULT_GROUPS if raw_groups in {"", "0", "none", "false", "disabled"} else tuple(
        dict.fromkeys(value.strip() for value in args.groups.split(",") if value.strip())
    )
    for group in groups:
        if train["feature_names"][group] != calibration["feature_names"][group]:
            raise RuntimeError(f"feature contract differs for {group}")
    raw_train = matrix(train, groups, "meta_train")
    raw_calibration = matrix(calibration, groups, "meta_calibration")
    train_metric, train_damaged = _exact_by_event(train)
    cal_metric, cal_damaged = _exact_by_event(calibration)
    events = sorted(set(train["image_events"]) | set(calibration["image_events"]))
    if len(events) != 14:
        raise RuntimeError(f"expected 14 events, observed {len(events)}")
    direction_train = direction_one_hot(train["direction_code"])
    direction_cal = direction_one_hot(calibration["direction_code"])
    storage = {
        variant: {"p3": [], "fixed": [], "selected": [], "accepted": 0,
                  "clusters": 0, "metric_prediction": [], "damaged_prediction": [],
                  "metric_utility": [], "damaged_utility": []}
        for variant in VARIANTS
    }
    folds, oof_events, oof_names = {}, [], []
    oof_artifact = {variant: {"train": {}, "calibration": {}} for variant in VARIANTS}
    for variant in VARIANTS:
        for split, cache in (("train", train), ("calibration", calibration)):
            count = len(cache["image_index"])
            oof_artifact[variant][split] = {
                "metric_prediction": torch.full((count,), float("nan")),
                "damaged_prediction": torch.full((count,), float("nan")),
                "accepted": torch.zeros(count, dtype=torch.bool),
            }

    for fold_index, event in enumerate(events):
        train_clusters = cluster_event_mask(train, event, equal=False)
        train_images = image_event_mask(train, event, equal=False)
        inner_clusters = cluster_event_mask(calibration, event, equal=False)
        inner_images = image_event_mask(calibration, event, equal=False)
        held_train_clusters, held_train_images = ~train_clusters, ~train_images
        held_cal_clusters, held_cal_images = ~inner_clusters, ~inner_images
        mean = raw_train[train_clusters].mean(0)
        std = raw_train[train_clusters].std(0, unbiased=False).clamp_min(1e-4)
        base_train = ((raw_train - mean) / std).clamp(-8, 8)
        base_cal = ((raw_calibration - mean) / std).clamp(-8, 8)
        metric_scale = target_scale(train_metric, train_clusters)
        damaged_scale = target_scale(train_damaged, train_clusters)
        metric_target = signed_log_transform(train_metric, metric_scale)
        damaged_target = signed_log_transform(train_damaged, damaged_scale)
        fold = {"target_scales": {"metric": metric_scale, "damaged": damaged_scale}, "variants": {}}
        print(f"E2 LOEO {fold_index+1}/14 {event}", flush=True)
        for variant in VARIANTS:
            train_features = torch.cat((base_train, direction_train), 1) if uses_direction(variant) else base_train
            cal_features = torch.cat((base_cal, direction_cal), 1) if uses_direction(variant) else base_cal
            model = train_regressor(
                train_features, train, train_clusters,
                metric_target, damaged_target, train_metric, train_damaged,
                impact=uses_impact(variant), metric_scale=metric_scale,
                damaged_scale=damaged_scale, impact_cap=args.impact_cap,
                steps=args.steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                # Use the same fold seed for every arm.  C1/C3 and C2/C4 then
                # differ only by impact weighting, not by sampling/init noise.
                seed=args.seed + fold_index * 1009,
            )
            train_pred_metric, train_pred_damaged = predict(model, train_features)
            cal_pred_metric, cal_pred_damaged = predict(model, cal_features)
            policy, diagnostics = choose_conservative_policy(
                calibration, risk_score(cal_pred_metric, cal_pred_damaged),
                inner_clusters, inner_images,
                maximum_reject_fraction=args.maximum_reject_fraction,
            )
            held_train_accept = apply_policy(
                risk_score(train_pred_metric[held_train_clusters], train_pred_damaged[held_train_clusters]), policy
            )
            held_cal_accept = apply_policy(
                risk_score(cal_pred_metric[held_cal_clusters], cal_pred_damaged[held_cal_clusters]), policy
            )
            train_hist = _held_histograms(train, held_train_clusters, held_train_images, held_train_accept)
            cal_hist = _held_histograms(calibration, held_cal_clusters, held_cal_images, held_cal_accept)
            p3_rows = torch.cat((train_hist[0], cal_hist[0]))
            fixed_rows = torch.cat((train_hist[1], cal_hist[1]))
            selected_rows = torch.cat((train_hist[2], cal_hist[2]))
            fixed_report = metrics_from_histogram(fixed_rows.sum(0))
            selected_report = metrics_from_histogram(selected_rows.sum(0))
            held_accept = torch.cat((held_train_accept, held_cal_accept))
            rows = storage[variant]
            rows["p3"].extend(p3_rows); rows["fixed"].extend(fixed_rows); rows["selected"].extend(selected_rows)
            rows["accepted"] += int(held_accept.sum()); rows["clusters"] += len(held_accept)
            rows["metric_prediction"].extend((train_pred_metric[held_train_clusters], cal_pred_metric[held_cal_clusters]))
            rows["damaged_prediction"].extend((train_pred_damaged[held_train_clusters], cal_pred_damaged[held_cal_clusters]))
            rows["metric_utility"].extend((train_metric[held_train_clusters], cal_metric[held_cal_clusters]))
            rows["damaged_utility"].extend((train_damaged[held_train_clusters], cal_damaged[held_cal_clusters]))
            for split, held, pm, pd, accepted in (
                ("train", held_train_clusters, train_pred_metric, train_pred_damaged, held_train_accept),
                ("calibration", held_cal_clusters, cal_pred_metric, cal_pred_damaged, held_cal_accept),
            ):
                oof_artifact[variant][split]["metric_prediction"][held] = pm[held]
                oof_artifact[variant][split]["damaged_prediction"][held] = pd[held]
                oof_artifact[variant][split]["accepted"][held] = accepted
            fold["variants"][variant] = {
                "policy": policy, "threshold_diagnostics": diagnostics,
                "accepted_fraction": float(held_accept.float().mean()),
                "gain_vs_fixed_s2_miou_pp": 100.0 * (selected_report["valid_miou"] - fixed_report["valid_miou"]),
                "damaged_f1_change_pp": 100.0 * (selected_report["damaged_f1"] - fixed_report["damaged_f1"]),
            }
            print(
                f"  {variant}: gain={fold['variants'][variant]['gain_vs_fixed_s2_miou_pp']:+.4f}pp "
                f"D-F1={fold['variants'][variant]['damaged_f1_change_pp']:+.4f}pp "
                f"accept={fold['variants'][variant]['accepted_fraction']:.3f}", flush=True,
            )
        folds[event] = fold
        held_train_ids = torch.nonzero(held_train_images, as_tuple=False).flatten().tolist()
        held_cal_ids = torch.nonzero(held_cal_images, as_tuple=False).flatten().tolist()
        names = [train["image_names"][index] for index in held_train_ids]
        names += [calibration["image_names"][index] for index in held_cal_ids]
        oof_names.extend(names); oof_events.extend([event] * len(names))

    all_images_once = len(oof_names) == len(train["image_names"]) + len(calibration["image_names"]) and len(set(oof_names)) == len(oof_names)
    summaries = {
        variant: _summarize(variant, storage[variant], oof_events, args.bootstrap_iterations, args.seed + 5000 + index)
        for index, variant in enumerate(VARIANTS)
    }
    c1, c2, c3, c4 = (summaries[name] for name in VARIANTS)
    attribution = {
        "direction_improves_benefit_impact_auc_ge_0_02": c2["prediction_quality"]["benefit_impact_weighted_auroc"] >= c1["prediction_quality"]["benefit_impact_weighted_auroc"] + .02,
        "impact_improves_harm_impact_auc_ge_0_05": c3["prediction_quality"]["harm_impact_weighted_auroc"] >= c1["prediction_quality"]["harm_impact_weighted_auroc"] + .05,
        "joint_gain_not_below_single_variants": c4["gain_vs_fixed_s2_miou_pp"] >= max(c2["gain_vs_fixed_s2_miou_pp"], c3["gain_vs_fixed_s2_miou_pp"]),
    }
    passed = all_images_once and c4["passed"] and all(attribution.values())
    decision = "ADMIT_DIRECTION_IMPACT_E3_LOW_CAPACITY_VERIFIER" if passed else "STOP_DIRECTION_IMPACT_E2_AFTER_LOEO"
    payload = {
        "protocol": PROTOCOL, "configuration": vars(args),
        "primary_variant": PRIMARY_VARIANT,
        "c0_v21_benefit_reference": previous["arms"]["b1_benefit_only"],
        "variants": summaries, "folds": folds,
        "attribution_checks": attribution,
        "all_images_predicted_once": all_images_once, "decision": decision,
        "provenance": {"train_cache_sha256": sha256(args.train_cache),
                       "calibration_cache_sha256": sha256(args.calibration_cache),
                       "e1_json_sha256": sha256(args.e1_json),
                       "v21_json_sha256": sha256(args.v21_json)},
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": PROTOCOL, "rows": oof_artifact,
                "meta_confirm_accessed": False, "val450_accessed": False,
                "test_accessed": False}, oof_output)
    for variant in VARIANTS:
        row = summaries[variant]
        print(f"E2 {variant:22s} gain={row['gain_vs_fixed_s2_miou_pp']:+.4f}pp D-F1={row['damaged_f1_change_pp']:+.4f}pp events={row['event_miou_nonnegative']}/14 impact-harm-AUC={row['prediction_quality']['harm_impact_weighted_auroc']:.4f} passed={row['passed']}")
    print(f"Attribution: {attribution}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
