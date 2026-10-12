from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_class_safe_macu_v2.core import (
    cluster_event_mask, image_event_mask,
)
from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_direction_impact_utility_v1.core import weighted_binary_auroc
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import bootstrap
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import per_image_selective_histograms
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import matrix, sha256
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import AUDIT_PROTOCOL, OOF_PROTOCOL, PRIMARY_VARIANT, PROTOCOL
from .core import (
    VARIANTS, apply_policy, choose_policy, expected_confusion_delta,
    expected_confusion_risk, neutral_truth_prior, outcome_confidence,
    outcome_counts, outcome_proportions, predict_outcomes, simple_outcome_risk,
    train_outcome_model,
)


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"unexpected {split} D1 cache")
    forbidden = ("meta_confirm_accessed", "val450_accessed", "official_instances_accessed", "test_accessed")
    if any(cache.get(name, True) for name in forbidden):
        raise RuntimeError(f"{split} cache provenance is contaminated")


def _held_rows(cache: dict, cluster_mask: torch.Tensor, image_mask: torch.Tensor, accept: torch.Tensor):
    global_accept = torch.ones(len(cache["image_index"]), dtype=torch.bool)
    global_accept[cluster_mask] = accept
    selected = per_image_selective_histograms(cache, global_accept)
    return (
        cache["p3_histograms"][image_mask].double(),
        cache["fixed_s2_histograms"][image_mask].double(),
        selected[image_mask].double(),
    )


def _event_rows(fixed: torch.Tensor, selected: torch.Tensor, events: list[str]) -> dict:
    result = {}
    for event in sorted(set(events)):
        ids = torch.tensor([value == event for value in events])
        base = metrics_from_histogram(fixed[ids].sum(0))
        changed = metrics_from_histogram(selected[ids].sum(0))
        result[event] = {
            "gain_vs_fixed_s2_miou_pp": 100 * (changed["valid_miou"] - base["valid_miou"]),
            "damaged_f1_change_pp": 100 * (changed["damaged_f1"] - base["damaged_f1"]),
        }
    return result


def _soft_ece(probability: torch.Tensor, target_fraction: torch.Tensor, bins: int = 10) -> float:
    total = max(len(probability), 1)
    value = 0.0
    for lower in torch.linspace(0, .9, bins):
        upper = lower + 1.0 / bins
        selected = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if bool(selected.any()):
            value += float(selected.sum()) / total * abs(
                float(probability[selected].mean() - target_fraction[selected].mean())
            )
    return value


def _summary(rows: dict, events: list[str], iterations: int, seed: int) -> dict:
    p3, fixed, selected = (torch.stack(rows[name]) for name in ("p3", "fixed", "selected"))
    fixed_report = metrics_from_histogram(fixed.sum(0))
    report = metrics_from_histogram(selected.sum(0))
    gain = 100 * (report["valid_miou"] - fixed_report["valid_miou"])
    damaged = 100 * (report["damaged_f1"] - fixed_report["damaged_f1"])
    by_event = _event_rows(fixed, selected, events)
    miou_events = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in by_event.values())
    damaged_events = sum(row["damaged_f1_change_pp"] >= 0 for row in by_event.values())
    probabilities = torch.cat(rows["probabilities"])
    targets = torch.cat(rows["targets"])
    counts = torch.cat(rows["counts"])
    area_weights = counts.sum(1)
    cross_entropy = float((-(targets * probabilities.clamp_min(1e-8).log()).sum(1)).mean())
    weighted_mae = float(
        ((probabilities - targets).abs() * area_weights[:, None]).sum()
        / (3 * area_weights.sum().clamp_min(1.0))
    )
    temporary = {"fixed_s2_histograms": fixed, "image_events": events}
    bootstrap_row = bootstrap(temporary, selected, iterations, seed)
    checks = {
        "gain_vs_fixed_s2_ge_0_20pp": gain >= .20,
        "damaged_f1_not_below_fixed_s2": damaged >= 0,
        "miou_nonnegative_events_ge_10": miou_events >= 10,
        "damaged_nonnegative_events_ge_10": damaged_events >= 10,
        "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
        "nontrivial_acceptance": rows["accepted"] < rows["clusters"],
    }
    return {
        "p3": metrics_from_histogram(p3.sum(0)), "fixed_s2": fixed_report,
        "selected": report, "gain_vs_fixed_s2_miou_pp": gain,
        "damaged_f1_change_pp": damaged,
        "accepted_fraction": rows["accepted"] / max(rows["clusters"], 1),
        "event_miou_nonnegative": miou_events,
        "event_damaged_nonnegative": damaged_events,
        "events": by_event, "bootstrap": bootstrap_row,
        "outcome_quality": {
            "soft_cross_entropy": cross_entropy,
            "pixel_weighted_probability_mae": weighted_mae,
            "corrected_majority_auroc": weighted_binary_auroc(
                counts[:, 0] > counts[:, 1], probabilities[:, 0], area_weights
            ),
            "harmed_majority_auroc": weighted_binary_auroc(
                counts[:, 1] > counts[:, 0], probabilities[:, 1], area_weights
            ),
            "harmed_probability_ece": _soft_ece(probabilities[:, 1], targets[:, 1]),
        },
        "checks": checks, "passed": all(checks.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("E2.1 cluster outcome 14-event LOEO")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--e2-json", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof-output", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--direction-shrinkage", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    output, oof_output = Path(args.output), Path(args.oof_output)
    if output.exists() or oof_output.exists():
        raise RuntimeError("refusing repeated E2.1 LOEO output")
    previous = json.loads(Path(args.e2_json).read_text(encoding="utf-8"))
    if previous.get("decision") != "STOP_DIRECTION_IMPACT_E2_AFTER_LOEO":
        raise RuntimeError("E2.1 requires the frozen failed E2 result")
    audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
    if audit.get("protocol") != AUDIT_PROTOCOL or audit.get("decision") != "ADMIT_CLUSTER_OUTCOME_E21_LOEO":
        raise RuntimeError("E2.1 LOEO refused: local outcome Oracle audit did not pass")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    if args.maximum_reject_fraction > .1000001:
        raise RuntimeError("E2.1 preregistration caps rejection at 10%")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("P3 provenance mismatch")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("meta split leakage")
    raw_groups = args.groups.strip().lower()
    groups = DEFAULT_GROUPS if raw_groups in {"", "0", "none", "false", "disabled"} else tuple(
        dict.fromkeys(value.strip() for value in args.groups.split(",") if value.strip())
    )
    for group in groups:
        if train["feature_names"][group] != calibration["feature_names"][group]:
            raise RuntimeError(f"feature contract differs for {group}")
    raw_train = matrix(train, groups, "meta_train")
    raw_cal = matrix(calibration, groups, "meta_calibration")
    events = sorted(set(train["image_events"]) | set(calibration["image_events"]))
    if len(events) != 14:
        raise RuntimeError(f"expected 14 events, got {len(events)}")
    stores = {
        name: {"p3": [], "fixed": [], "selected": [], "probabilities": [],
               "targets": [], "counts": [], "accepted": 0, "clusters": 0}
        for name in VARIANTS
    }
    oof = {name: {split: {
        "probabilities": torch.full((len(cache["image_index"]), 3), float("nan")),
        "accepted": torch.zeros(len(cache["image_index"]), dtype=torch.bool),
    } for split, cache in (("train", train), ("calibration", calibration))} for name in VARIANTS}
    folds, image_names, image_events = {}, [], []
    for fold_index, event in enumerate(events):
        train_clusters = cluster_event_mask(train, event, equal=False)
        train_images = image_event_mask(train, event, equal=False)
        inner_clusters = cluster_event_mask(calibration, event, equal=False)
        inner_images = image_event_mask(calibration, event, equal=False)
        held_train_clusters, held_train_images = ~train_clusters, ~train_images
        held_cal_clusters, held_cal_images = ~inner_clusters, ~inner_images
        mean = raw_train[train_clusters].mean(0)
        std = raw_train[train_clusters].std(0, unbiased=False).clamp_min(1e-4)
        x_train = ((raw_train - mean) / std).clamp(-8, 8)
        x_cal = ((raw_cal - mean) / std).clamp(-8, 8)
        shared = train_outcome_model(
            x_train, train["direction_code"], train, train_clusters,
            conditioned=False, steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, direction_shrinkage=args.direction_shrinkage,
            seed=args.seed + fold_index * 1009,
        )
        conditioned = train_outcome_model(
            x_train, train["direction_code"], train, train_clusters,
            conditioned=True, steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, direction_shrinkage=args.direction_shrinkage,
            seed=args.seed + fold_index * 1009,
        )
        probabilities = {
            "shared_train": predict_outcomes(shared, x_train, train["direction_code"]),
            "shared_cal": predict_outcomes(shared, x_cal, calibration["direction_code"]),
            "direction_train": predict_outcomes(conditioned, x_train, train["direction_code"]),
            "direction_cal": predict_outcomes(conditioned, x_cal, calibration["direction_code"]),
        }
        prior = neutral_truth_prior(train, train_clusters)
        reference = train["fixed_s2_histograms"][train_images].sum(0)
        direction_delta_train = expected_confusion_delta(
            probabilities["direction_train"], train["direction_code"], train["cluster_area"], prior
        )
        direction_delta_cal = expected_confusion_delta(
            probabilities["direction_cal"], calibration["direction_code"], calibration["cluster_area"], prior
        )
        base_risks = {
            "b1_shared_outcome": (
                simple_outcome_risk(probabilities["shared_train"]),
                simple_outcome_risk(probabilities["shared_cal"]),
            ),
            "b2_direction_outcome": (
                simple_outcome_risk(probabilities["direction_train"]),
                simple_outcome_risk(probabilities["direction_cal"]),
            ),
            "b3_expected_confusion": (
                expected_confusion_risk(direction_delta_train, reference),
                expected_confusion_risk(direction_delta_cal, reference),
            ),
        }
        base_risks["b4_confident_expected_confusion"] = (
            base_risks["b3_expected_confusion"][0] * outcome_confidence(probabilities["direction_train"]),
            base_risks["b3_expected_confusion"][1] * outcome_confidence(probabilities["direction_cal"]),
        )
        fold = {"variants": {}}
        print(f"E2.1 LOEO {fold_index+1}/14 {event}", flush=True)
        for variant in VARIANTS:
            probability_train = probabilities["shared_train"] if variant == "b1_shared_outcome" else probabilities["direction_train"]
            probability_cal = probabilities["shared_cal"] if variant == "b1_shared_outcome" else probabilities["direction_cal"]
            train_risk, cal_risk = base_risks[variant]
            policy, diagnostics = choose_policy(
                calibration, cal_risk, inner_clusters, inner_images,
                maximum_reject_fraction=args.maximum_reject_fraction,
            )
            accept_train = apply_policy(train_risk[held_train_clusters], policy)
            accept_cal = apply_policy(cal_risk[held_cal_clusters], policy)
            train_rows = _held_rows(train, held_train_clusters, held_train_images, accept_train)
            cal_rows = _held_rows(calibration, held_cal_clusters, held_cal_images, accept_cal)
            p3_rows = torch.cat((train_rows[0], cal_rows[0]))
            fixed_rows = torch.cat((train_rows[1], cal_rows[1]))
            selected_rows = torch.cat((train_rows[2], cal_rows[2]))
            fixed_report = metrics_from_histogram(fixed_rows.sum(0))
            selected_report = metrics_from_histogram(selected_rows.sum(0))
            held_probability = torch.cat((probability_train[held_train_clusters], probability_cal[held_cal_clusters]))
            held_targets = torch.cat((outcome_proportions(train)[held_train_clusters], outcome_proportions(calibration)[held_cal_clusters]))
            held_counts = torch.cat((outcome_counts(train)[held_train_clusters], outcome_counts(calibration)[held_cal_clusters]))
            held_accept = torch.cat((accept_train, accept_cal))
            store = stores[variant]
            store["p3"].extend(p3_rows); store["fixed"].extend(fixed_rows); store["selected"].extend(selected_rows)
            store["probabilities"].append(held_probability); store["targets"].append(held_targets); store["counts"].append(held_counts)
            store["accepted"] += int(held_accept.sum()); store["clusters"] += len(held_accept)
            for split, held, probability, accepted in (
                ("train", held_train_clusters, probability_train, accept_train),
                ("calibration", held_cal_clusters, probability_cal, accept_cal),
            ):
                oof[variant][split]["probabilities"][held] = probability[held]
                oof[variant][split]["accepted"][held] = accepted
            fold["variants"][variant] = {
                "policy": policy, "policy_diagnostics": diagnostics,
                "accepted_fraction": float(held_accept.float().mean()),
                "gain_vs_fixed_s2_miou_pp": 100 * (selected_report["valid_miou"] - fixed_report["valid_miou"]),
                "damaged_f1_change_pp": 100 * (selected_report["damaged_f1"] - fixed_report["damaged_f1"]),
            }
            print(
                f"  {variant}: gain={fold['variants'][variant]['gain_vs_fixed_s2_miou_pp']:+.4f}pp "
                f"D-F1={fold['variants'][variant]['damaged_f1_change_pp']:+.4f}pp "
                f"accept={fold['variants'][variant]['accepted_fraction']:.3f}", flush=True,
            )
        folds[event] = fold
        train_ids = torch.nonzero(held_train_images, as_tuple=False).flatten().tolist()
        cal_ids = torch.nonzero(held_cal_images, as_tuple=False).flatten().tolist()
        names = [train["image_names"][i] for i in train_ids] + [calibration["image_names"][i] for i in cal_ids]
        image_names.extend(names); image_events.extend([event] * len(names))
    all_once = (
        len(image_names) == len(train["image_names"]) + len(calibration["image_names"])
        and len(set(image_names)) == len(image_names)
    )
    summaries = {name: _summary(store, image_events, args.bootstrap_iterations, args.seed + 7000 + i)
                 for i, (name, store) in enumerate(stores.items())}
    b1, b2, b3, b4 = (summaries[name] for name in VARIANTS)
    attribution = {
        "direction_reduces_pixel_weighted_mae": b2["outcome_quality"]["pixel_weighted_probability_mae"] < b1["outcome_quality"]["pixel_weighted_probability_mae"],
        "expected_confusion_gain_not_below_direction": b3["gain_vs_fixed_s2_miou_pp"] >= b2["gain_vs_fixed_s2_miou_pp"],
        "confidence_gain_not_below_expected_confusion": b4["gain_vs_fixed_s2_miou_pp"] >= b3["gain_vs_fixed_s2_miou_pp"],
    }
    passed = all_once and b4["passed"] and all(attribution.values())
    decision = "ADMIT_CLUSTER_OUTCOME_E3_FREEZE" if passed else "STOP_CLUSTER_OUTCOME_E21_AFTER_LOEO"
    payload = {
        "protocol": PROTOCOL, "configuration": vars(args), "primary_variant": PRIMARY_VARIANT,
        "variants": summaries, "folds": folds, "attribution_checks": attribution,
        "all_images_predicted_once": all_once, "decision": decision,
        "provenance": {"train_cache_sha256": sha256(args.train_cache),
                       "calibration_cache_sha256": sha256(args.calibration_cache),
                       "e2_json_sha256": sha256(args.e2_json),
                       "audit_json_sha256": sha256(args.audit_json)},
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": OOF_PROTOCOL, "rows": oof, "meta_confirm_accessed": False,
                "val450_accessed": False, "test_accessed": False}, oof_output)
    for variant, row in summaries.items():
        print(f"E2.1 {variant:34s} gain={row['gain_vs_fixed_s2_miou_pp']:+.4f}pp "
              f"D-F1={row['damaged_f1_change_pp']:+.4f}pp events={row['event_miou_nonnegative']}/14 "
              f"harm-AUC={row['outcome_quality']['harmed_majority_auroc']:.4f} passed={row['passed']}")
    print(f"Attribution: {attribution}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
