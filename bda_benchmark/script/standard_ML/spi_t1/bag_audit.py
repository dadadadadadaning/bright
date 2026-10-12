#!/usr/bin/env python3
"""Validation-only MICB Bag versus pixel-output diagnostic.

The audit never trains a model and never accesses the test split.  It keeps
three region sources separate because they answer different questions:

* semantic_cc: label-derived pseudo components (teacher-forced diagnostic);
* official_instance: released building masks (oracle-geometry diagnostic);
* predicted_loc_cc: connected components from the model localization output.

For every region, the MICB Bag decision is compared with the majority raw
pixel decision.  Counterfactual Bag broadcasting is reported only as a
diagnostic; it is not enabled in the P3 model and is not a deployable result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[3]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    BuildingBagSpec,
    DamageFormer_S8_MICB_DSMIL,
    build_training_bag_specs,
    connected_components_4,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (  # noqa: E402
    checkpoint_state,
    fast_histogram,
    file_sha256,
    metrics_from_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (  # noqa: E402
    H9RGBSARInferenceDataset,
    coco_instances,
)
from bda_benchmark.script.standard_ML.spi_stage0.annotations import (  # noqa: E402
    load_stage0_instance_records,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import (  # noqa: E402
    load_semantic_target,
)


SOURCES = ("semantic_cc", "official_instance", "predicted_loc_cc")
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


@dataclass(frozen=True)
class EvaluationBag:
    spec: BuildingBagSpec
    target_label: int
    purity: float = 1.0


def _majority_label(values: torch.Tensor) -> tuple[int, float]:
    values = values[(values >= 0) & (values <= 3)].long().cpu()
    if values.numel() == 0:
        return 0, 0.0
    counts = torch.bincount(values, minlength=4)
    label = int(counts.argmax().item())
    return label, float(counts[label].item() / values.numel())


def semantic_component_bags(
    target: torch.Tensor, minimum_area: int
) -> list[EvaluationBag]:
    specs = build_training_bag_specs(
        (target > 0)[None], target[None], min_area=minimum_area
    )
    return [
        EvaluationBag(spec=spec, target_label=int(spec.label), purity=1.0)
        for spec in specs
    ]


def official_instance_bags(
    instances: Sequence[object],
) -> list[EvaluationBag]:
    output = []
    for instance in instances:
        label = int(instance.label)
        mask = torch.from_numpy(np.asarray(instance.mask, dtype=np.bool_).copy())
        output.append(
            EvaluationBag(
                spec=BuildingBagSpec(
                    batch_index=0,
                    mask=mask,
                    label=label,
                    area=int(mask.sum().item()),
                ),
                target_label=label,
                purity=1.0,
            )
        )
    return output


def predicted_localization_bags(
    localization_prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_area: int,
) -> list[EvaluationBag]:
    output = []
    for component in connected_components_4(localization_prediction > 0):
        area = int(component.sum())
        if area < minimum_area:
            continue
        mask = torch.from_numpy(component.copy())
        target_label, purity = _majority_label(target[mask])
        output.append(
            EvaluationBag(
                spec=BuildingBagSpec(
                    batch_index=0,
                    mask=mask,
                    label=target_label if target_label in (1, 2, 3) else -1,
                    area=area,
                ),
                target_label=target_label,
                purity=purity,
            )
        )
    return output


def new_counts() -> Dict[str, object]:
    return {
        "images": 0,
        "candidate_bags": 0,
        "evaluable_foreground_bags": 0,
        "background_majority_bags": 0,
        "low_purity_bags": 0,
        "bag_confusion": torch.zeros(3, 3, dtype=torch.int64),
        "pixel_region_confusion": torch.zeros(3, 4, dtype=torch.int64),
        "existence_confusion": torch.zeros(2, 2, dtype=torch.int64),
        "severity_confusion": torch.zeros(2, 2, dtype=torch.int64),
        "affected_exist_correct_joint_intact": 0,
        "bag_correct_pixel_wrong": 0,
        "bag_wrong_pixel_correct": 0,
        "d_to_i_regions": 0,
        "d_to_i_regions_bag_recovers": 0,
        "x_to_i_regions": 0,
        "x_to_i_regions_bag_recovers": 0,
        "d_to_i_pixels": 0,
        "d_to_i_pixels_bag_recovers": 0,
        "x_to_i_pixels": 0,
        "x_to_i_pixels_bag_recovers": 0,
        "base_histogram": torch.zeros(4, 4, dtype=torch.float64),
        "broadcast_histogram": torch.zeros(4, 4, dtype=torch.float64),
        "oracle_correct_only_histogram": torch.zeros(4, 4, dtype=torch.float64),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def update_region_counts(
    counts: MutableMapping[str, object],
    *,
    target_label: int,
    pixel_label: int,
    bag_label: int,
    existence_label: int | None = None,
    severity_label: int | None = None,
    semantic_pixels: torch.Tensor,
    pixel_predictions: torch.Tensor,
) -> None:
    """Update Bag/pixel agreement and exact D->I / X->I recoverability."""

    if target_label not in (1, 2, 3):
        return
    counts["evaluable_foreground_bags"] = int(counts["evaluable_foreground_bags"]) + 1
    counts["bag_confusion"][target_label - 1, bag_label - 1] += 1
    counts["pixel_region_confusion"][target_label - 1, pixel_label] += 1
    target_existence = 0 if target_label == 1 else 1
    predicted_existence = (
        (0 if bag_label == 1 else 1)
        if existence_label is None
        else int(existence_label)
    )
    counts["existence_confusion"][target_existence, predicted_existence] += 1
    if target_label in (2, 3):
        predicted_severity = (
            bag_label - 2 if bag_label in (2, 3) else 0
        ) if severity_label is None else int(severity_label)
        counts["severity_confusion"][target_label - 2, predicted_severity] += 1
        if predicted_existence == 1 and bag_label == 1:
            counts["affected_exist_correct_joint_intact"] = int(
                counts["affected_exist_correct_joint_intact"]
            ) + 1
    bag_correct = bag_label == target_label
    pixel_correct = pixel_label == target_label
    if bag_correct and not pixel_correct:
        counts["bag_correct_pixel_wrong"] = int(counts["bag_correct_pixel_wrong"]) + 1
    if not bag_correct and pixel_correct:
        counts["bag_wrong_pixel_correct"] = int(counts["bag_wrong_pixel_correct"]) + 1

    if target_label == 2 and pixel_label == 1:
        counts["d_to_i_regions"] = int(counts["d_to_i_regions"]) + 1
        if bag_label == 2:
            counts["d_to_i_regions_bag_recovers"] = int(
                counts["d_to_i_regions_bag_recovers"]
            ) + 1
    if target_label == 3 and pixel_label == 1:
        counts["x_to_i_regions"] = int(counts["x_to_i_regions"]) + 1
        if bag_label == 3:
            counts["x_to_i_regions_bag_recovers"] = int(
                counts["x_to_i_regions_bag_recovers"]
            ) + 1

    d_to_i = (semantic_pixels == 2) & (pixel_predictions == 1)
    x_to_i = (semantic_pixels == 3) & (pixel_predictions == 1)
    d_count = int(d_to_i.sum().item())
    x_count = int(x_to_i.sum().item())
    counts["d_to_i_pixels"] = int(counts["d_to_i_pixels"]) + d_count
    counts["x_to_i_pixels"] = int(counts["x_to_i_pixels"]) + x_count
    if bag_label == 2:
        counts["d_to_i_pixels_bag_recovers"] = int(
            counts["d_to_i_pixels_bag_recovers"]
        ) + d_count
    if bag_label == 3:
        counts["x_to_i_pixels_bag_recovers"] = int(
            counts["x_to_i_pixels_bag_recovers"]
        ) + x_count


def finalize_counts(counts: Mapping[str, object]) -> Dict[str, object]:
    bag_hist = counts["bag_confusion"].to(torch.float64)
    pixel_hist = counts["pixel_region_confusion"].to(torch.float64)
    bags = int(counts["evaluable_foreground_bags"])
    bag_correct = int(torch.diag(bag_hist).sum().item())
    pixel_correct = int(sum(pixel_hist[index, index + 1] for index in range(3)).item())
    pixel_wrong = bags - pixel_correct
    result = {
        key: int(counts[key])
        for key in (
            "images",
            "candidate_bags",
            "evaluable_foreground_bags",
            "background_majority_bags",
            "low_purity_bags",
            "bag_correct_pixel_wrong",
            "bag_wrong_pixel_correct",
            "affected_exist_correct_joint_intact",
            "d_to_i_regions",
            "d_to_i_regions_bag_recovers",
            "x_to_i_regions",
            "x_to_i_regions_bag_recovers",
            "d_to_i_pixels",
            "d_to_i_pixels_bag_recovers",
            "x_to_i_pixels",
            "x_to_i_pixels_bag_recovers",
        )
    }
    result.update(
        {
            "bag_confusion_intact_damaged_destroyed": bag_hist.long().tolist(),
            "pixel_region_confusion_rows_ix_cols_bidx": pixel_hist.long().tolist(),
            "existence_confusion_rows_true_cols_pred": counts[
                "existence_confusion"
            ].tolist(),
            "severity_confusion_rows_true_dx_cols_pred": counts[
                "severity_confusion"
            ].tolist(),
            "bag_accuracy": _safe_ratio(bag_correct, bags),
            "pixel_region_accuracy": _safe_ratio(pixel_correct, bags),
            "bag_correct_pixel_wrong_fraction_all": _safe_ratio(
                result["bag_correct_pixel_wrong"], bags
            ),
            "bag_correct_given_pixel_wrong": _safe_ratio(
                result["bag_correct_pixel_wrong"], pixel_wrong
            ),
            "bag_harm_given_pixel_correct": _safe_ratio(
                result["bag_wrong_pixel_correct"], pixel_correct
            ),
            "existence_accuracy": _safe_ratio(
                torch.diag(counts["existence_confusion"]).sum().item(), bags
            ),
            "severity_accuracy_given_affected": _safe_ratio(
                torch.diag(counts["severity_confusion"]).sum().item(),
                counts["severity_confusion"].sum().item(),
            ),
            "factorization_reversal_given_affected": _safe_ratio(
                result["affected_exist_correct_joint_intact"],
                counts["severity_confusion"].sum().item(),
            ),
            "d_to_i_region_recovery_ceiling": _safe_ratio(
                result["d_to_i_regions_bag_recovers"], result["d_to_i_regions"]
            ),
            "x_to_i_region_recovery_ceiling": _safe_ratio(
                result["x_to_i_regions_bag_recovers"], result["x_to_i_regions"]
            ),
            "d_to_i_pixel_recovery_ceiling": _safe_ratio(
                result["d_to_i_pixels_bag_recovers"], result["d_to_i_pixels"]
            ),
            "x_to_i_pixel_recovery_ceiling": _safe_ratio(
                result["x_to_i_pixels_bag_recovers"], result["x_to_i_pixels"]
            ),
            "raw_pixel_semantic": metrics_from_histogram(counts["base_histogram"]),
            "bag_broadcast_semantic": metrics_from_histogram(
                counts["broadcast_histogram"]
            ),
            "oracle_correct_bag_only_semantic": metrics_from_histogram(
                counts["oracle_correct_only_histogram"]
            ),
        }
    )
    return result


def diagnose_sources(reports: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Produce transparent diagnostic flags, never a model-selection decision."""

    pseudo = reports["semantic_cc"]
    official = reports["official_instance"]
    deploy = reports["predicted_loc_cc"]
    findings = []
    if float(pseudo["bag_accuracy"]) - float(official["bag_accuracy"]) >= 0.05:
        findings.append("PSEUDO_COMPONENT_TO_TRUE_INSTANCE_DOMAIN_GAP")
    if float(official["bag_accuracy"]) - float(deploy["bag_accuracy"]) >= 0.05:
        findings.append("PREDICTED_LOCALIZATION_GEOMETRY_LIMITS_BAG_ACCURACY")
    if float(deploy["factorization_reversal_given_affected"]) >= 0.05:
        findings.append("HIERARCHICAL_FACTORIZATION_OFTEN_REVERSES_AFFECTED_TO_INTACT")
    if float(deploy["bag_correct_given_pixel_wrong"]) >= 0.20:
        findings.append("MICB_CONTAINS_COMPLEMENTARY_EVIDENCE_WHEN_PIXELS_ARE_WRONG")
    raw = float(deploy["raw_pixel_semantic"]["valid_miou"])
    broadcast = float(deploy["bag_broadcast_semantic"]["valid_miou"])
    if float(deploy["bag_correct_given_pixel_wrong"]) >= 0.20 and broadcast <= raw:
        findings.append("BAG_EVIDENCE_EXISTS_BUT_UNCONDITIONAL_BROADCAST_IS_HARMFUL")
    if broadcast > raw:
        findings.append("PREDICTED_BAG_BROADCAST_HAS_POSITIVE_DIAGNOSTIC_NET_GAIN")
    background_fraction = _safe_ratio(
        deploy["background_majority_bags"], deploy["candidate_bags"]
    )
    if background_fraction >= 0.10:
        findings.append("LOCALIZATION_COMPONENTS_CONTAIN_MANY_BACKGROUND_MAJORITY_REGIONS")
    if not findings:
        findings.append("NO_SINGLE_DOMINANT_MICB_FAILURE_AT_DIAGNOSTIC_THRESHOLDS")
    return {
        "thresholds_are_diagnostic_not_selection_rules": True,
        "findings": findings,
        "pseudo_minus_official_bag_accuracy_pp": 100.0
        * (float(pseudo["bag_accuracy"]) - float(official["bag_accuracy"])),
        "official_minus_predicted_bag_accuracy_pp": 100.0
        * (float(official["bag_accuracy"]) - float(deploy["bag_accuracy"])),
        "predicted_broadcast_valid_miou_gain_pp": 100.0 * (broadcast - raw),
        "predicted_background_majority_fraction": background_fraction,
    }


def _prepare_head_inputs(
    model: DamageFormer_S8_MICB_DSMIL,
    auxiliary: Mapping[str, object],
    bags: Sequence[EvaluationBag],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    specs = [bag.spec for bag in bags]
    if not specs:
        device = auxiliary["fused_roi"].device
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_float, empty_long, empty_long
    result = model.micb_dsmil(
        rgb_feature=auxiliary["rgb_feature"],
        sar_feature=auxiliary["sar_feature"],
        fused_feature=auxiliary["fused_roi"],
        bag_specs=specs,
    )
    if int(result["bag_count"]) != len(specs):
        raise RuntimeError(
            "one or more non-empty audit bags vanished at feature resolution: "
            f"requested={len(specs)}, produced={result['bag_count']}"
        )
    probabilities = result["log_probs3"].float().exp()
    return (
        probabilities.argmax(1) + 1,
        probabilities.amax(1),
        result["exist_logits"].argmax(1),
        result["severity_logits"].argmax(1),
    )


def evaluate_source_image(
    count_tables: Sequence[MutableMapping[str, object]],
    *,
    model: DamageFormer_S8_MICB_DSMIL,
    auxiliary: Mapping[str, object],
    bags: Sequence[EvaluationBag],
    target: torch.Tensor,
    base_prediction: torch.Tensor,
    minimum_purity: float,
) -> None:
    bag_predictions, confidences, existence_predictions, severity_predictions = (
        _prepare_head_inputs(model, auxiliary, bags)
    )
    corrected = base_prediction.clone()
    oracle_corrected = base_prediction.clone()
    best_confidence = torch.full_like(base_prediction, -torch.inf, dtype=torch.float32)
    oracle_confidence = torch.full_like(base_prediction, -torch.inf, dtype=torch.float32)
    for counts in count_tables:
        counts["images"] = int(counts["images"]) + 1
        counts["candidate_bags"] = int(counts["candidate_bags"]) + len(bags)

    for index, bag in enumerate(bags):
        mask = bag.spec.mask.detach().cpu().bool()
        target_label = int(bag.target_label)
        for counts in count_tables:
            if target_label == 0:
                counts["background_majority_bags"] = int(
                    counts["background_majority_bags"]
                ) + 1
            if float(bag.purity) < minimum_purity:
                counts["low_purity_bags"] = int(counts["low_purity_bags"]) + 1
        bag_label = int(bag_predictions[index].item())
        confidence = float(confidences[index].item())
        pixel_label, _ = _majority_label(base_prediction[mask])
        if target_label in (1, 2, 3) and float(bag.purity) >= minimum_purity:
            for counts in count_tables:
                update_region_counts(
                    counts,
                    target_label=target_label,
                    pixel_label=pixel_label,
                    bag_label=bag_label,
                    existence_label=int(existence_predictions[index].item()),
                    severity_label=int(severity_predictions[index].item()),
                    semantic_pixels=target[mask],
                    pixel_predictions=base_prediction[mask],
                )

        update = mask & (confidence > best_confidence)
        corrected[update] = bag_label
        best_confidence[update] = confidence
        if bag_label == target_label and target_label in (1, 2, 3):
            oracle_update = mask & (confidence > oracle_confidence)
            oracle_corrected[oracle_update] = bag_label
            oracle_confidence[oracle_update] = confidence

    for counts in count_tables:
        counts["base_histogram"] += fast_histogram(base_prediction, target)
        counts["broadcast_histogram"] += fast_histogram(corrected, target)
        counts["oracle_correct_only_histogram"] += fast_histogram(
            oracle_corrected, target
        )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("MICB Bag versus raw-pixel validation audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    annotations = parser.add_mutually_exclusive_group(required=True)
    annotations.add_argument("--instance_coco_json", default="")
    annotations.add_argument("--instance_json_dir", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--minimum_area", type=int, default=32)
    parser.add_argument("--minimum_purity", type=float, default=0.5)
    parser.add_argument("--expected_images", type=int, default=697)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.minimum_area < 1:
        raise ValueError("minimum_area must be positive")
    if not 0.0 <= args.minimum_purity <= 1.0:
        raise ValueError("minimum_purity must be between zero and one")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatible}")
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False)

    dataset = H9RGBSARInferenceDataset(
        args.dataset_path, args.val_list, args.sar_stats_json
    )
    if args.expected_images > 0 and len(dataset) != args.expected_images:
        raise RuntimeError(
            f"validation split mismatch: expected={args.expected_images}, observed={len(dataset)}"
        )
    records, annotation_manifest = load_stage0_instance_records(
        dataset.data_list,
        unified_coco_json=args.instance_coco_json,
        per_image_json_dir=args.instance_json_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    overall = {source: new_counts() for source in SOURCES}
    events: Dict[str, Dict[str, Dict[str, object]]] = {}
    processed = 0
    started = time.perf_counter()
    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        name = str(names[0])
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        output_loc, output_damage, auxiliary = model(
            rgb, p3_sar_view(sar), return_aux=True
        )
        base_prediction = output_damage[0].float().argmax(0).cpu()
        localization_prediction = output_loc[0].float().argmax(0).cpu()
        target = load_semantic_target(args.dataset_path, name)
        record = records[name]
        instance_bags = official_instance_bags(
            coco_instances(record, minimum_area=args.minimum_area)
        )
        source_bags = {
            "semantic_cc": semantic_component_bags(target, args.minimum_area),
            "official_instance": instance_bags,
            "predicted_loc_cc": predicted_localization_bags(
                localization_prediction, target, args.minimum_area
            ),
        }
        event = name.rsplit("_", 1)[0]
        if event not in events:
            events[event] = {source: new_counts() for source in SOURCES}
        for source in SOURCES:
            evaluate_source_image(
                (overall[source], events[event][source]),
                model=model,
                auxiliary=auxiliary,
                bags=source_bags[source],
                target=target,
                base_prediction=base_prediction,
                minimum_purity=args.minimum_purity,
            )
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"MICB Bag audit: {processed}/{len(dataset)} | "
                f"elapsed={elapsed:.1f}s | rate={processed/max(elapsed, 1e-9):.2f} img/s",
                flush=True,
            )

    overall_reports = {source: finalize_counts(overall[source]) for source in SOURCES}
    payload = {
        "protocol": "SPI_MICB_BAG_PIXEL_FUNNEL_VALIDATION_v1",
        "split": "cvprw26_public_val",
        "training": "NONE",
        "test_accessed": False,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": file_sha256(args.checkpoint),
            "step": int(checkpoint.get("step", 0)),
            "mil_residual_scale": float(model.mil_residual_scale),
        },
        "data": {
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "val_list": str(Path(args.val_list).resolve()),
            "processed_images": processed,
            "annotation_manifest": annotation_manifest,
        },
        "region_source_contract": {
            "semantic_cc": "validation semantic labels define class-wise CC; teacher-forced, not deployable",
            "official_instance": "official instance masks; oracle geometry, not deployable",
            "predicted_loc_cc": "model localization connected components; deployable geometry",
            "bag_broadcast": "counterfactual diagnostic only; P3 inference remains unchanged",
            "oracle_correct_only": "uses target correctness as a gate; upper bound only",
        },
        "settings": {
            "minimum_area": args.minimum_area,
            "minimum_purity": args.minimum_purity,
        },
        "overall": overall_reports,
        "diagnosis": diagnose_sources(overall_reports),
        "events": {
            event: {source: finalize_counts(table[source]) for source in SOURCES}
            for event, table in sorted(events.items())
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 116)
    print("MICB Bag vs raw pixel | semantic TIFF primary | validation only | no training")
    print(
        f"{'Region source':<22} {'Bags':>8} {'BagAcc':>9} {'PixelAcc':>10} "
        f"{'BagOK|PixBad':>13} {'D->I rec':>10} {'X->I rec':>10} {'Broadcast mIoU':>15}"
    )
    for source in SOURCES:
        row = payload["overall"][source]
        print(
            f"{source:<22} {row['evaluable_foreground_bags']:8d} "
            f"{100*row['bag_accuracy']:9.3f} {100*row['pixel_region_accuracy']:10.3f} "
            f"{100*row['bag_correct_given_pixel_wrong']:13.3f} "
            f"{100*row['d_to_i_region_recovery_ceiling']:10.3f} "
            f"{100*row['x_to_i_region_recovery_ceiling']:10.3f} "
            f"{100*row['bag_broadcast_semantic']['valid_miou']:15.3f}"
        )
    print(f"Saved: {output}")
    print("Findings: " + ", ".join(payload["diagnosis"]["findings"]))
    print("Training/test: NONE / NEVER")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
