#!/usr/bin/env python3
"""A0-A3 deployable region-feedback audit on Standard-ML semantic Val450."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (
    BuildingBagSpec,
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    checkpoint_state,
    fast_histogram,
    file_sha256,
    metrics_from_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (
    H9RGBSARInferenceDataset,
)
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    VARIANTS,
    broadcast_labels,
    constant_label_region_audit,
    erode_region,
    foreground_majority_label,
    gated_majority_feedback,
    mean_probability_label,
    prediction_change_audit,
    predicted_building_regions,
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target


def micb_labels(
    model: DamageFormer_S8_MICB_DSMIL,
    auxiliary: Mapping[str, object],
    regions: Sequence[torch.Tensor],
) -> list[int]:
    if not regions:
        return []
    specs = [
        BuildingBagSpec(batch_index=0, mask=mask, label=-1, area=int(mask.sum()))
        for mask in regions
    ]
    result = model.micb_dsmil(
        rgb_feature=auxiliary["rgb_feature"],
        sar_feature=auxiliary["sar_feature"],
        fused_feature=auxiliary["fused_roi"],
        bag_specs=specs,
    )
    if int(result["bag_count"]) != len(regions):
        raise RuntimeError(
            "one or more predicted regions vanished at MICB feature resolution: "
            f"requested={len(regions)}, produced={result['bag_count']}"
        )
    return (result["log_probs3"].float().argmax(dim=1) + 1).cpu().tolist()


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("P3 deployable region feedback Val450 audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--minimum_area", type=int, default=32)
    parser.add_argument("--gate_minimum_purity", type=float, default=0.70)
    parser.add_argument("--gate_minimum_margin", type=float, default=0.10)
    parser.add_argument(
        "--gate_maximum_area",
        type=int,
        default=65536,
        help="0 disables the maximum-area gate",
    )
    parser.add_argument("--core_radius", type=int, default=2)
    parser.add_argument("--expected_images", type=int, default=450)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_miou", type=float, default=0.605020)
    parser.add_argument("--reproduction_tolerance", type=float, default=0.00001)
    args = parser.parse_args(argv)

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
            f"Standard-ML validation mismatch: expected={args.expected_images}, "
            f"observed={len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    totals = {variant: torch.zeros((4, 4), dtype=torch.float64) for variant in VARIANTS}
    events: dict[str, dict[str, torch.Tensor]] = {}
    image_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    change_totals = {
        variant: {
            "changed_pixels": 0,
            "corrected_pixels": 0,
            "harmed_pixels": 0,
            "net_gain_pixels": 0,
            "raw_to_updated_transition": torch.zeros((4, 4), dtype=torch.long),
        }
        for variant in VARIANTS
        if variant != "a0_raw_p3"
    }
    region_count = 0
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
        probabilities = output_damage[0].float().softmax(dim=0).cpu()
        raw = probabilities.argmax(dim=0)
        localization = output_loc[0].float().argmax(dim=0).cpu()
        target = load_semantic_target(args.dataset_path, name)
        regions = predicted_building_regions(localization, args.minimum_area)
        region_count += len(regions)

        majority = [
            foreground_majority_label(raw, probabilities, mask) for mask in regions
        ]
        mean_probability = [
            mean_probability_label(probabilities, mask) for mask in regions
        ]
        bag = micb_labels(model, auxiliary, regions)
        gated, gate_records = gated_majority_feedback(
            raw,
            probabilities,
            regions,
            minimum_purity=args.gate_minimum_purity,
            minimum_margin=args.gate_minimum_margin,
            maximum_area=args.gate_maximum_area,
            core_radius=0,
        )
        gated_core, core_records = gated_majority_feedback(
            raw,
            probabilities,
            regions,
            minimum_purity=args.gate_minimum_purity,
            minimum_margin=args.gate_minimum_margin,
            maximum_area=args.gate_maximum_area,
            core_radius=args.core_radius,
        )
        predictions = {
            "a0_raw_p3": raw,
            "a1_majority": broadcast_labels(raw, regions, majority),
            "a2_mean_probability": broadcast_labels(raw, regions, mean_probability),
            "a3_micb": broadcast_labels(raw, regions, bag),
            "a4_gated_majority": gated,
            "a5_gated_core_majority": gated_core,
        }
        event = name.rsplit("_", 1)[0]
        event_table = events.setdefault(
            event,
            {variant: torch.zeros((4, 4), dtype=torch.float64) for variant in VARIANTS},
        )
        row_histograms = {}
        for variant, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[variant] += histogram
            event_table[variant] += histogram
            row_histograms[variant] = histogram.long().tolist()
            if variant != "a0_raw_p3":
                audit = prediction_change_audit(raw, prediction, target)
                accumulator = change_totals[variant]
                for key in (
                    "changed_pixels",
                    "corrected_pixels",
                    "harmed_pixels",
                    "net_gain_pixels",
                ):
                    accumulator[key] += int(audit[key])
                accumulator["raw_to_updated_transition"] += torch.as_tensor(
                    audit["raw_to_updated_transition"], dtype=torch.long
                )
        for region_index, mask in enumerate(regions):
            empty_mask = torch.zeros_like(mask)
            a4_write_mask = (
                mask if bool(gate_records[region_index]["applied"]) else empty_mask
            )
            a5_write_mask = (
                erode_region(mask, args.core_radius)
                if bool(core_records[region_index]["applied"])
                else empty_mask
            )
            valid_region_target = target[mask]
            valid_region_target = valid_region_target[
                (valid_region_target >= 0) & (valid_region_target <= 3)
            ]
            target_counts = torch.bincount(valid_region_target.long(), minlength=4)
            region_rows.append(
                {
                    "name": name,
                    "event": event,
                    **gate_records[region_index],
                    "core_write_area": int(core_records[region_index]["write_area"]),
                    "target_class_pixels": target_counts.tolist(),
                    "a1_change": constant_label_region_audit(
                        raw, target, mask, mask, majority[region_index]
                    ),
                    "a4_change": constant_label_region_audit(
                        raw, target, mask, a4_write_mask, majority[region_index]
                    ),
                    "a5_change": constant_label_region_audit(
                        raw, target, mask, a5_write_mask, majority[region_index]
                    ),
                }
            )
        image_rows.append(
            {"name": name, "event": event, "regions": len(regions), "histograms": row_histograms}
        )
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"SPI region feedback: {processed}/{len(dataset)} | "
                f"elapsed={elapsed:.1f}s | rate={processed/max(elapsed, 1e-9):.2f} img/s",
                flush=True,
            )

    reports = {variant: metrics_from_histogram(totals[variant]) for variant in VARIANTS}
    raw_miou = reports["a0_raw_p3"]["valid_miou"]
    reproduction_error = 100.0 * (raw_miou - args.expected_p3_miou)
    full_audit = args.max_images <= 0 and processed == args.expected_images
    bootstrap = (
        stratified_paired_bootstrap(
            image_rows,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
            comparison_pairs=(
                ("a1_majority", "a2_mean_probability"),
                ("a4_gated_majority", "a1_majority"),
                ("a5_gated_core_majority", "a1_majority"),
                ("a5_gated_core_majority", "a4_gated_majority"),
            ),
        )
        if full_audit
        else None
    )
    event_reports = {
        event: {
            variant: metrics_from_histogram(table[variant])
            for variant in VARIANTS
        }
        for event, table in sorted(events.items())
    }
    event_deltas = {
        variant: {
            "improved": sum(
                event_reports[event][variant]["valid_miou"]
                > event_reports[event]["a0_raw_p3"]["valid_miou"]
                for event in event_reports
            ),
            "nonnegative": sum(
                event_reports[event][variant]["valid_miou"]
                >= event_reports[event]["a0_raw_p3"]["valid_miou"]
                for event in event_reports
            ),
            "total_events": len(event_reports),
        }
        for variant in VARIANTS
        if variant != "a0_raw_p3"
    }
    serializable_changes = {}
    for variant, values in change_totals.items():
        changed = int(values["changed_pixels"])
        corrected = int(values["corrected_pixels"])
        harmed = int(values["harmed_pixels"])
        serializable_changes[variant] = {
            "changed_pixels": changed,
            "corrected_pixels": corrected,
            "harmed_pixels": harmed,
            "net_gain_pixels": int(values["net_gain_pixels"]),
            "precision_of_change": corrected / changed if changed else 0.0,
            "benefit_harm_ratio": corrected / harmed if harmed else None,
            "raw_to_updated_transition": values[
                "raw_to_updated_transition"
            ].tolist(),
        }
    decisions = {}
    if bootstrap is not None:
        minimum_nonnegative_events = math.ceil(7.0 * len(event_reports) / 11.0)
        for variant in (
            "a1_majority",
            "a4_gated_majority",
            "a5_gated_core_majority",
        ):
            report = reports[variant]
            ci = bootstrap["deltas_vs_a0"][variant]
            checks = {
                "p3_reproduction": abs(raw_miou - args.expected_p3_miou)
                <= args.reproduction_tolerance,
                "valid_miou_gain_ge_0_5pp": 100.0
                * (report["valid_miou"] - raw_miou)
                >= 0.5,
                "damaged_f1_gain_ge_1_0pp": 100.0
                * (report["damaged_f1"] - reports["a0_raw_p3"]["damaged_f1"])
                >= 1.0,
                "intact_iou_drop_le_0_5pp": 100.0
                * (report["intact_iou"] - reports["a0_raw_p3"]["intact_iou"])
                >= -0.5,
                "destroyed_iou_drop_le_0_5pp": 100.0
                * (report["destroyed_iou"] - reports["a0_raw_p3"]["destroyed_iou"])
                >= -0.5,
                f"event_nonnegative_ge_{minimum_nonnegative_events}_of_{len(event_reports)}":
                event_deltas[variant]["nonnegative"] >= minimum_nonnegative_events,
                "bootstrap_ci95_low_gt_zero": ci["ci95_low"] > 0.0,
            }
            decisions[variant] = {
                "checks": checks,
                "all_pass": all(checks.values()),
            }
    payload = {
        "protocol": "SPI_P3_DEPLOYABLE_REGION_FEEDBACK_STANDARD_ML_VAL450_v2",
        "split": "standard_ml_val450",
        "training": "NONE",
        "official_instances_accessed": False,
        "original_standard_test_accessed": False,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": file_sha256(args.checkpoint),
            "step": int(checkpoint.get("step", 0)),
            "mil_residual_scale": float(model.mil_residual_scale),
        },
        "settings": vars(args),
        "processed_images": processed,
        "predicted_region_count": region_count,
        "reports": reports,
        "event_reports": event_reports,
        "event_win_counts_vs_a0": event_deltas,
        "paired_bootstrap": bootstrap,
        "change_audit_vs_a0": serializable_changes,
        "region_audit": region_rows,
        "preregistered_decisions": decisions,
        "p3_reproduction": {
            "expected_valid_miou": args.expected_p3_miou,
            "observed_valid_miou": raw_miou,
            "error_pp": reproduction_error,
            "passed": abs(raw_miou - args.expected_p3_miou)
            <= args.reproduction_tolerance,
        },
        "contract": {
            "region_construction": "P3 localization connected components only",
            "semantic_ground_truth": "metrics only",
            "a1": "foreground pixel argmax majority broadcast",
            "a2": "foreground mean probability broadcast",
            "a3": "MICB bag prediction broadcast; counterfactual because residual scale is zero",
            "a4": "A1 majority only when prediction-only purity/margin/area gate passes",
            "a5": "A4 gate with majority written only to an eroded region core",
            "ground_truth_usage": "semantic TIFF is read after prediction for metrics/audit only",
        },
        "image_histograms": image_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 112)
    print("P3 deployable region feedback | Standard-ML Val450 | semantic TIFF only | no training")
    print(f"{'Variant':<24} {'mIoU':>10} {'I-IoU':>10} {'D-IoU':>10} {'X-IoU':>10} {'D-F1':>10} {'D->I':>10} {'X->I':>10}")
    for variant in VARIANTS:
        row = reports[variant]
        print(
            f"{variant:<24} {100*row['valid_miou']:10.4f} "
            f"{100*row['intact_iou']:10.4f} {100*row['damaged_iou']:10.4f} "
            f"{100*row['destroyed_iou']:10.4f} {100*row['damaged_f1']:10.4f} "
            f"{100*row['damaged_to_intact']:10.4f} "
            f"{100*row['destroyed_to_intact']:10.4f}"
        )
    print(
        "P3 reproduction: "
        f"expected={100*args.expected_p3_miou:.4f}% "
        f"observed={100*raw_miou:.4f}% error={reproduction_error:+.4f} pp "
        f"passed={payload['p3_reproduction']['passed']}"
    )
    if bootstrap is not None:
        for variant, row in bootstrap["deltas_vs_a0"].items():
            print(
                f"Bootstrap {variant}-A0: mean={row['mean']:+.4f} pp "
                f"95%CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )
        for name, row in bootstrap["comparisons"].items():
            print(
                f"Bootstrap {name}: mean={row['mean']:+.4f} pp "
                f"95%CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )
    for variant in ("a1_majority", "a4_gated_majority", "a5_gated_core_majority"):
        change = serializable_changes[variant]
        wins = event_deltas[variant]
        print(
            f"Audit {variant}: changed={change['changed_pixels']} "
            f"corrected={change['corrected_pixels']} harmed={change['harmed_pixels']} "
            f"net={change['net_gain_pixels']:+d} "
            f"events_nonnegative={wins['nonnegative']}/{wins['total_events']}"
        )
        if variant in decisions:
            print(
                f"Decision {variant}: "
                f"{'PASS' if decisions[variant]['all_pass'] else 'INCONCLUSIVE_OR_FAIL'} "
                f"{decisions[variant]['checks']}"
            )
    print(f"Saved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
