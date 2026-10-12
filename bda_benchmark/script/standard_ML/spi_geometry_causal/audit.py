#!/usr/bin/env python3
"""Locked-A6 causal audit: local smoothing vs LocCC vs optional RGID-G3."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (
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
from bda_benchmark.script.standard_ML.spi_geometry_causal.core import (
    BASE_VARIANTS,
    RGID_VARIANT,
    event_macro_paired_bootstrap,
    instance_labels_to_regions,
    local_window_gated_majority,
)
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    gated_majority_feedback,
    predicted_building_regions,
    prediction_change_audit,
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target


def _load_rgid(checkpoint_path: str, device: torch.device):
    experiment_root = (
        Path(__file__).resolve().parents[3] / "experiments" / "rgid_s9ic_clean"
    )
    if str(experiment_root) not in sys.path:
        sys.path.insert(0, str(experiment_root))
    from rgid.evaluation import load_head_checkpoint
    from rgid.geometry import RGIDDecoder

    head, config, checkpoint = load_head_checkpoint(checkpoint_path, device=device)
    calibration = checkpoint.get("boundary_calibration")
    if not isinstance(calibration, Mapping) or "boundary_gate_threshold" not in calibration:
        raise RuntimeError(
            "RGID-G3 requires a checkpoint calibrated on training data with "
            "boundary_calibration.boundary_gate_threshold"
        )
    threshold = float(calibration["boundary_gate_threshold"])
    return head.eval().requires_grad_(False), RGIDDecoder(config), checkpoint, threshold


def _merge_decode_stats(total: dict[str, object], row: Mapping[str, object]) -> None:
    for key, value in row.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, Mapping):
            target = total.setdefault(key, {})
            for sub_key, sub_value in value.items():
                target[str(sub_key)] = target.get(str(sub_key), 0) + sub_value


def _event_win_counts(event_reports, variants, baseline):
    return {
        variant: {
            "improved": sum(
                table[variant]["valid_miou"] > table[baseline]["valid_miou"]
                for table in event_reports.values()
            ),
            "nonnegative": sum(
                table[variant]["valid_miou"] >= table[baseline]["valid_miou"]
                for table in event_reports.values()
            ),
            "total_events": len(event_reports),
        }
        for variant in variants
        if variant != baseline
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SPI locked-A6 geometry causal audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--rgid_checkpoint", default="")
    parser.add_argument("--rgid_provenance_json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--minimum_area", type=int, default=32)
    parser.add_argument("--local_kernel_size", type=int, default=5)
    parser.add_argument("--a6_minimum_purity", type=float, default=0.70)
    parser.add_argument("--a6_minimum_margin", type=float, default=0.10)
    parser.add_argument("--expected_images", type=int, default=450)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_miou", type=float, default=0.605020)
    parser.add_argument("--reproduction_tolerance", type=float, default=0.00001)
    args = parser.parse_args(argv)

    if args.local_kernel_size != 5:
        raise RuntimeError("local kernel is protocol-locked to 5; do not tune on Val450")
    if args.a6_minimum_purity != 0.70 or args.a6_minimum_margin != 0.10:
        raise RuntimeError("A6 thresholds are locked to purity=0.70 and margin=0.10")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("P3 checkpoint must be a mapping")
    p3 = DamageFormer_S8_MICB_DSMIL()
    incompatible = p3.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.to(device).eval().requires_grad_(False)

    if bool(args.rgid_checkpoint) != bool(args.rgid_provenance_json):
        raise RuntimeError(
            "provide rgid_checkpoint and rgid_provenance_json together"
        )
    rgid_provenance = None
    rgid = None
    if args.rgid_checkpoint:
        rgid_provenance = json.loads(
            Path(args.rgid_provenance_json).read_text(encoding="utf-8")
        )
        if not isinstance(rgid_provenance, Mapping):
            raise TypeError("RGID provenance JSON must contain one JSON object")
        required_provenance = {
            "protocol": "RGID_PSEUDO_ONLY_TRAIN_v1",
            "official_instances_accessed": False,
            "standard_ml_val_accessed": False,
            "original_standard_test_accessed": False,
            "calibration_split": "train_only",
        }
        for key, expected in required_provenance.items():
            if rgid_provenance.get(key) != expected:
                raise RuntimeError(
                    f"RGID provenance violation for {key}: "
                    f"expected={expected!r}, observed={rgid_provenance.get(key)!r}"
                )
        observed_p3_hash = file_sha256(args.checkpoint)
        if rgid_provenance.get("p3_sha256") != observed_p3_hash:
            raise RuntimeError("RGID and semantic audit use different frozen P3 checkpoints")
        rgid = _load_rgid(args.rgid_checkpoint, device)
    variants = list(BASE_VARIANTS) + ([RGID_VARIANT] if rgid else [])
    comparison_pairs = [("c2_loccc_a6", "c1_local_window_a6")]
    if rgid:
        comparison_pairs.append((RGID_VARIANT, "c2_loccc_a6"))

    dataset = H9RGBSARInferenceDataset(
        args.dataset_path, args.val_list, args.sar_stats_json
    )
    if args.expected_images > 0 and len(dataset) != args.expected_images:
        raise RuntimeError(
            f"split mismatch: expected={args.expected_images}, observed={len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    totals = {
        variant: torch.zeros((4, 4), dtype=torch.float64) for variant in variants
    }
    events: dict[str, dict[str, torch.Tensor]] = {}
    image_rows: list[dict[str, object]] = []
    changes = {
        variant: {"changed": 0, "corrected": 0, "harmed": 0, "net": 0}
        for variant in variants
        if variant != "c0_raw_p3"
    }
    region_totals = {"loccc": 0, "rgid": 0}
    local_totals = {"accepted_pixels": 0, "changed_pixels": 0}
    decode_totals: dict[str, object] = {}
    processed = 0
    started = time.perf_counter()

    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        name = str(names[0])
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        loc_logits, damage_logits, auxiliary = p3(
            rgb, p3_sar_view(sar), return_aux=True
        )
        probabilities = damage_logits[0].float().softmax(dim=0).cpu()
        raw = probabilities.argmax(dim=0)
        localization = loc_logits[0].float().argmax(dim=0).cpu()
        building_mask = localization > 0

        loccc_regions = predicted_building_regions(localization, args.minimum_area)
        region_totals["loccc"] += len(loccc_regions)
        eligible_building_mask = torch.zeros_like(building_mask)
        for region in loccc_regions:
            eligible_building_mask |= region

        local_prediction, local_report = local_window_gated_majority(
            raw,
            probabilities,
            eligible_building_mask,
            kernel_size=args.local_kernel_size,
            minimum_purity=args.a6_minimum_purity,
            minimum_margin=args.a6_minimum_margin,
        )
        for key in local_totals:
            local_totals[key] += int(local_report[key])

        loccc_prediction, _ = gated_majority_feedback(
            raw,
            probabilities,
            loccc_regions,
            minimum_purity=args.a6_minimum_purity,
            minimum_margin=args.a6_minimum_margin,
            maximum_area=0,
            core_radius=0,
        )
        predictions = {
            "c0_raw_p3": raw,
            "c1_local_window_a6": local_prediction,
            "c2_loccc_a6": loccc_prediction,
        }

        if rgid:
            head, decoder, _, boundary_threshold = rgid
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                geometry_output = head(auxiliary["fused_roi"].float())
            decoded = decoder.decode(
                geometry_output["center_logits"][0, 0],
                geometry_output["boundary_logits"][0, 0],
                building_mask,
                mode="g3",
                inputs_are_logits=True,
                boundary_gate_threshold=boundary_threshold,
            )
            # C2 and C3 must cover exactly the same eligible P3 building support.
            # Keep tiny RGID children when their parent passed the C2 area rule;
            # otherwise dropping them would confound splitting with coverage.
            rgid_labels = torch.as_tensor(decoded.instance_labels).long().cpu()
            rgid_labels[~eligible_building_mask] = 0
            rgid_regions = instance_labels_to_regions(
                rgid_labels, minimum_area=1
            )
            rgid_support = torch.zeros_like(eligible_building_mask)
            for region in rgid_regions:
                rgid_support |= region
            if not torch.equal(rgid_support, eligible_building_mask):
                missing = int((eligible_building_mask & ~rgid_support).sum().item())
                extra = int((rgid_support & ~eligible_building_mask).sum().item())
                raise RuntimeError(
                    "RGID support mismatch against LocCC control: "
                    f"missing={missing}, extra={extra}"
                )
            region_totals["rgid"] += len(rgid_regions)
            rgid_prediction, _ = gated_majority_feedback(
                raw,
                probabilities,
                rgid_regions,
                minimum_purity=args.a6_minimum_purity,
                minimum_margin=args.a6_minimum_margin,
                maximum_area=0,
                core_radius=0,
            )
            predictions[RGID_VARIANT] = rgid_prediction
            _merge_decode_stats(decode_totals, decoded.stats.to_dict())

        # GT is intentionally loaded only after all deployable predictions exist.
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        event_table = events.setdefault(
            event,
            {
                variant: torch.zeros((4, 4), dtype=torch.float64)
                for variant in variants
            },
        )
        histograms = {}
        for variant, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[variant] += histogram
            event_table[variant] += histogram
            histograms[variant] = histogram.long().tolist()
            if variant != "c0_raw_p3":
                row = prediction_change_audit(raw, prediction, target)
                changes[variant]["changed"] += int(row["changed_pixels"])
                changes[variant]["corrected"] += int(row["corrected_pixels"])
                changes[variant]["harmed"] += int(row["harmed_pixels"])
                changes[variant]["net"] += int(row["net_gain_pixels"])
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"SPI geometry causal: {processed}/{len(dataset)} | "
                f"elapsed={elapsed:.1f}s | rate={processed/max(elapsed, 1e-9):.2f} img/s",
                flush=True,
            )

    reports = {
        variant: metrics_from_histogram(totals[variant]) for variant in variants
    }
    event_reports = {
        event: {
            variant: metrics_from_histogram(table[variant])
            for variant in variants
        }
        for event, table in sorted(events.items())
    }
    event_wins = _event_win_counts(event_reports, variants, "c0_raw_p3")
    full = args.max_images <= 0 and processed == args.expected_images
    stratified = (
        stratified_paired_bootstrap(
            image_rows,
            variants=variants,
            baseline="c0_raw_p3",
            iterations=args.bootstrap_iterations,
            seed=args.seed,
            comparison_pairs=comparison_pairs,
        )
        if full
        else None
    )
    event_macro = (
        event_macro_paired_bootstrap(
            image_rows,
            variants=variants,
            baseline="c0_raw_p3",
            comparison_pairs=comparison_pairs,
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        )
        if full
        else None
    )

    raw_miou = reports["c0_raw_p3"]["valid_miou"]
    reproduction = {
        "expected_valid_miou": args.expected_p3_miou,
        "observed_valid_miou": raw_miou,
        "error_pp": 100.0 * (raw_miou - args.expected_p3_miou),
        "passed": abs(raw_miou - args.expected_p3_miou)
        <= args.reproduction_tolerance,
    }
    minimum_events = math.ceil(7.0 * len(event_reports) / 11.0)
    decision: dict[str, object] = {
        "minimum_nonnegative_events": minimum_events,
        "structure_specificity": None,
        "rgid_geometry": None,
    }
    if stratified is not None:
        comparison = stratified["comparisons"][
            "c2_loccc_a6_minus_c1_local_window_a6"
        ]
        decision["structure_specificity"] = {
            "delta_miou_pp": 100.0
            * (
                reports["c2_loccc_a6"]["valid_miou"]
                - reports["c1_local_window_a6"]["valid_miou"]
            ),
            "ci95_low": comparison["ci95_low"],
            "pass": comparison["ci95_low"] > 0.0,
        }
        if rgid:
            rgid_comparison = stratified["comparisons"][
                "c3_rgid_g3_a6_minus_c2_loccc_a6"
            ]
            rgid_report = reports[RGID_VARIANT]
            checks = {
                "geometry_gain_ge_0_2pp": 100.0
                * (rgid_report["valid_miou"] - reports["c2_loccc_a6"]["valid_miou"])
                >= 0.2,
                "geometry_ci95_low_gt_zero": rgid_comparison["ci95_low"] > 0.0,
                "vs_p3_gain_ge_0_5pp": 100.0
                * (rgid_report["valid_miou"] - raw_miou)
                >= 0.5,
                "damaged_f1_gain_ge_1pp": 100.0
                * (
                    rgid_report["damaged_f1"]
                    - reports["c0_raw_p3"]["damaged_f1"]
                )
                >= 1.0,
                "event_nonnegative": event_wins[RGID_VARIANT]["nonnegative"]
                >= minimum_events,
            }
            decision["rgid_geometry"] = {
                "checks": checks,
                "pass": all(checks.values()),
            }

    payload = {
        "protocol": "SPI_LOCKED_A6_GEOMETRY_CAUSAL_STANDARD_ML_VAL450_v1",
        "split": "standard_ml_val450",
        "training": "NONE",
        "official_instances_accessed": False,
        "original_standard_test_accessed": False,
        "processed_images": processed,
        "variants": variants,
        "locked_contract": {
            "p3_sha256": file_sha256(args.checkpoint),
            "a6_minimum_purity": args.a6_minimum_purity,
            "a6_minimum_margin": args.a6_minimum_margin,
            "a6_maximum_area": None,
            "a6_core_radius": 0,
            "local_kernel_size": args.local_kernel_size,
            "rgid_mode": "g3" if rgid else None,
        },
        "rgid_checkpoint": (
            {
                "path": str(Path(args.rgid_checkpoint).resolve()),
                "sha256": file_sha256(args.rgid_checkpoint),
                "boundary_gate_threshold": rgid[3],
                "training_provenance": rgid[2].get("train_cache"),
            }
            if rgid
            else None
        ),
        "rgid_provenance": rgid_provenance,
        "p3_reproduction": reproduction,
        "reports": reports,
        "event_reports": event_reports,
        "event_win_counts_vs_c0": event_wins,
        "change_audit_vs_c0": changes,
        "region_totals": region_totals,
        "local_control_totals": local_totals,
        "rgid_decode_totals": decode_totals if rgid else None,
        "event_stratified_image_bootstrap": stratified,
        "event_macro_bootstrap": event_macro,
        "decision": decision,
        "image_histograms": image_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=" * 118)
    print("Locked A6 geometry causal audit | semantic TIFF primary | no training/instances/test")
    print(f"{'Variant':<28} {'mIoU':>10} {'gain':>10} {'D-F1':>10} {'I-IoU':>10} {'D-IoU':>10} {'X-IoU':>10}")
    for variant in variants:
        row = reports[variant]
        print(
            f"{variant:<28} {100*row['valid_miou']:10.4f} "
            f"{100*(row['valid_miou']-raw_miou):+10.4f} "
            f"{100*row['damaged_f1']:10.4f} {100*row['intact_iou']:10.4f} "
            f"{100*row['damaged_iou']:10.4f} {100*row['destroyed_iou']:10.4f}"
        )
    print(f"P3 reproduction: {reproduction}")
    if stratified is not None:
        print("Event-stratified paired bootstrap (mIoU percentage points):")
        for key, row in stratified["comparisons"].items():
            print(
                f"  {key}: mean={row['mean']:+.4f} "
                f"95%CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )
        print("Equal-event-weight bootstrap (mIoU percentage points):")
        for key, row in event_macro["comparisons"].items():
            print(
                f"  {key}: mean={row['mean']:+.4f} "
                f"95%CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )
        print(f"Event wins vs P3: {event_wins}")
        print(f"Pixel correction audit vs P3: {changes}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
