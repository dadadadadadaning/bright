#!/usr/bin/env python3
"""Frozen-P3 audit for selective soft building-context refinement."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_selective_context_v1.core import (
    VARIANTS,
    SelectiveContextConfig,
    selective_context_refinement,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
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
    event_macro_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    gated_majority_feedback,
    predicted_building_regions,
    prediction_change_audit,
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SPI selective building-context audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--expected_images", type=int, default=450)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--minimum_area", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--profile_timing", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_miou", type=float, default=0.605020)
    args = parser.parse_args(argv)

    # Protocol-locked constants. Change them only in a new experiment version.
    config = SelectiveContextConfig()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("P3 checkpoint must be a mapping")
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    model.to(device).eval().requires_grad_(False)

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

    totals = {name: torch.zeros((4, 4), dtype=torch.float64) for name in VARIANTS}
    events: dict[str, dict[str, torch.Tensor]] = {}
    image_rows: list[dict[str, object]] = []
    changes = {
        name: {"changed": 0, "corrected": 0, "harmed": 0, "net": 0}
        for name in VARIANTS[1:]
    }
    region_count = 0
    accepted = {name: 0 for name in VARIANTS[2:]}
    processed = 0
    started = time.perf_counter()
    timing = {
        "data_wait": 0.0,
        "model_forward": 0.0,
        "connected_regions": 0.0,
        "four_refinements": 0.0,
        "target_and_metrics": 0.0,
    }
    previous_end = started

    for rgb, sar, names in loader:
        batch_ready = time.perf_counter()
        timing["data_wait"] += batch_ready - previous_end
        if args.max_images > 0 and processed >= args.max_images:
            break
        name = str(names[0])
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_started = time.perf_counter()
        loc_logits, damage_logits = model(rgb, p3_sar_view(sar))
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["model_forward"] += time.perf_counter() - phase_started
        probabilities = damage_logits[0].float().softmax(dim=0).cpu()
        raw = probabilities.argmax(dim=0)
        localization = loc_logits[0].float().argmax(dim=0).cpu()
        phase_started = time.perf_counter()
        regions = predicted_building_regions(localization, args.minimum_area)
        timing["connected_regions"] += time.perf_counter() - phase_started
        region_count += len(regions)

        phase_started = time.perf_counter()
        hard, _ = gated_majority_feedback(
            raw,
            probabilities,
            regions,
            minimum_purity=config.minimum_purity,
            minimum_margin=config.minimum_margin,
            maximum_area=0,
            core_radius=0,
        )
        soft, soft_rows = selective_context_refinement(
            raw, probabilities, regions, config=config, mode="soft_region"
        )
        selective, selective_rows = selective_context_refinement(
            raw, probabilities, regions, config=config, mode="uncertain_interior"
        )
        local, local_rows = selective_context_refinement(
            raw, probabilities, regions, config=config, mode="local_control"
        )
        accepted["s2_loccc_soft_context"] += sum(r["accepted"] for r in soft_rows)
        accepted["s3_loccc_uncertain_interior"] += sum(
            r["accepted"] for r in selective_rows
        )
        accepted["s4_local_uncertain_control"] += sum(
            r["accepted"] for r in local_rows
        )
        timing["four_refinements"] += time.perf_counter() - phase_started
        predictions = {
            "s0_raw_p3": raw,
            "s1_loccc_hard_a6": hard,
            "s2_loccc_soft_context": soft,
            "s3_loccc_uncertain_interior": selective,
            "s4_local_uncertain_control": local,
        }

        # Labels are loaded only after every deployable prediction is fixed.
        phase_started = time.perf_counter()
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        event_table = events.setdefault(
            event,
            {key: torch.zeros((4, 4), dtype=torch.float64) for key in VARIANTS},
        )
        histograms = {}
        for key, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[key] += histogram
            event_table[key] += histogram
            histograms[key] = histogram.long().tolist()
            if key != "s0_raw_p3":
                audit = prediction_change_audit(raw, prediction, target)
                changes[key]["changed"] += int(audit["changed_pixels"])
                changes[key]["corrected"] += int(audit["corrected_pixels"])
                changes[key]["harmed"] += int(audit["harmed_pixels"])
                changes[key]["net"] += int(audit["net_gain_pixels"])
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        timing["target_and_metrics"] += time.perf_counter() - phase_started
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"SPI selective context: {processed}/{len(dataset)} | "
                f"elapsed={elapsed:.1f}s | rate={processed/max(elapsed, 1e-9):.2f} img/s",
                flush=True,
            )
            if args.profile_timing:
                means = " | ".join(
                    f"{key}={value/processed:.3f}s/img"
                    for key, value in timing.items()
                )
                print(f"SPI timing | {means}", flush=True)
        previous_end = time.perf_counter()

    reports = {key: metrics_from_histogram(totals[key]) for key in VARIANTS}
    event_reports = {
        event: {key: metrics_from_histogram(table[key]) for key in VARIANTS}
        for event, table in sorted(events.items())
    }
    wins = {
        key: {
            "improved": sum(
                rows[key]["valid_miou"] > rows["s0_raw_p3"]["valid_miou"]
                for rows in event_reports.values()
            ),
            "nonnegative": sum(
                rows[key]["valid_miou"] >= rows["s0_raw_p3"]["valid_miou"]
                for rows in event_reports.values()
            ),
            "total": len(event_reports),
        }
        for key in VARIANTS[1:]
    }
    full = args.max_images <= 0 and processed == args.expected_images
    pairs = (
        ("s3_loccc_uncertain_interior", "s1_loccc_hard_a6"),
        ("s3_loccc_uncertain_interior", "s4_local_uncertain_control"),
    )
    bootstrap = (
        stratified_paired_bootstrap(
            image_rows,
            variants=VARIANTS,
            baseline="s0_raw_p3",
            comparison_pairs=pairs,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        if full
        else None
    )
    event_macro = (
        event_macro_paired_bootstrap(
            image_rows,
            variants=VARIANTS,
            baseline="s0_raw_p3",
            comparison_pairs=pairs,
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        )
        if full
        else None
    )
    p3_miou = reports["s0_raw_p3"]["valid_miou"]
    reproduction = {
        "expected": args.expected_p3_miou,
        "observed": p3_miou,
        "error_pp": 100.0 * (p3_miou - args.expected_p3_miou),
        "passed": abs(p3_miou - args.expected_p3_miou) <= 0.00001,
    }
    decision = None
    if bootstrap is not None:
        candidate = reports["s3_loccc_uncertain_interior"]
        minimum_events = math.ceil(7.0 * len(event_reports) / 11.0)
        structure_ci = bootstrap["comparisons"][
            "s3_loccc_uncertain_interior_minus_s4_local_uncertain_control"
        ]["ci95_low"]
        checks = {
            "p3_reproduced": reproduction["passed"],
            "miou_gain_ge_0_5pp": 100.0 * (candidate["valid_miou"] - p3_miou) >= 0.5,
            "damaged_f1_gain_ge_1pp": 100.0 * (
                candidate["damaged_f1"] - reports["s0_raw_p3"]["damaged_f1"]
            ) >= 1.0,
            "no_foreground_class_drop_gt_0_5pp": min(
                100.0 * (candidate[name] - reports["s0_raw_p3"][name])
                for name in ("intact_iou", "damaged_iou", "destroyed_iou")
            ) >= -0.5,
            "event_nonnegative": wins["s3_loccc_uncertain_interior"]["nonnegative"]
            >= minimum_events,
            "structure_specific_ci_low_gt_zero": structure_ci > 0.0,
        }
        decision = {"checks": checks, "pass": all(checks.values())}

    payload = {
        "protocol": "SPI_SELECTIVE_CONTEXT_STANDARD_ML_VAL450_v1",
        "split": "standard_ml_val450",
        "training": "NONE",
        "official_instances_accessed": False,
        "original_standard_test_accessed": False,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config": config.__dict__,
        "processed_images": processed,
        "region_count": region_count,
        "accepted_regions": accepted,
        "timing_seconds": timing if args.profile_timing else None,
        "p3_reproduction": reproduction,
        "reports": reports,
        "event_reports": event_reports,
        "event_wins": wins,
        "change_audit": changes,
        "event_stratified_image_bootstrap": bootstrap,
        "equal_event_bootstrap": event_macro,
        "decision": decision,
        "image_histograms": image_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=" * 116)
    print("Selective context | frozen P3 | semantic TIFF Val450 | no instances/training/test")
    print(f"{'Variant':<34} {'mIoU':>10} {'gain':>10} {'D-F1':>10} {'I-IoU':>10} {'D-IoU':>10} {'X-IoU':>10}")
    for key in VARIANTS:
        row = reports[key]
        print(
            f"{key:<34} {100*row['valid_miou']:10.4f} "
            f"{100*(row['valid_miou']-p3_miou):+10.4f} "
            f"{100*row['damaged_f1']:10.4f} {100*row['intact_iou']:10.4f} "
            f"{100*row['damaged_iou']:10.4f} {100*row['destroyed_iou']:10.4f}"
        )
    print(f"P3 reproduction: {reproduction}")
    print(f"Event wins: {wins}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
