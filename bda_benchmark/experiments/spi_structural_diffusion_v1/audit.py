#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_selective_context_v1.core import (
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
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    predicted_building_regions,
    prediction_change_audit,
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import (
    VARIANTS,
    StructuralDiffusionConfig,
    regions_to_label_map,
    residual_to_full_logits,
    structural_diffusion,
)


MODES = {
    "n2_loccc_uniform_diffusion": "uniform",
    "n3_feature_edge_diffusion": "feature",
    "n4_modality_safe_diffusion": "modality",
    "n5_semantic_barrier_diffusion": "semantic",
}


def _accumulate_change(total: dict[str, int], audit: Mapping[str, object]) -> None:
    total["changed"] += int(audit["changed_pixels"])
    total["corrected"] += int(audit["corrected_pixels"])
    total["harmed"] += int(audit["harmed_pixels"])
    total["net"] += int(audit["net_gain_pixels"])


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Frozen-P3 N0-N5 structural diffusion audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=450)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--profile-timing", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected-p3-miou", type=float, default=0.605020)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise TypeError("P3 checkpoint must be a mapping")
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(checkpoint_state(source), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    model.to(device).eval().requires_grad_(False)

    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.val_list, args.sar_stats_json)
    if args.max_images <= 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    diffusion_config = StructuralDiffusionConfig()
    selective_config = SelectiveContextConfig()
    totals = {key: torch.zeros((4, 4), dtype=torch.float64) for key in VARIANTS}
    event_totals: dict[str, dict[str, torch.Tensor]] = {}
    image_rows: list[dict[str, object]] = []
    changes = {
        key: {"changed": 0, "corrected": 0, "harmed": 0, "net": 0}
        for key in VARIANTS[1:]
    }
    diagnostics = {
        key: {"images": 0, "support_fraction": 0.0, "mean_alpha_on_support": 0.0,
              "mean_modality_reliability": 0.0}
        for key in MODES
    }
    timing = {"data_wait": 0.0, "model_forward": 0.0, "region_build": 0.0,
              "n1_context": 0.0, "n2_n5_diffusion": 0.0, "metrics": 0.0}
    processed = 0
    started = time.perf_counter()
    previous_end = started
    for rgb, sar, names in loader:
        ready = time.perf_counter()
        timing["data_wait"] += ready - previous_end
        if args.max_images > 0 and processed >= args.max_images:
            break
        name = str(names[0])
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        phase = time.perf_counter()
        loc_logits, damage_logits, auxiliary = model(rgb, sar, return_aux=True)
        base_low_logits = model.clf_2(auxiliary["fused_roi"])
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["model_forward"] += time.perf_counter() - phase

        base_full_prob = damage_logits[0].float().softmax(0).cpu()
        raw = base_full_prob.argmax(0)
        localization = loc_logits[0].float().argmax(0).cpu()
        phase = time.perf_counter()
        regions = predicted_building_regions(localization, args.minimum_area)
        low_size = tuple(base_low_logits.shape[-2:])
        low_labels = regions_to_label_map(regions, low_size).to(device)
        full_support = torch.zeros_like(raw, dtype=torch.bool)
        for mask in regions:
            full_support |= mask.bool()
        timing["region_build"] += time.perf_counter() - phase

        phase = time.perf_counter()
        n1, _ = selective_context_refinement(
            raw, base_full_prob, regions, config=selective_config, mode="soft_region"
        )
        timing["n1_context"] += time.perf_counter() - phase

        phase = time.perf_counter()
        base_low_prob = base_low_logits.float().softmax(1)
        predictions = {"n0_raw_p3": raw, "n1_selective_context_s2": n1}
        support_device = full_support[None, None].to(device)
        for key, mode in MODES.items():
            refined, record = structural_diffusion(
                base_low_prob,
                low_labels,
                fused_feature=auxiliary["fused_roi"],
                rgb_feature=auxiliary["rgb_feature"],
                sar_feature=auxiliary["sar_feature"],
                mode=mode,
                config=diffusion_config,
            )
            refined_full_logits = residual_to_full_logits(
                base_low_prob, refined, damage_logits.float(), support_device
            )
            predictions[key] = refined_full_logits[0].argmax(0).cpu()
            diagnostics[key]["images"] += 1
            for field in ("support_fraction", "mean_alpha_on_support", "mean_modality_reliability"):
                diagnostics[key][field] += float(record[field])
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["n2_n5_diffusion"] += time.perf_counter() - phase

        # Ground truth is loaded only after every deployable prediction is fixed.
        phase = time.perf_counter()
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        table = event_totals.setdefault(
            event, {key: torch.zeros((4, 4), dtype=torch.float64) for key in VARIANTS}
        )
        histograms = {}
        for key, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[key] += histogram
            table[key] += histogram
            histograms[key] = histogram.long().tolist()
            if key != "n0_raw_p3":
                _accumulate_change(changes[key], prediction_change_audit(raw, prediction, target))
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        timing["metrics"] += time.perf_counter() - phase
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"SPI structural diffusion: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)
            if args.profile_timing:
                print("SPI timing | " + " | ".join(
                    f"{key}={value/processed:.3f}s/img" for key, value in timing.items()
                ), flush=True)
        previous_end = time.perf_counter()

    for record in diagnostics.values():
        count = max(int(record["images"]), 1)
        for field in ("support_fraction", "mean_alpha_on_support", "mean_modality_reliability"):
            record[field] /= count
    reports = {key: metrics_from_histogram(value) for key, value in totals.items()}
    event_reports = {
        event: {key: metrics_from_histogram(value) for key, value in table.items()}
        for event, table in sorted(event_totals.items())
    }
    event_wins = {
        key: {
            "nonnegative_vs_p3": sum(
                rows[key]["valid_miou"] >= rows["n0_raw_p3"]["valid_miou"]
                for rows in event_reports.values()
            ),
            "total": len(event_reports),
        }
        for key in VARIANTS[1:]
    }
    full = args.max_images <= 0 and processed == args.expected_images
    adjacent = tuple(zip(VARIANTS[2:], VARIANTS[1:-1]))
    bootstrap = stratified_paired_bootstrap(
        image_rows,
        variants=VARIANTS,
        baseline="n0_raw_p3",
        comparison_pairs=adjacent + (("n5_semantic_barrier_diffusion", "n1_selective_context_s2"),),
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    ) if full else None
    p3 = reports["n0_raw_p3"]
    reproduction_required = args.expected_p3_miou >= 0.0
    reproduction = {
        "expected": args.expected_p3_miou if reproduction_required else None,
        "observed": p3["valid_miou"],
        "error_pp": (
            100.0 * (p3["valid_miou"] - args.expected_p3_miou)
            if reproduction_required else None
        ),
        "passed": (
            abs(p3["valid_miou"] - args.expected_p3_miou) <= 0.00001
            if reproduction_required else True
        ),
        "basis": "locked_expected_metric" if reproduction_required else "locked_checkpoint_sha256",
    }
    candidates = VARIANTS[2:]
    best = max(candidates, key=lambda key: reports[key]["valid_miou"])
    gain_p3 = 100.0 * (reports[best]["valid_miou"] - p3["valid_miou"])
    gain_n1 = 100.0 * (reports[best]["valid_miou"] - reports["n1_selective_context_s2"]["valid_miou"])
    checks = None
    decision = None
    if bootstrap is not None:
        bootstrap_vs_n1 = stratified_paired_bootstrap(
            image_rows,
            variants=VARIANTS,
            baseline="n0_raw_p3",
            comparison_pairs=((best, "n1_selective_context_s2"),),
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        )["comparisons"][f"{best}_minus_n1_selective_context_s2"]
        checks = {
            "p3_reproduced": reproduction["passed"],
            "best_gain_vs_p3_ge_0_5pp": gain_p3 >= 0.5,
            "best_gain_vs_n1_ge_0_2pp": gain_n1 >= 0.2,
            "damaged_f1_gain_vs_p3_ge_1pp": 100.0 * (reports[best]["damaged_f1"] - p3["damaged_f1"]) >= 1.0,
            "no_foreground_drop_gt_0_5pp": min(
                100.0 * (reports[best][metric] - p3[metric])
                for metric in ("intact_iou", "damaged_iou", "destroyed_iou")
            ) >= -0.5,
            "at_least_10_events_nonnegative": event_wins[best]["nonnegative_vs_p3"] >= 10,
            "bootstrap_vs_n1_low_gt_zero": bootstrap_vs_n1["ci95_low"] > 0.0,
        }
        decision = "PROMOTE_STRUCTURAL_DIFFUSION" if all(checks.values()) else "KEEP_N1_STOP_OR_REDESIGN_DIFFUSION"
    payload = {
        "protocol": "SPI_STRUCTURAL_DIFFUSION_STANDARD_ML_VAL450_v1",
        "training": "NONE",
        "official_instances_accessed": False,
        "test_accessed": False,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "processed_images": processed,
        "config": diffusion_config.__dict__,
        "p3_reproduction": reproduction,
        "reports": reports,
        "event_reports": event_reports,
        "event_wins": event_wins,
        "change_audit": changes,
        "diffusion_diagnostics": diagnostics,
        "timing_seconds": timing if args.profile_timing else None,
        "bootstrap": bootstrap,
        "best_diffusion": best,
        "best_gain_vs_p3_pp": gain_p3,
        "best_gain_vs_n1_pp": gain_n1,
        "checks": checks,
        "decision": decision,
        "image_histograms": image_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=" * 118)
    print("Structural diffusion | frozen P3 | semantic TIFF | no instances/training/test")
    print(f"{'Variant':<38} {'mIoU':>10} {'gain':>10} {'D-F1':>10} {'I/D/X IoU':>24}")
    for key in VARIANTS:
        row = reports[key]
        print(f"{key:<38} {100*row['valid_miou']:10.4f} {100*(row['valid_miou']-p3['valid_miou']):+10.4f} "
              f"{100*row['damaged_f1']:10.4f} {100*row['intact_iou']:7.2f}/{100*row['damaged_iou']:7.2f}/{100*row['destroyed_iou']:7.2f}")
    print(f"Best diffusion: {best} | vs P3={gain_p3:+.4f}pp | vs N1={gain_n1:+.4f}pp")
    print(f"P3 reproduction: {reproduction}")
    print(f"Checks: {checks}\nDecision: {decision}\nSaved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
