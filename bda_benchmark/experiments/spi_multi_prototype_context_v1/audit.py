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

from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig, selective_context_refinement
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import checkpoint_state, fast_histogram, file_sha256, metrics_from_histogram, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_region_feedback.core import predicted_building_regions, prediction_change_audit, stratified_paired_bootstrap
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import VARIANTS, MultiPrototypeConfig, multi_prototype_context, probability_residual_to_prediction, regions_to_label_map


ORACLE = "o1_best_deployable_pixel_oracle"


def _add_change(total, audit):
    total["changed"] += int(audit["changed_pixels"])
    total["corrected"] += int(audit["corrected_pixels"])
    total["harmed"] += int(audit["harmed_pixels"])
    total["net"] += int(audit["net_gain_pixels"])


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Frozen-P3 multi-prototype context audit")
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
    parser.add_argument("--profile-timing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected-p3-miou", type=float, default=0.605020)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("P3 checkpoint must be a mapping")
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    model.to(device).eval().requires_grad_(False)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.val_list, args.sar_stats_json)
    if args.max_images <= 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    config = MultiPrototypeConfig()
    s2_config = SelectiveContextConfig()
    metric_keys = VARIANTS + (ORACLE,)
    totals = {key: torch.zeros((4, 4), dtype=torch.float64) for key in metric_keys}
    event_totals = {}
    image_rows = []
    changes = {key: {"changed": 0, "corrected": 0, "harmed": 0, "net": 0} for key in VARIANTS[1:]}
    diagnostics = {"probability": {}, "feature": {}}
    timing = {"data_wait": 0.0, "model_forward": 0.0, "regions_and_s2": 0.0,
              "multi_prototype": 0.0, "metrics": 0.0}
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
        loc_logits, damage_logits, aux = model(rgb, sar, return_aux=True)
        base_low_logits = model.clf_2(aux["fused_roi"])
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["model_forward"] += time.perf_counter() - phase
        full_probability = damage_logits[0].float().softmax(0).cpu()
        raw = full_probability.argmax(0)
        localization = loc_logits[0].argmax(0).cpu()

        phase = time.perf_counter()
        regions = predicted_building_regions(localization, args.minimum_area)
        n1, _ = selective_context_refinement(raw, full_probability, regions, config=s2_config, mode="soft_region")
        low_labels = regions_to_label_map(regions, tuple(base_low_logits.shape[-2:])).to(device)
        full_support = torch.zeros_like(raw, dtype=torch.bool)
        for mask in regions:
            full_support |= mask.bool()
        timing["regions_and_s2"] += time.perf_counter() - phase

        phase = time.perf_counter()
        base_low_probability = base_low_logits.float().softmax(1)
        probability_refined, _, _, probability_record = multi_prototype_context(
            base_low_probability, aux["fused_roi"], low_labels,
            config=config, use_feature_assignment=False,
        )
        feature_refined, eligible_low, selective_low, feature_record = multi_prototype_context(
            base_low_probability, aux["fused_roi"], low_labels,
            config=config, use_feature_assignment=True,
        )
        support_device = full_support[None, None].to(device)
        m2 = probability_residual_to_prediction(base_low_probability, probability_refined, damage_logits.float(), support_device)[0].cpu()
        m3 = probability_residual_to_prediction(base_low_probability, feature_refined, damage_logits.float(), support_device)[0].cpu()
        eligible_full = F.interpolate(eligible_low[:, None].float(), size=raw.shape, mode="nearest")[0, 0].cpu().bool() & full_support
        selective_full = F.interpolate(selective_low[:, None].float(), size=raw.shape, mode="nearest")[0, 0].cpu().bool() & full_support
        m4 = n1.clone()
        m4[eligible_full] = m3[eligible_full]
        m5 = n1.clone()
        m5[selective_full] = m3[selective_full]
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["multi_prototype"] += time.perf_counter() - phase
        predictions = {
            "m0_raw_p3": raw,
            "m1_single_prototype_s2": n1,
            "m2_probability_multiproto": m2,
            "m3_feature_multiproto": m3,
            "m4_hybrid_single_multi": m4,
            "m5_hybrid_selective_multi": m5,
        }
        for label, record in (("probability", probability_record), ("feature", feature_record)):
            for key, value in record.items():
                diagnostics[label][key] = diagnostics[label].get(key, 0.0) + float(value)

        # All deployable predictions are fixed before the semantic target is loaded.
        phase = time.perf_counter()
        target = load_semantic_target(args.dataset_path, name)
        oracle = raw.clone()
        for candidate in (n1, m2, m3, m4, m5):
            use = (candidate == target) & (oracle != target) & (target != 255)
            oracle[use] = candidate[use]
        predictions[ORACLE] = oracle
        event = name.rsplit("_", 1)[0]
        table = event_totals.setdefault(event, {key: torch.zeros((4, 4), dtype=torch.float64) for key in metric_keys})
        histograms = {}
        for key, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[key] += histogram
            table[key] += histogram
            histograms[key] = histogram.long().tolist()
            if key in changes:
                _add_change(changes[key], prediction_change_audit(raw, prediction, target))
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        timing["metrics"] += time.perf_counter() - phase
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"SPI multi-prototype: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)
            if args.profile_timing:
                print("SPI timing | " + " | ".join(f"{k}={v/processed:.3f}s/img" for k, v in timing.items()), flush=True)
        previous_end = time.perf_counter()

    for record in diagnostics.values():
        for key in tuple(record):
            record[key] /= max(processed, 1)
    reports = {key: metrics_from_histogram(value) for key, value in totals.items()}
    event_reports = {event: {key: metrics_from_histogram(value) for key, value in table.items()} for event, table in sorted(event_totals.items())}
    event_wins = {key: {"nonnegative_vs_p3": sum(rows[key]["valid_miou"] >= rows["m0_raw_p3"]["valid_miou"] for rows in event_reports.values()), "total": len(event_reports)} for key in VARIANTS[1:]}
    full = args.max_images <= 0 and processed == args.expected_images
    pairs = (("m2_probability_multiproto", "m1_single_prototype_s2"),
             ("m3_feature_multiproto", "m2_probability_multiproto"),
             ("m4_hybrid_single_multi", "m1_single_prototype_s2"),
             ("m5_hybrid_selective_multi", "m4_hybrid_single_multi"),
             ("m5_hybrid_selective_multi", "m1_single_prototype_s2"))
    bootstrap = stratified_paired_bootstrap(image_rows, variants=metric_keys, baseline="m0_raw_p3",
                                             comparison_pairs=pairs, iterations=args.bootstrap_iterations,
                                             seed=args.seed) if full else None
    p3 = reports["m0_raw_p3"]
    reproduction_required = args.expected_p3_miou >= 0
    reproduction = {"expected": args.expected_p3_miou if reproduction_required else None,
                    "observed": p3["valid_miou"],
                    "error_pp": 100*(p3["valid_miou"]-args.expected_p3_miou) if reproduction_required else None,
                    "passed": abs(p3["valid_miou"]-args.expected_p3_miou) <= 1e-5 if reproduction_required else True,
                    "basis": "locked_expected_metric" if reproduction_required else "locked_checkpoint_sha256"}
    best = max(VARIANTS[2:], key=lambda key: reports[key]["valid_miou"])
    gain_p3 = 100 * (reports[best]["valid_miou"] - p3["valid_miou"])
    gain_s2 = 100 * (reports[best]["valid_miou"] - reports["m1_single_prototype_s2"]["valid_miou"])
    checks = decision = None
    if bootstrap is not None:
        best_bootstrap = stratified_paired_bootstrap(
            image_rows, variants=metric_keys, baseline="m0_raw_p3",
            comparison_pairs=((best, "m1_single_prototype_s2"),),
            iterations=args.bootstrap_iterations, seed=args.seed + 1,
        )["comparisons"][f"{best}_minus_m1_single_prototype_s2"]
        checks = {
            "p3_reproduced": reproduction["passed"],
            "best_beats_s2_ge_0_2pp": gain_s2 >= 0.2,
            "best_beats_p3_ge_0_5pp": gain_p3 >= 0.5,
            "damaged_f1_gain_vs_p3_ge_1pp": 100*(reports[best]["damaged_f1"]-p3["damaged_f1"]) >= 1.0,
            "no_foreground_drop_gt_0_5pp": min(100*(reports[best][metric]-p3[metric]) for metric in ("intact_iou", "damaged_iou", "destroyed_iou")) >= -0.5,
            "at_least_10_events_nonnegative": event_wins[best]["nonnegative_vs_p3"] >= 10,
            "bootstrap_vs_s2_low_gt_zero": best_bootstrap["ci95_low"] > 0,
        }
        decision = "PROMOTE_MULTI_PROTOTYPE_TO_VAL450" if all(checks.values()) else "KEEP_S2_STOP_MULTI_PROTOTYPE_V1"
    payload = {"protocol": "SPI_MULTI_PROTOTYPE_CONTEXT_V1", "training": "NONE",
               "official_instances_accessed": False, "test_accessed": False,
               "checkpoint_sha256": file_sha256(args.checkpoint), "processed_images": processed,
               "config": config.__dict__, "p3_reproduction": reproduction, "reports": reports,
               "event_reports": event_reports, "event_wins": event_wins, "change_audit": changes,
               "prototype_diagnostics": diagnostics, "timing_seconds": timing if args.profile_timing else None,
               "bootstrap": bootstrap, "best_multi_prototype": best,
               "best_gain_vs_p3_pp": gain_p3, "best_gain_vs_s2_pp": gain_s2,
               "checks": checks, "decision": decision, "image_histograms": image_rows}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("="*118)
    print("Multi-prototype context | frozen P3 | semantic TIFF | no instances/training/test")
    for key in metric_keys:
        row = reports[key]
        print(f"{key:<38} mIoU={100*row['valid_miou']:.4f} gain={100*(row['valid_miou']-p3['valid_miou']):+.4f}pp D-F1={100*row['damaged_f1']:.4f}")
    print(f"Best: {best} | vs P3={gain_p3:+.4f}pp | vs S2={gain_s2:+.4f}pp")
    print(f"P3 reproduction: {reproduction}\nChecks: {checks}\nDecision: {decision}\nSaved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("="*118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
