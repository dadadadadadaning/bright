from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import build_region_maps
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
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import ALL_VARIANTS, PROTOCOL, TRAINABLE_VARIANTS
from .core import DistillationConfig, StructuralResidualStudent, build_structural_batch
from .train import _make_modules


def _load_modules(checkpoint_path: str, p3_sha: str, expected_steps: int, device: torch.device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("protocol") != PROTOCOL or saved.get("base_checkpoint_sha256") != p3_sha:
        raise RuntimeError("student checkpoint protocol/P3 provenance mismatch")
    if tuple(saved.get("variants", ())) != TRAINABLE_VARIANTS:
        raise RuntimeError("student variant contract mismatch")
    if int(saved.get("optimizer_step", -1)) != expected_steps:
        raise RuntimeError(
            f"student checkpoint step mismatch: expected={expected_steps} observed={saved.get('optimizer_step')}"
        )
    if saved.get("oracle_used") is not False or saved.get("test_accessed") is not False:
        raise RuntimeError("checkpoint violates no-Oracle/no-Test contract")
    hidden = int(saved.get("args", {}).get("hidden_channels", 64))
    modules = _make_modules(hidden)
    modules.load_state_dict(saved["trainable_states"], strict=True)
    return modules.to(device).eval().requires_grad_(False), saved


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("No-Oracle S2 structural distillation audit")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=450)
    parser.add_argument("--expected-steps", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--minimum-feature-area", type=int, default=2)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected-p3-miou", type=float, default=0.605020)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    p3_sha = file_sha256(args.base_checkpoint)
    base_saved = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(base_saved), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)
    modules, student_saved = _load_modules(args.student_checkpoint, p3_sha, args.expected_steps, device)
    train_args = student_saved.get("args", {})
    low_config = DistillationConfig(
        float(train_args.get("minimum_purity", 0.70)),
        float(train_args.get("minimum_margin", 0.10)),
        float(train_args.get("blend_strength", 0.75)),
    )
    exact_s2_config = SelectiveContextConfig(
        minimum_purity=low_config.minimum_purity,
        minimum_margin=low_config.minimum_margin,
        blend_strength=low_config.blend_strength,
    )
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.val_list, args.sar_stats_json)
    if args.max_images <= 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    totals = {name: torch.zeros((4, 4), dtype=torch.float64) for name in ALL_VARIANTS}
    events = defaultdict(lambda: {name: torch.zeros((4, 4), dtype=torch.float64) for name in ALL_VARIANTS})
    image_rows = []
    processed = 0
    accepted_regions = total_regions = 0
    started = time.perf_counter()
    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        name = str(names[0])
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits, auxiliary = p3(rgb, sar, return_aux=True)
        feature = auxiliary["fused_roi"]
        base_low = p3.clf_2(feature)
        region_maps = build_region_maps(
            auxiliary["p_building"], threshold=args.building_threshold,
            minimum_area=args.minimum_feature_area, core_radius=0, ring_radius=0,
        )
        structural = build_structural_batch(base_low, feature, region_maps, low_config)
        accepted_regions += structural.accepted_regions
        total_regions += structural.total_regions
        base_probability = F.softmax(base_low.float(), dim=1).to(feature.dtype)

        # Fix all deployable predictions before loading semantic target.
        predictions = {"k0_frozen_p3": damage_logits[0].argmax(0).cpu()}
        full_probability = damage_logits[0].float().softmax(0).cpu()
        localization = loc_logits[0].argmax(0).cpu()
        full_regions = predicted_building_regions(localization, args.minimum_area)
        predictions["k1_fixed_s2"], _ = selective_context_refinement(
            predictions["k0_frozen_p3"], full_probability, full_regions,
            config=exact_s2_config, mode="soft_region",
        )
        for variant, module in modules.items():
            low = base_low + module(feature, base_probability, structural)
            predictions[variant] = F.interpolate(
                low, size=damage_logits.shape[-2:], mode="bilinear", align_corners=False
            )[0].argmax(0).cpu()

        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        histograms = {}
        for variant, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[variant] += histogram
            events[event][variant] += histogram
            histograms[variant] = histogram.long().tolist()
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"S2 distillation audit: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)

    reports = {name: metrics_from_histogram(histogram) for name, histogram in totals.items()}
    event_reports = {
        event: {name: metrics_from_histogram(histogram) for name, histogram in rows.items()}
        for event, rows in sorted(events.items())
    }
    full = args.max_images <= 0 and processed == args.expected_images
    pairs = (
        ("k3_s2_distilled_local", "k2_semantic_local_control"),
        ("k4_s2_distilled_gbc", "k3_s2_distilled_local"),
        ("k5_s2_distilled_gbc_variance", "k4_s2_distilled_gbc"),
        ("k3_s2_distilled_local", "k1_fixed_s2"),
        ("k4_s2_distilled_gbc", "k1_fixed_s2"),
        ("k5_s2_distilled_gbc_variance", "k1_fixed_s2"),
    )
    bootstrap = stratified_paired_bootstrap(
        image_rows, variants=ALL_VARIANTS, baseline="k0_frozen_p3",
        comparison_pairs=pairs, iterations=args.bootstrap_iterations, seed=args.seed,
    ) if full else None
    p3_report = reports["k0_frozen_p3"]
    reproduction_required = args.expected_p3_miou >= 0
    reproduction = {
        "expected": args.expected_p3_miou if reproduction_required else None,
        "observed": p3_report["valid_miou"],
        "error_pp": 100 * (p3_report["valid_miou"] - args.expected_p3_miou) if reproduction_required else None,
        "passed": abs(p3_report["valid_miou"] - args.expected_p3_miou) <= 1e-5 if reproduction_required else True,
        "basis": "locked_expected_metric" if reproduction_required else "locked_checkpoint_sha256",
    }
    event_wins = {
        name: sum(rows[name]["valid_miou"] >= rows["k0_frozen_p3"]["valid_miou"] for rows in event_reports.values())
        for name in TRAINABLE_VARIANTS
    }
    best = max(TRAINABLE_VARIANTS[1:], key=lambda name: reports[name]["valid_miou"])
    checks = decision = None
    if bootstrap is not None:
        k3_vs_k2 = bootstrap["comparisons"]["k3_s2_distilled_local_minus_k2_semantic_local_control"]
        causal_checks = {
            "p3_reproduced": reproduction["passed"],
            "k3_beats_k2_ge_0_2pp": 100 * (reports["k3_s2_distilled_local"]["valid_miou"] - reports["k2_semantic_local_control"]["valid_miou"]) >= 0.2,
            "k3_not_below_s2_gt_0_2pp": 100 * (reports["k3_s2_distilled_local"]["valid_miou"] - reports["k1_fixed_s2"]["valid_miou"]) >= -0.2,
            "k3_damaged_f1_gain_ge_0_5pp": 100 * (reports["k3_s2_distilled_local"]["damaged_f1"] - p3_report["damaged_f1"]) >= 0.5,
            "k3_no_foreground_drop_gt_0_5pp": min(
                100 * (reports["k3_s2_distilled_local"][metric] - p3_report[metric])
                for metric in ("intact_iou", "damaged_iou", "destroyed_iou")
            ) >= -0.5,
            "k3_event_nonnegative": event_wins["k3_s2_distilled_local"] >= min(10, len(event_reports)),
            "k3_vs_k2_bootstrap_low_gt_zero": k3_vs_k2["ci95_low"] > 0,
        }
        final_checks = {
            "best_gain_vs_p3_ge_0_5pp": 100 * (reports[best]["valid_miou"] - p3_report["valid_miou"]) >= 0.5,
            "best_damaged_f1_gain_vs_p3_ge_1pp": 100 * (reports[best]["damaged_f1"] - p3_report["damaged_f1"]) >= 1.0,
            "best_event_nonnegative": event_wins[best] >= min(10, len(event_reports)),
            "best_no_foreground_drop_gt_0_5pp": min(
                100 * (reports[best][metric] - p3_report[metric])
                for metric in ("intact_iou", "damaged_iou", "destroyed_iou")
            ) >= -0.5,
            "best_vs_p3_bootstrap_low_gt_zero": bootstrap["deltas_vs_a0"][best]["ci95_low"] > 0,
        }
        checks = {"causal": causal_checks, "final": final_checks}
        if args.expected_steps < 3000:
            decision = "PROMOTE_S2_DISTILLATION_TO_FULL" if all(causal_checks.values()) else "STOP_S2_DISTILLATION_AFTER_PILOT"
        else:
            learned_valid = all(final_checks.values()) and reproduction["passed"]
            best_vs_s2 = bootstrap["comparisons"][f"{best}_minus_k1_fixed_s2"]
            beats_s2 = (
                reports[best]["valid_miou"] > reports["k1_fixed_s2"]["valid_miou"]
                and best_vs_s2["ci95_low"] > 0
            )
            decision = (
                "S2_DISTILLED_MODEL_SURPASSES_FIXED_S2" if learned_valid and beats_s2
                else "S2_DISTILLED_MODEL_VALID_BELOW_FIXED_S2" if learned_valid
                else "KEEP_FIXED_S2_STOP_TRAINED_MODEL"
            )

    payload = {
        "protocol": PROTOCOL,
        "oracle_used": False,
        "official_instances_accessed": False,
        "test_accessed": False,
        "processed_images": processed,
        "base_checkpoint_sha256": p3_sha,
        "student_checkpoint": str(args.student_checkpoint),
        "student_steps": args.expected_steps,
        "reports": reports,
        "event_reports": event_reports,
        "event_wins_vs_p3": event_wins,
        "p3_reproduction": reproduction,
        "region_diagnostics": {"total": total_regions, "accepted": accepted_regions},
        "bootstrap": bootstrap,
        "best_distilled_variant": best,
        "checks": checks,
        "decision": decision,
        "image_histograms": image_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=" * 118)
    print("S2 structural distillation | semantic TIFF | no instances/Oracle/test")
    for name in ALL_VARIANTS:
        row = reports[name]
        gain = 100 * (row["valid_miou"] - p3_report["valid_miou"])
        print(f"{name:<40} mIoU={100*row['valid_miou']:.4f} gain={gain:+.4f}pp D-F1={100*row['damaged_f1']:.4f}")
    print(f"Best distilled: {best}")
    print(f"P3 reproduction: {reproduction}\nChecks: {checks}\nDecision: {decision}")
    print(f"Saved: {output}\nOfficial instances/Oracle/test: NEVER / NEVER / NEVER")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
