from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    checkpoint_state, fast_histogram, file_sha256, metrics_from_histogram, p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import VARIANTS
from .model import DamageReadoutControl, ParameterMatchedAdapter, adapter_inputs
from .regions import build_region_maps


ALL_VARIANTS = ("m0_frozen_p3",) + VARIANTS
PROTOCOL = "SPI_CORE_RING_ADAPTER_PURE_SEMANTIC_VAL450_v1"


def gains(candidate, reference):
    keys = ("valid_miou", "intact_iou", "damaged_iou", "destroyed_iou", "damaged_f1", "damaged_to_intact", "destroyed_to_intact")
    return {key: 100.0 * (float(candidate[key]) - float(reference[key])) for key in keys}


def load_modules(args, p3, device, base_sha):
    modules = {}
    manifests = {}
    joint_path = Path(args.checkpoint_root) / "joint_last.pth"
    joint = torch.load(joint_path, map_location="cpu", weights_only=False) if joint_path.is_file() else None
    for variant in VARIANTS:
        path = joint_path if joint is not None else Path(args.checkpoint_root) / variant / "last.pth"
        if joint is not None:
            saved = joint
            if tuple(saved.get("variants", ())) != VARIANTS or saved.get("base_checkpoint_sha256") != base_sha:
                raise RuntimeError(f"joint checkpoint provenance mismatch: {path}")
        else:
            if not path.is_file():
                raise FileNotFoundError(f"missing trained variant: {path}")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("variant") != variant or saved.get("base_checkpoint_sha256") != base_sha:
                raise RuntimeError(f"checkpoint provenance mismatch: {path}")
        if int(saved.get("optimizer_step", 0)) != args.expected_steps:
            raise RuntimeError(
                f"checkpoint step mismatch: {path} expected={args.expected_steps} "
                f"observed={saved.get('optimizer_step')}"
            )
        hidden = int(saved.get("args", {}).get("hidden_channels", 64))
        module = DamageReadoutControl(p3.clf_2) if variant == VARIANTS[0] else ParameterMatchedAdapter(256, hidden, 4)
        if joint is not None:
            prefix = variant + "."
            state = {key[len(prefix):]: value for key, value in saved["trainable_states"].items() if key.startswith(prefix)}
        else:
            state = saved["trainable_state"]
        module.load_state_dict(state, strict=True)
        modules[variant] = module.to(device).eval().requires_grad_(False)
        manifests[variant] = {
            "path": str(path), "optimizer_step": int(saved["optimizer_step"]),
            "seed": int(saved["seed"]), "base_checkpoint_sha256": saved["base_checkpoint_sha256"],
        }
    return modules, manifests


def stratified_paired_bootstrap(image_rows, candidate, reference, iterations, seed):
    grouped = defaultdict(list)
    for event, histograms in image_rows:
        grouped[event].append(histograms)
    rng = np.random.default_rng(seed)
    cand_hist = np.zeros((iterations, 4, 4), dtype=np.float64)
    ref_hist = np.zeros((iterations, 4, 4), dtype=np.float64)
    for rows in grouped.values():
        candidate_rows = np.stack([row[candidate].numpy() for row in rows])
        reference_rows = np.stack([row[reference].numpy() for row in rows])
        indices = rng.integers(0, len(rows), size=(iterations, len(rows)))
        cand_hist += candidate_rows[indices].sum(axis=1)
        ref_hist += reference_rows[indices].sum(axis=1)

    def vector_miou(hist):
        intersection = np.diagonal(hist, axis1=1, axis2=2)
        union = hist.sum(axis=1) + hist.sum(axis=2) - intersection
        iou = intersection / np.maximum(union, 1.0)
        return iou[:, 1:4].mean(axis=1)

    values = 100.0 * (vector_miou(cand_hist) - vector_miou(ref_hist))
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": iterations,
        "unit": "images resampled within each event",
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=450)
    parser.add_argument("--expected-steps", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected-p3-miou", type=float, default=0.60502)
    parser.add_argument("--minimum-feature-area", type=int, default=2)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--core-radius", type=int, default=1)
    parser.add_argument("--ring-radius", type=int, default=2)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    base_sha = file_sha256(args.base_checkpoint)
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(base_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)
    modules, manifests = load_modules(args, p3, device, base_sha)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.val_list, args.sar_stats_json)
    if args.max_images <= 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} validation images, observed {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    histograms = {name: torch.zeros(4, 4, dtype=torch.float64) for name in ALL_VARIANTS}
    event_histograms = defaultdict(lambda: {name: torch.zeros(4, 4, dtype=torch.float64) for name in ALL_VARIANTS})
    image_rows = []
    region_args = dict(threshold=args.building_threshold, minimum_area=args.minimum_feature_area,
                       core_radius=args.core_radius, ring_radius=args.ring_radius)
    processed = 0
    total_regions = 0
    started = time.perf_counter()
    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        _, base_output, auxiliary = p3(rgb, sar, return_aux=True)
        feature = auxiliary["fused_roi"]
        base_low = p3.clf_2(feature)
        regions = build_region_maps(auxiliary["p_building"], **region_args)
        total_regions += regions.region_count
        outputs = {"m0_frozen_p3": base_output}
        outputs["m1_readout_finetune"] = F.interpolate(modules[VARIANTS[0]](feature), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        for variant in VARIANTS[1:]:
            context, contrast, support = adapter_inputs(variant, feature, regions)
            low = base_low + modules[variant](feature, context, contrast, support)
            outputs[variant] = F.interpolate(low, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        name = str(names[0])
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        row = {}
        for variant, output in outputs.items():
            prediction = output[0].float().argmax(0).cpu()
            histogram = fast_histogram(prediction, target)
            histograms[variant] += histogram
            event_histograms[event][variant] += histogram
            row[variant] = histogram
        image_rows.append((event, row))
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"Core/Ring audit: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)

    reports = {name: metrics_from_histogram(hist) for name, hist in histograms.items()}
    for name in ALL_VARIANTS:
        reports[name]["confusion"] = histograms[name].long().tolist()
    event_reports = {
        event: {name: metrics_from_histogram(hist) for name, hist in table.items()}
        for event, table in sorted(event_histograms.items())
    }
    comparisons = {}
    pairs = (
        ("m1_minus_m0", "m1_readout_finetune", "m0_frozen_p3"),
        ("m2_minus_m1", "m2_local_adapter", "m1_readout_finetune"),
        ("m3_minus_m2", "m3_loccc_region", "m2_local_adapter"),
        ("m4_minus_m3", "m4_core_ring_region", "m3_loccc_region"),
        ("m4_minus_m0", "m4_core_ring_region", "m0_frozen_p3"),
    )
    for label, candidate, reference in pairs:
        event_deltas = [100.0 * (table[candidate]["valid_miou"] - table[reference]["valid_miou"]) for table in event_reports.values()]
        comparisons[label] = {
            "gain_pp": gains(reports[candidate], reports[reference]),
            "event_nonnegative": sum(value >= 0 for value in event_deltas),
            "event_total": len(event_deltas),
            "bootstrap": stratified_paired_bootstrap(image_rows, candidate, reference, args.bootstrap_iterations, args.seed),
        }
    reproduction_error = 100.0 * (reports["m0_frozen_p3"]["valid_miou"] - args.expected_p3_miou)
    final_gain = comparisons["m4_minus_m0"]["gain_pp"]
    checks = {
        "p3_reproduced": abs(reproduction_error) <= 0.05,
        "m3_region_gain_ge_0_2pp": comparisons["m3_minus_m2"]["gain_pp"]["valid_miou"] >= 0.2,
        "m3_region_ci_low_gt_zero": comparisons["m3_minus_m2"]["bootstrap"]["ci95_low_pp"] > 0,
        "m4_core_ring_gain_ge_0_2pp": comparisons["m4_minus_m3"]["gain_pp"]["valid_miou"] >= 0.2,
        "m4_core_ring_ci_low_gt_zero": comparisons["m4_minus_m3"]["bootstrap"]["ci95_low_pp"] > 0,
        "m4_vs_p3_gain_ge_0_5pp": final_gain["valid_miou"] >= 0.5,
        "m4_damaged_f1_gain_ge_1pp": final_gain["damaged_f1"] >= 1.0,
        "no_foreground_iou_drop_gt_0_5pp": min(final_gain[k] for k in ("intact_iou", "damaged_iou", "destroyed_iou")) >= -0.5,
        "at_least_10_events_nonnegative": comparisons["m4_minus_m0"]["event_nonnegative"] >= 10,
        "m4_vs_p3_ci_low_gt_zero": comparisons["m4_minus_m0"]["bootstrap"]["ci95_low_pp"] > 0,
    }
    if all(checks.values()):
        decision = (
            "PROCEED_TO_RCM_CAA_SCREEN"
            if args.expected_steps >= 3000
            else "PROCEED_TO_FULL_CONFIRMATION"
        )
    else:
        decision = "STOP_AND_DIAGNOSE_FIRST_FAILED_CAUSAL_STEP"
    payload = {
        "protocol": PROTOCOL, "semantic_ground_truth": "dataset/target/*_building_damage.tif",
        "processed_images": processed, "base_checkpoint": args.base_checkpoint,
        "base_checkpoint_sha256": base_sha, "checkpoints": manifests,
        "official_instances_accessed": False, "test_accessed": False,
        "p3_reproduction": {"expected": args.expected_p3_miou, "observed": reports["m0_frozen_p3"]["valid_miou"], "error_pp": reproduction_error, "passed": checks["p3_reproduced"]},
        "region_definition": region_args, "total_loccc_regions": total_regions,
        "reports": reports, "event_reports": event_reports, "comparisons": comparisons,
        "decision": {"checks": checks, "decision": decision},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 116)
    for name in ALL_VARIANTS:
        report = reports[name]
        print(f"{name:28s} mIoU={100*report['valid_miou']:.4f} D-F1={100*report['damaged_f1']:.4f} I/D/X={100*report['intact_iou']:.2f}/{100*report['damaged_iou']:.2f}/{100*report['destroyed_iou']:.2f}")
    print(f"Decision: {decision}\nChecks: {checks}\nSaved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
