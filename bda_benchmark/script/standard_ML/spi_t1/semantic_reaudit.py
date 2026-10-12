#!/usr/bin/env python3
"""Pure semantic re-audit of locked P3 and paired T0/T1 checkpoints.

This evaluator reads only RGB, SAR and ``target/*_building_damage.tif``.
Official instance annotations are deliberately not accepted.  It is intended
to separate checkpoint drift from validation-split shift and records whether
the chosen list belongs to the original standard-ML test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[3]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
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
)
from bda_benchmark.script.standard_ML.spi_t1.audit import (  # noqa: E402
    load_semantic_target,
    validate_pair_metadata,
)


VARIANTS = ("p3_locked_source", "t0_original_objective", "t1_component_balanced")


def metric_gain_pp(candidate: Mapping[str, float], reference: Mapping[str, float]):
    keys = (
        "valid_miou",
        "background_iou",
        "intact_iou",
        "damaged_iou",
        "destroyed_iou",
        "damaged_f1",
        "damaged_to_intact",
        "destroyed_to_intact",
    )
    return {key: 100.0 * (float(candidate[key]) - float(reference[key])) for key in keys}


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    dataset: H9RGBSARInferenceDataset,
    device: torch.device,
    num_workers: int,
    max_images: int,
    progress_every: int,
    variant: str,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]], int]:
    model = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict {variant} checkpoint load failed: {incompatible}")
    model.eval().requires_grad_(False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    overall = torch.zeros(4, 4, dtype=torch.float64)
    event_histograms: Dict[str, torch.Tensor] = {}
    processed = 0
    started = time.perf_counter()
    for rgb, sar, names in loader:
        if max_images > 0 and processed >= max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        _, logits = model(rgb, p3_sar_view(sar))
        prediction = logits[0].float().argmax(0).cpu()
        name = str(names[0])
        target = load_semantic_target(str(dataset.dataset_path), name)
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"prediction/semantic shape mismatch for {name}: "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
            )
        histogram = fast_histogram(prediction, target)
        overall += histogram
        event = name.rsplit("_", 1)[0]
        event_histograms.setdefault(event, torch.zeros(4, 4, dtype=torch.float64))
        event_histograms[event] += histogram
        processed += 1
        if processed % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Pure semantic re-audit {variant}: {processed}/{len(dataset)} | "
                f"elapsed={elapsed:.1f}s | rate={processed/max(elapsed, 1e-9):.2f} img/s",
                flush=True,
            )
    report = metrics_from_histogram(overall)
    report["confusion"] = overall.long().tolist()  # type: ignore[assignment]
    events = {
        event: metrics_from_histogram(histogram)
        for event, histogram in sorted(event_histograms.items())
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, events, processed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("P3/T0/T1 pure semantic validation re-audit")
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--t0_checkpoint", required=True)
    parser.add_argument("--t1_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--expected_images", type=int, required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--expected_p3_valid_miou", type=float, default=None)
    parser.add_argument("--reproduction_tolerance_pp", type=float, default=0.05)
    parser.add_argument("--original_standard_test_accessed", action="store_true")
    args = parser.parse_args(argv)

    paths = {
        "p3_locked_source": args.p3_checkpoint,
        "t0_original_objective": args.t0_checkpoint,
        "t1_component_balanced": args.t1_checkpoint,
    }
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    if any(not isinstance(value, Mapping) for value in checkpoints.values()):
        raise TypeError("all checkpoints must be mappings")
    pair_manifest = validate_pair_metadata(
        checkpoints["t0_original_objective"],
        checkpoints["t1_component_balanced"],
    )
    p3_hash = file_sha256(args.p3_checkpoint)
    if p3_hash != pair_manifest["base_checkpoint_sha256"]:
        raise RuntimeError(
            "P3 file is not the source checkpoint used by T0/T1: "
            f"source={pair_manifest['base_checkpoint_sha256']}, observed={p3_hash}"
        )

    dataset = H9RGBSARInferenceDataset(
        args.dataset_path, args.val_list, args.sar_stats_json
    )
    if len(dataset) != args.expected_images:
        raise RuntimeError(
            f"split size mismatch: expected={args.expected_images}, observed={len(dataset)}"
        )
    device = torch.device(args.device)
    reports: Dict[str, Mapping[str, object]] = {}
    event_reports: Dict[str, Dict[str, Mapping[str, float]]] = {}
    processed_counts = []
    for variant in VARIANTS:
        report, events, processed = evaluate_checkpoint(
            checkpoints[variant],
            dataset=dataset,
            device=device,
            num_workers=args.num_workers,
            max_images=args.max_images,
            progress_every=args.progress_every,
            variant=variant,
        )
        reports[variant] = report
        event_reports[variant] = events
        processed_counts.append(processed)
    if len(set(processed_counts)) != 1:
        raise RuntimeError(f"variants processed different image counts: {processed_counts}")

    gains = {
        "t0_minus_p3_pp": metric_gain_pp(
            reports["t0_original_objective"], reports["p3_locked_source"]
        ),
        "t1_minus_t0_pp": metric_gain_pp(
            reports["t1_component_balanced"], reports["t0_original_objective"]
        ),
        "t1_minus_p3_pp": metric_gain_pp(
            reports["t1_component_balanced"], reports["p3_locked_source"]
        ),
    }
    reproduction = None
    if args.expected_p3_valid_miou is not None:
        observed = float(reports["p3_locked_source"]["valid_miou"])
        error_pp = 100.0 * (observed - args.expected_p3_valid_miou)
        reproduction = {
            "expected_valid_miou": args.expected_p3_valid_miou,
            "observed_valid_miou": observed,
            "error_pp": error_pp,
            "tolerance_pp": args.reproduction_tolerance_pp,
            "passed": abs(error_pp) <= args.reproduction_tolerance_pp,
        }
    events = {}
    for event in sorted(event_reports["p3_locked_source"]):
        table = {variant: event_reports[variant][event] for variant in VARIANTS}
        events[event] = {
            "reports": table,
            "t1_minus_t0_pp": metric_gain_pp(
                table["t1_component_balanced"], table["t0_original_objective"]
            ),
            "t1_minus_p3_pp": metric_gain_pp(
                table["t1_component_balanced"], table["p3_locked_source"]
            ),
        }
    payload = {
        "protocol": "SPI_T0_T1_PURE_SEMANTIC_REAUDIT_v1",
        "split": args.split_name,
        "semantic_ground_truth": "dataset/target/*_building_damage.tif",
        "instance_annotations_accessed": False,
        "test_accessed": bool(args.original_standard_test_accessed),
        "test_protocol": "original_BRIGHT_standard_ML",
        "processed_images": processed_counts[0],
        "pair_manifest": pair_manifest,
        "checkpoints": {
            name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "reports": reports,
        "gains": gains,
        "p3_reference_reproduction": reproduction,
        "events": events,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 112)
    print(
        f"Pure semantic re-audit | split={args.split_name} | "
        "semantic TIFF only | instances=NEVER | "
        f"original-standard-test-accessed={args.original_standard_test_accessed}"
    )
    print(
        f"{'Variant':<28} {'Valid mIoU':>12} {'I IoU':>10} "
        f"{'D IoU':>10} {'X IoU':>10} {'D->I':>10} {'X->I':>10}"
    )
    for variant in VARIANTS:
        row = reports[variant]
        print(
            f"{variant:<28} {100*row['valid_miou']:12.4f} "
            f"{100*row['intact_iou']:10.4f} {100*row['damaged_iou']:10.4f} "
            f"{100*row['destroyed_iou']:10.4f} "
            f"{100*row['damaged_to_intact']:10.4f} "
            f"{100*row['destroyed_to_intact']:10.4f}"
        )
    print(f"T0-P3 pp: {gains['t0_minus_p3_pp']}")
    print(f"T1-T0 pp: {gains['t1_minus_t0_pp']}")
    if reproduction is not None:
        print(
            "P3 reference reproduction: "
            f"expected={100*reproduction['expected_valid_miou']:.4f}% "
            f"observed={100*reproduction['observed_valid_miou']:.4f}% "
            f"error={reproduction['error_pp']:+.4f} pp "
            f"passed={reproduction['passed']}"
        )
    print(f"Saved: {output}")
    print(
        "Instance annotations/original-standard-test accessed: "
        f"False / {args.original_standard_test_accessed}"
    )
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
