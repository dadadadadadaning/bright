#!/usr/bin/env python3
"""One-shot locked Test878 evaluation for frozen P3 + fixed S2 only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
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
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (
    H9RGBSARInferenceDataset,
)
from bda_benchmark.script.standard_ML.spi_geometry_causal.core import (
    event_macro_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    predicted_building_regions,
    prediction_change_audit,
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target


VARIANTS = ("p3", "p3_plus_fixed_s2")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Locked P3 + fixed-S2 Test878 evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--expected_images", type=int, default=878)
    parser.add_argument("--minimum_area", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--bootstrap_iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_miou", type=float, default=0.5740859816060846)
    args = parser.parse_args(argv)

    destination = Path(args.output_json)
    if destination.exists():
        raise FileExistsError(
            f"locked Test output already exists; refusing to overwrite: {destination}"
        )

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
        args.dataset_path, args.test_list, args.sar_stats_json
    )
    if len(dataset) != args.expected_images:
        raise RuntimeError(
            f"Test split mismatch: expected={args.expected_images}, observed={len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    totals = {key: torch.zeros((4, 4), dtype=torch.float64) for key in VARIANTS}
    events: dict[str, dict[str, torch.Tensor]] = {}
    image_rows: list[dict[str, object]] = []
    change_totals = {"changed": 0, "corrected": 0, "harmed": 0, "net": 0}
    accepted_regions = 0
    region_count = 0

    for processed, (rgb, sar, names) in enumerate(loader, start=1):
        name = str(names[0])
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        loc_logits, damage_logits = model(rgb, p3_sar_view(sar))
        probabilities = damage_logits[0].float().softmax(dim=0).cpu()
        raw = probabilities.argmax(dim=0)
        localization = loc_logits[0].float().argmax(dim=0).cpu()
        regions = predicted_building_regions(localization, args.minimum_area)
        region_count += len(regions)
        fixed_s2, records = selective_context_refinement(
            raw,
            probabilities,
            regions,
            config=config,
            mode="soft_region",
        )
        accepted_regions += sum(int(row["accepted"]) for row in records)

        # The target is loaded only after both deployable predictions are frozen.
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        event_table = events.setdefault(
            event,
            {key: torch.zeros((4, 4), dtype=torch.float64) for key in VARIANTS},
        )
        predictions = {"p3": raw, "p3_plus_fixed_s2": fixed_s2}
        histograms: dict[str, list[list[int]]] = {}
        for key, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[key] += histogram
            event_table[key] += histogram
            histograms[key] = histogram.long().tolist()
        audit = prediction_change_audit(raw, fixed_s2, target)
        change_totals["changed"] += int(audit["changed_pixels"])
        change_totals["corrected"] += int(audit["corrected_pixels"])
        change_totals["harmed"] += int(audit["harmed_pixels"])
        change_totals["net"] += int(audit["net_gain_pixels"])
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        if processed % args.progress_every == 0:
            print(f"Locked Test878: {processed}/{len(dataset)}", flush=True)

    reports = {key: metrics_from_histogram(totals[key]) for key in VARIANTS}
    event_reports = {
        event: {key: metrics_from_histogram(table[key]) for key in VARIANTS}
        for event, table in sorted(events.items())
    }
    wins = {
        "improved": sum(
            rows["p3_plus_fixed_s2"]["valid_miou"] > rows["p3"]["valid_miou"]
            for rows in event_reports.values()
        ),
        "nonnegative": sum(
            rows["p3_plus_fixed_s2"]["valid_miou"] >= rows["p3"]["valid_miou"]
            for rows in event_reports.values()
        ),
        "total": len(event_reports),
    }
    p3_miou = reports["p3"]["valid_miou"]
    reproduction = {
        "expected": args.expected_p3_miou,
        "observed": p3_miou,
        "error_pp": 100.0 * (p3_miou - args.expected_p3_miou),
        "passed": abs(p3_miou - args.expected_p3_miou) <= 1e-5,
    }
    bootstrap = stratified_paired_bootstrap(
        image_rows,
        variants=VARIANTS,
        baseline="p3",
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    event_bootstrap = event_macro_paired_bootstrap(
        image_rows,
        variants=VARIANTS,
        baseline="p3",
        comparison_pairs=(("p3_plus_fixed_s2", "p3"),),
        iterations=args.bootstrap_iterations,
        seed=args.seed + 1,
    )
    payload = {
        "protocol": "SPI_SELECTIVE_CONTEXT_LOCKED_TEST878_v1",
        "split": "standard_ml_test878",
        "selection_policy": "P3 and fixed S2 frozen before this run; Test is evaluation-only",
        "test_accessed": True,
        "official_instances_accessed": False,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config": config.__dict__,
        "processed_images": len(dataset),
        "region_count": region_count,
        "accepted_regions": accepted_regions,
        "p3_reproduction": reproduction,
        "reports": reports,
        "gain_vs_p3_pp": 100.0 * (
            reports["p3_plus_fixed_s2"]["valid_miou"] - p3_miou
        ),
        "event_reports": event_reports,
        "event_wins": wins,
        "change_audit": change_totals,
        "event_stratified_image_bootstrap": bootstrap,
        "equal_event_bootstrap": event_bootstrap,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=" * 96)
    print("LOCKED Test878 | frozen P3 vs frozen P3 + fixed S2")
    for key in VARIANTS:
        row = reports[key]
        print(
            f"{key:<20} mIoU={100*row['valid_miou']:.4f} "
            f"D-F1={100*row['damaged_f1']:.4f} "
            f"I/D/X IoU={100*row['intact_iou']:.4f}/"
            f"{100*row['damaged_iou']:.4f}/{100*row['destroyed_iou']:.4f}"
        )
    print(f"gain_vs_p3_pp={payload['gain_vs_p3_pp']:+.4f}")
    print(f"event_wins={wins}")
    print(f"p3_reproduction={reproduction}")
    print(f"Saved: {destination}")
    print("Test accessed: YES; do not tune S2 from this result")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
