#!/usr/bin/env python3
"""Raw-output paired audit for SPI T0 versus T1 on public validation only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import imageio.v2 as imageio
import numpy as np
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
    coco_instances,
)
from bda_benchmark.script.standard_ML.spi_stage0.annotations import (  # noqa: E402
    load_stage0_instance_records,
)
from bda_benchmark.script.standard_ML.spi_stage0.core import (  # noqa: E402
    finalize_variant_accumulator,
    gt_target_raster,
    merge_variant_accumulator,
    new_variant_accumulator,
    update_variant_accumulator,
)
from bda_benchmark.script.standard_ML.spi_t1.core import (  # noqa: E402
    SOURCE_VARIANT,
    VARIANTS,
    build_t1_decision,
    paired_gain,
)


PAIR_FIELDS = (
    "dataset_path",
    "train_list",
    "seed",
    "crop_size",
    "batch_size",
    "accumulation_steps",
    "optimizer_steps",
    "learning_rate",
    "weight_decay",
    "amp_dtype",
    "ignore_index",
    "minimum_component_area",
    "maximum_bags_per_class",
    "lambda_mil",
    "lambda_exist",
    "lambda_severity",
    "lambda_intact_guard",
)


def load_semantic_target(dataset_path: str, sample_name: str) -> torch.Tensor:
    """Load the original BRIGHT semantic raster used by the P3 protocol."""

    path = Path(dataset_path) / "target" / f"{sample_name}_building_damage.tif"
    if not path.is_file():
        raise FileNotFoundError(f"missing semantic target raster: {path}")
    array = np.asarray(imageio.imread(path))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            f"semantic target must be a two-dimensional label raster: "
            f"path={path}, shape={array.shape}"
        )
    target = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.long)
    valid = target != 255
    if bool(valid.any()) and bool(((target[valid] < 0) | (target[valid] > 3)).any()):
        values = torch.unique(target[valid]).tolist()
        raise ValueError(f"unexpected semantic labels in {path}: {values}")
    return target


def finalize_dual_ground_truth_report(
    instance_accumulator: Mapping[str, object],
    semantic_histogram: torch.Tensor,
) -> Dict[str, object]:
    """Use semantic TIFFs for semantic metrics and instances for building metrics.

    The former ``semantic`` result is retained under
    ``instance_raster_semantic`` so annotation-domain effects remain visible.
    """

    report = finalize_variant_accumulator(instance_accumulator)
    instance_raster_semantic = report["semantic"]
    report["semantic"] = metrics_from_histogram(semantic_histogram)
    report["instance_raster_semantic"] = instance_raster_semantic
    return report


def validate_pair_metadata(
    t0: Mapping[str, object], t1: Mapping[str, object]
) -> Dict[str, object]:
    for name, checkpoint, expected_variant in (
        ("T0", t0, VARIANTS[0]),
        ("T1", t1, VARIANTS[1]),
    ):
        if checkpoint.get("protocol") != "SPI_T0_T1_FIXED_STEP_PAIRED_v1":
            raise RuntimeError(f"{name} checkpoint has an unexpected protocol")
        if checkpoint.get("variant") != expected_variant:
            raise RuntimeError(f"{name} checkpoint variant mismatch")
        if bool(checkpoint.get("test_accessed", True)):
            raise RuntimeError(f"{name} checkpoint does not certify test isolation")
    if t0.get("base_checkpoint_sha256") != t1.get("base_checkpoint_sha256"):
        raise RuntimeError("T0 and T1 did not start from the same P3 checkpoint")
    if int(t0.get("step", -1)) != int(t1.get("step", -2)):
        raise RuntimeError("T0 and T1 have different optimizer-step counts")
    t0_args, t1_args = t0.get("args"), t1.get("args")
    if not isinstance(t0_args, Mapping) or not isinstance(t1_args, Mapping):
        raise RuntimeError("paired checkpoints are missing frozen arguments")
    mismatches = {
        field: (t0_args.get(field), t1_args.get(field))
        for field in PAIR_FIELDS
        if t0_args.get(field) != t1_args.get(field)
    }
    if mismatches:
        raise RuntimeError(f"T0/T1 paired-control mismatch: {mismatches}")
    if float(t0.get("effective_lambda_cb", -1.0)) != 0.0:
        raise RuntimeError("T0 effective_lambda_cb must be zero")
    if float(t1.get("effective_lambda_cb", 0.0)) <= 0.0:
        raise RuntimeError("T1 effective_lambda_cb must be positive")
    return {
        "base_checkpoint_sha256": t0.get("base_checkpoint_sha256"),
        "seed": int(t0.get("seed", -1)),
        "optimizer_steps": int(t0.get("step", -1)),
        "t1_lambda_cb": float(t1.get("effective_lambda_cb", 0.0)),
        "matched_fields": list(PAIR_FIELDS),
    }


@torch.inference_mode()
def evaluate_one(
    checkpoint: Mapping[str, object],
    *,
    dataset: H9RGBSARInferenceDataset,
    records: Mapping[str, Mapping[str, object]],
    device: torch.device,
    num_workers: int,
    minimum_gt_area: int,
    max_batches: int,
    progress_every: int,
    variant: str,
) -> tuple[Dict[str, object], Dict[str, Dict[str, object]], int]:
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
    overall = new_variant_accumulator()
    semantic_histogram = torch.zeros(4, 4, dtype=torch.float64)
    events: MutableMapping[str, Dict[str, object]] = {}
    event_semantic_histograms: MutableMapping[str, torch.Tensor] = {}
    processed = 0
    started = time.perf_counter()
    for batch_index, (rgb, sar, names) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        _, damage_logits = model(rgb, p3_sar_view(sar))
        prediction = damage_logits.float().argmax(1).cpu()
        for sample_index, raw_name in enumerate(names):
            name = str(raw_name)
            record = records[name]
            image = record["image"]
            assert isinstance(image, Mapping)
            height, width = int(image["height"]), int(image["width"])
            predicted_map = prediction[sample_index]
            if tuple(predicted_map.shape) != (height, width):
                raise RuntimeError(
                    f"prediction/annotation mismatch for {name}: "
                    f"prediction={tuple(predicted_map.shape)}, target={(height, width)}"
                )
            instances = coco_instances(record, minimum_area=minimum_gt_area)
            instance_target_map = gt_target_raster(
                instances, height=height, width=width
            )
            semantic_target_map = load_semantic_target(dataset.dataset_path, name)
            if tuple(semantic_target_map.shape) != (height, width):
                raise RuntimeError(
                    f"semantic/instance geometry mismatch for {name}: "
                    f"semantic={tuple(semantic_target_map.shape)}, "
                    f"instance={(height, width)}"
                )
            sample = new_variant_accumulator()
            update_variant_accumulator(
                sample, predicted_map, instance_target_map, instances
            )
            merge_variant_accumulator(overall, sample)
            semantic_histogram.add_(
                fast_histogram(predicted_map, semantic_target_map)
            )
            event = name.rsplit("_", 1)[0]
            if event not in events:
                events[event] = new_variant_accumulator()
                event_semantic_histograms[event] = torch.zeros(
                    4, 4, dtype=torch.float64
                )
            merge_variant_accumulator(events[event], sample)
            event_semantic_histograms[event].add_(
                fast_histogram(predicted_map, semantic_target_map)
            )
            processed += 1
            if processed % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"SPI T1 audit {variant}: {processed}/{len(dataset)} | "
                    f"elapsed={elapsed:.1f}s | rate={processed / max(elapsed, 1e-9):.2f} img/s",
                    flush=True,
                )
    report = finalize_dual_ground_truth_report(overall, semantic_histogram)
    event_reports = {
        name: finalize_dual_ground_truth_report(
            accumulator, event_semantic_histograms[name]
        )
        for name, accumulator in events.items()
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, event_reports, processed


def _write_event_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "event",
        "images",
        "t0_valid_miou",
        "t1_valid_miou",
        "t1_vs_t0_valid_miou_gain_pp",
        "t0_instance_raster_valid_miou",
        "t1_instance_raster_valid_miou",
        "t0_building_macro_f1",
        "t1_building_macro_f1",
        "t1_vs_t0_building_macro_f1_gain_pp",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            reports = row["reports"]
            gain = row["gain_pp"]
            assert isinstance(reports, Mapping) and isinstance(gain, Mapping)
            t0, t1 = reports[VARIANTS[0]], reports[VARIANTS[1]]
            writer.writerow(
                {
                    "event": row["event"],
                    "images": row["images"],
                    "t0_valid_miou": t0["semantic"]["valid_miou"],
                    "t1_valid_miou": t1["semantic"]["valid_miou"],
                    "t1_vs_t0_valid_miou_gain_pp": gain["valid_miou_pp"],
                    "t0_instance_raster_valid_miou": t0[
                        "instance_raster_semantic"
                    ]["valid_miou"],
                    "t1_instance_raster_valid_miou": t1[
                        "instance_raster_semantic"
                    ]["valid_miou"],
                    "t0_building_macro_f1": t0["building"]["macro_f1_present_gt"],
                    "t1_building_macro_f1": t1["building"]["macro_f1_present_gt"],
                    "t1_vs_t0_building_macro_f1_gain_pp": gain["building_macro_f1_pp"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SPI T0/T1 raw paired public-validation audit")
    parser.add_argument("--t0_checkpoint", required=True)
    parser.add_argument("--t1_checkpoint", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    annotations = parser.add_mutually_exclusive_group(required=True)
    annotations.add_argument("--instance_coco_json", default="")
    annotations.add_argument("--instance_json_dir", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_event_tsv", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--minimum_gt_area", type=int, default=1)
    parser.add_argument("--expected_images", type=int, default=697)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--minimum_valid_miou_gain_pp", type=float, default=0.5)
    parser.add_argument("--minimum_damaged_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_building_macro_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    parser.add_argument("--maximum_intact_iou_drop_pp", type=float, default=0.5)
    args = parser.parse_args(argv)

    t0 = torch.load(args.t0_checkpoint, map_location="cpu", weights_only=False)
    t1 = torch.load(args.t1_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(t0, Mapping) or not isinstance(t1, Mapping):
        raise TypeError("T0 and T1 checkpoints must be mappings")
    pair_manifest = validate_pair_metadata(t0, t1)
    observed_base_hash = file_sha256(args.base_checkpoint)
    if observed_base_hash != pair_manifest["base_checkpoint_sha256"]:
        raise RuntimeError(
            "audit base checkpoint does not match the source used by T0/T1: "
            f"expected={pair_manifest['base_checkpoint_sha256']}, observed={observed_base_hash}"
        )
    source = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise TypeError("locked source P3 checkpoint must be a mapping")
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.val_list, args.sar_stats_json)
    if args.expected_images > 0 and len(dataset) != args.expected_images:
        raise RuntimeError(
            f"public validation split mismatch: expected={args.expected_images}, observed={len(dataset)}"
        )
    records, annotation_manifest = load_stage0_instance_records(
        dataset.data_list,
        unified_coco_json=args.instance_coco_json,
        per_image_json_dir=args.instance_json_dir,
    )
    device = torch.device(args.device)
    reports: Dict[str, Mapping[str, object]] = {}
    event_tables: Dict[str, Dict[str, Mapping[str, object]]] = {}
    processed_counts = []
    for variant, checkpoint in (
        (SOURCE_VARIANT, source),
        (VARIANTS[0], t0),
        (VARIANTS[1], t1),
    ):
        report, event_reports, processed = evaluate_one(
            checkpoint,
            dataset=dataset,
            records=records,
            device=device,
            num_workers=args.num_workers,
            minimum_gt_area=args.minimum_gt_area,
            max_batches=args.max_batches,
            progress_every=args.progress_every,
            variant=variant,
        )
        reports[variant] = report
        processed_counts.append(processed)
        for event, event_report in event_reports.items():
            event_tables.setdefault(event, {})[variant] = event_report
    if len(set(processed_counts)) != 1:
        raise RuntimeError(f"T0/T1 processed different image counts: {processed_counts}")

    event_rows = []
    for event, table in sorted(event_tables.items()):
        expected_variants = {SOURCE_VARIANT, *VARIANTS}
        if set(table) != expected_variants:
            raise RuntimeError(f"event {event} is missing one source/paired variant")
        gain = paired_gain(table[VARIANTS[1]], table[VARIANTS[0]])
        event_rows.append(
            {
                "event": event,
                "images": int(table[VARIANTS[0]]["images"]),
                "reports": table,
                "gain_pp": gain,
                "t1_vs_t0_building_macro_f1_gain_pp": gain["building_macro_f1_pp"],
            }
        )
    decision = build_t1_decision(
        reports,
        event_rows,
        minimum_valid_miou_gain_pp=args.minimum_valid_miou_gain_pp,
        minimum_damaged_f1_gain_pp=args.minimum_damaged_f1_gain_pp,
        minimum_building_macro_f1_gain_pp=args.minimum_building_macro_f1_gain_pp,
        minimum_nonnegative_events=args.minimum_nonnegative_events,
        maximum_intact_iou_drop_pp=args.maximum_intact_iou_drop_pp,
    )
    output_path = Path(args.output_json)
    event_path = Path(args.output_event_tsv) if args.output_event_tsv else output_path.with_suffix(".events.tsv")
    payload = {
        "protocol": "SPI_T0_T1_RAW_PAIRED_DUAL_GT_AUDIT_v2",
        "split": "cvprw26_public_val",
        "test_accessed": False,
        "processed_images": processed_counts[0],
        "pair_manifest": pair_manifest,
        "annotation_manifest": annotation_manifest,
        "metric_ground_truth": {
            "semantic": "dataset/target/*_building_damage.tif",
            "building_and_geometry": "official_instance_json",
            "instance_raster_semantic_retained": True,
        },
        "reports": reports,
        "events": event_rows,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_event_tsv(event_path, event_rows)

    print("=" * 104)
    print(
        "SPI T0/T1 raw paired verifier | semantic TIFF primary | "
        "official instances for building metrics"
    )
    for variant in (SOURCE_VARIANT, *VARIANTS):
        sem = reports[variant]["semantic"]
        instance_sem = reports[variant]["instance_raster_semantic"]
        building = reports[variant]["building"]
        print(
            f"{variant:<28} valid-mIoU={sem['valid_miou']:.4f} "
            f"instanceGT-mIoU={instance_sem['valid_miou']:.4f} "
            f"D-F1={sem['damaged_f1']:.4f} B-macro={building['macro_f1_present_gt']:.4f}"
        )
    print(f"Gain pp : {decision['gain_pp']}")
    print(f"Checks  : {decision['checks']}")
    print(f"Decision: {decision['decision']}")
    print(f"Saved   : {output_path}")
    print("Test accessed: False")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
