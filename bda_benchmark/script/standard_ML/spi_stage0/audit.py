# -*- coding: utf-8 -*-
"""Same-split, no-training Stage-0 semantic-to-pseudo-instance audit.

The script is intentionally validation-only.  It never modifies P3/A2/RGID,
never reads the hidden test split, and never treats official instances as a
deployable input.  Its sole purpose is to decide whether T1 joint training is
scientifically justified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[3]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    build_h6_sdcr_damageformer,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (  # noqa: E402
    EXPECTED_P3_SHA256,
    VALID_A2_PROTOCOLS,
    checkpoint_state,
    file_sha256,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import (  # noqa: E402
    H9RGBSARInferenceDataset,
    coco_instances,
    components_from_localization,
    index_prediction_json,
    predicted_coco_instances,
)
from bda_benchmark.script.standard_ML.spi_stage0.annotations import (  # noqa: E402
    load_stage0_instance_records,
)
from bda_benchmark.script.standard_ML.spi_stage0.core import (  # noqa: E402
    STAGE0_VARIANTS,
    build_stage0_decision,
    damage_class_components,
    evaluate_geometry,
    finalize_variant_accumulator,
    gt_target_raster,
    merge_variant_accumulator,
    new_variant_accumulator,
    p3_a2_gt_instance_oracle_raster,
    pooled_instance_raster,
    update_variant_accumulator,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


def _new_variant_table(include_rgid: bool) -> Dict[str, Dict[str, object]]:
    return {
        name: new_variant_accumulator()
        for name in STAGE0_VARIANTS
        if include_rgid or name != "e3_learned_rgid"
    }


def _finalize_table(table: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    return {
        name: finalize_variant_accumulator(accumulator)
        for name, accumulator in table.items()
    }


def _gain(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> Dict[str, float]:
    candidate_sem, baseline_sem = candidate["semantic"], baseline["semantic"]
    candidate_build, baseline_build = candidate["building"], baseline["building"]
    assert isinstance(candidate_sem, Mapping) and isinstance(baseline_sem, Mapping)
    assert isinstance(candidate_build, Mapping) and isinstance(baseline_build, Mapping)
    return {
        "valid_miou_pp": 100.0
        * (float(candidate_sem["valid_miou"]) - float(baseline_sem["valid_miou"])),
        "damaged_f1_pp": 100.0
        * (float(candidate_sem["damaged_f1"]) - float(baseline_sem["damaged_f1"])),
        "intact_iou_pp": 100.0
        * (float(candidate_sem["intact_iou"]) - float(baseline_sem["intact_iou"])),
        "building_macro_f1_pp": 100.0
        * (
            float(candidate_build["macro_f1_present_gt"])
            - float(baseline_build["macro_f1_present_gt"])
        ),
    }


def _event_row(event: str, table: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    reports = _finalize_table(table)
    baseline = reports["e0_raw_p3"]
    output: Dict[str, object] = {
        "event": event,
        "images": int(baseline["images"]),  # type: ignore[index]
        "variants": reports,
        "gains_vs_e0": {
            name: _gain(report, baseline)  # type: ignore[arg-type]
            for name, report in reports.items()
            if name != "e0_raw_p3"
        },
    }
    output["e4_vs_e0_building_macro_f1_gain_pp"] = output["gains_vs_e0"][  # type: ignore[index]
        "e4_gt_instance_pool"
    ]["building_macro_f1_pp"]
    return output


def _write_event_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "event",
        "images",
        "e0_valid_miou",
        "e4_valid_miou",
        "e4_valid_miou_gain_pp",
        "e0_building_macro_f1",
        "e4_building_macro_f1",
        "e4_building_macro_f1_gain_pp",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            variants = row["variants"]
            gains = row["gains_vs_e0"]
            assert isinstance(variants, Mapping) and isinstance(gains, Mapping)
            e0, e4 = variants["e0_raw_p3"], variants["e4_gt_instance_pool"]
            assert isinstance(e0, Mapping) and isinstance(e4, Mapping)
            e0_sem, e4_sem = e0["semantic"], e4["semantic"]
            e0_build, e4_build = e0["building"], e4["building"]
            e4_gain = gains["e4_gt_instance_pool"]
            writer.writerow(
                {
                    "event": row["event"],
                    "images": row["images"],
                    "e0_valid_miou": e0_sem["valid_miou"],  # type: ignore[index]
                    "e4_valid_miou": e4_sem["valid_miou"],  # type: ignore[index]
                    "e4_valid_miou_gain_pp": e4_gain["valid_miou_pp"],  # type: ignore[index]
                    "e0_building_macro_f1": e0_build["macro_f1_present_gt"],  # type: ignore[index]
                    "e4_building_macro_f1": e4_build["macro_f1_present_gt"],  # type: ignore[index]
                    "e4_building_macro_f1_gain_pp": e4_gain["building_macro_f1_pp"],  # type: ignore[index]
                }
            )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SPI Stage-0 same-split no-training audit")
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--a2_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    annotation_group = parser.add_mutually_exclusive_group(required=True)
    annotation_group.add_argument("--instance_coco_json", default="")
    annotation_group.add_argument("--instance_json_dir", default="")
    parser.add_argument("--rgid_predictions_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_event_tsv", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--minimum_gt_area", type=int, default=1)
    parser.add_argument("--minimum_component_area", type=int, default=32)
    parser.add_argument("--rgid_score_threshold", type=float, default=0.05)
    parser.add_argument("--rgid_mask_nms_iou", type=float, default=0.70)
    parser.add_argument("--match_iou", type=float, default=0.50)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_sha256", default=EXPECTED_P3_SHA256)
    parser.add_argument("--expected_evaluation_images", type=int, default=697)
    parser.add_argument("--minimum_valid_miou_gain_pp", type=float, default=0.5)
    parser.add_argument("--minimum_building_macro_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_damaged_f1_gain_pp", type=float, default=1.0)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    parser.add_argument("--maximum_intact_iou_drop_pp", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.minimum_component_area < 1 or args.minimum_gt_area < 1:
        raise ValueError("minimum areas must be positive")
    if not 0.0 < args.match_iou <= 1.0:
        raise ValueError("match_iou must lie in (0,1]")
    if args.progress_every < 1:
        raise ValueError("progress_every must be positive")
    if args.rgid_predictions_json and not args.instance_coco_json:
        raise RuntimeError(
            "RGID COCO predictions require the unified instance_coco_json so image_id "
            "values belong to the same public validation split"
        )

    set_seed(args.seed)
    device = torch.device(args.device)
    p3_hash = file_sha256(args.p3_checkpoint)
    if args.expected_p3_sha256 and p3_hash != args.expected_p3_sha256:
        raise RuntimeError(
            f"P3 SHA256 mismatch: expected={args.expected_p3_sha256}, observed={p3_hash}"
        )
    p3_checkpoint = torch.load(args.p3_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL()
    incompatible = p3.load_state_dict(checkpoint_state(p3_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")

    a2_checkpoint = torch.load(args.a2_checkpoint, map_location="cpu", weights_only=False)
    if str(a2_checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError("a2_checkpoint is not an H6-A2 checkpoint")
    if a2_checkpoint.get("protocol") not in VALID_A2_PROTOCOLS:
        raise RuntimeError(f"unexpected H6-A2 protocol: {a2_checkpoint.get('protocol')!r}")
    a2 = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = a2.load_state_dict(checkpoint_state(a2_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict A2 load failed: {incompatible}")
    for model in (p3, a2):
        model.to(device).eval().requires_grad_(False)
    print(
        f"SPI Stage-0 execution devices: inference={device} "
        f"pooling={device} geometry={device}",
        flush=True,
    )

    dataset = H9RGBSARInferenceDataset(
        args.dataset_path,
        args.val_data_list_path,
        args.sar_stats_path,
    )
    if args.expected_evaluation_images > 0 and len(dataset) != args.expected_evaluation_images:
        raise RuntimeError(
            f"evaluation split mismatch: expected={args.expected_evaluation_images}, "
            f"observed={len(dataset)}"
        )
    coco_index, annotation_manifest = load_stage0_instance_records(
        dataset.data_list,
        unified_coco_json=args.instance_coco_json,
        per_image_json_dir=args.instance_json_dir,
    )

    rgid_index = None
    if args.rgid_predictions_json:
        payload = json.loads(Path(args.rgid_predictions_json).read_text(encoding="utf-8-sig"))
        rgid_index = index_prediction_json(payload)
    include_rgid = rgid_index is not None

    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    overall = _new_variant_table(include_rgid)
    events: MutableMapping[str, Dict[str, Dict[str, object]]] = {}
    processed_images = 0
    audit_started = time.perf_counter()

    for batch_index, (rgb, sar, names) in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        p3_loc_logits, p3_damage_logits = p3(rgb, p3_sar_view(sar))
        a2_logits = a2(rgb, sar, return_dict=True)["native_output_dam"]
        # Keep region probabilities on CUDA.  The former eager .cpu() made all
        # instance probability pooling CPU-only immediately after model forward.
        p3_probability = p3_damage_logits.float().softmax(1)
        a2_probability = a2_logits.float().softmax(1)
        p3_raw = p3_probability.argmax(1).cpu()
        p3_localization = p3_loc_logits.float().argmax(1).cpu()

        for sample_index, raw_name in enumerate(names):
            name = str(raw_name)
            record = coco_index[name]
            image = record["image"]
            assert isinstance(image, Mapping)
            height, width = int(image["height"]), int(image["width"])
            probability = p3_probability[sample_index]
            native_probability = a2_probability[sample_index]
            if tuple(probability.shape[-2:]) != (height, width):
                raise RuntimeError(
                    f"prediction/annotation geometry mismatch for {name}: "
                    f"prediction={tuple(probability.shape[-2:])}, target={(height, width)}"
                )
            gt_instances = coco_instances(record, minimum_area=args.minimum_gt_area)
            target_map = gt_target_raster(gt_instances, height=height, width=width)
            raw_map = p3_raw[sample_index].cpu()
            localization = p3_localization[sample_index].numpy() > 0

            binary_instances = components_from_localization(
                localization,
                minimum_area=args.minimum_component_area,
            )
            binary_map, _, _, binary_overlap = pooled_instance_raster(
                probability,
                binary_instances,
                height=height,
                width=width,
            )
            damage_instances = damage_class_components(
                localization,
                raw_map.numpy(),
                minimum_area=args.minimum_component_area,
            )
            damage_map, _, _, damage_overlap = pooled_instance_raster(
                probability,
                damage_instances,
                height=height,
                width=width,
            )
            gt_pool_map, _, _, gt_overlap = pooled_instance_raster(
                probability,
                gt_instances,
                height=height,
                width=width,
                preserve_high_score=False,
            )
            oracle_map = p3_a2_gt_instance_oracle_raster(
                probability,
                native_probability,
                gt_instances,
                height=height,
                width=width,
            )

            predictions = {
                "e0_raw_p3": (raw_map, None, 0),
                "e1_binary_cc": (
                    binary_map,
                    evaluate_geometry(
                        binary_instances,
                        gt_instances,
                        match_iou=args.match_iou,
                        device=device,
                    ),
                    binary_overlap,
                ),
                "e2_damage_class_cc": (
                    damage_map,
                    evaluate_geometry(
                        damage_instances,
                        gt_instances,
                        match_iou=args.match_iou,
                        device=device,
                    ),
                    damage_overlap,
                ),
                "e4_gt_instance_pool": (
                    gt_pool_map,
                    evaluate_geometry(
                        gt_instances,
                        gt_instances,
                        match_iou=args.match_iou,
                        device=device,
                    ),
                    gt_overlap,
                ),
                "e5_gt_instance_p3_a2_oracle": (oracle_map, None, gt_overlap),
            }
            if rgid_index is not None:
                rgid_instances = predicted_coco_instances(
                    rgid_index.get(int(image["id"]), []),
                    height=height,
                    width=width,
                    score_threshold=args.rgid_score_threshold,
                    minimum_area=args.minimum_component_area,
                    mask_nms_iou=args.rgid_mask_nms_iou,
                )
                rgid_map, _, _, rgid_overlap = pooled_instance_raster(
                    probability,
                    rgid_instances,
                    height=height,
                    width=width,
                )
                predictions["e3_learned_rgid"] = (
                    rgid_map,
                    evaluate_geometry(
                        rgid_instances,
                        gt_instances,
                        match_iou=args.match_iou,
                        device=device,
                    ),
                    rgid_overlap,
                )

            event = name.rsplit("_", 1)[0]
            if event not in events:
                events[event] = _new_variant_table(include_rgid)
            for variant, (prediction, geometry, overlap) in predictions.items():
                sample_accumulator = new_variant_accumulator()
                update_variant_accumulator(
                    sample_accumulator,
                    prediction,
                    target_map,
                    gt_instances,
                    geometry=geometry,
                    overlap_pixels=overlap,
                )
                merge_variant_accumulator(overall[variant], sample_accumulator)
                merge_variant_accumulator(events[event][variant], sample_accumulator)
            processed_images += 1
            if processed_images % args.progress_every == 0:
                elapsed = time.perf_counter() - audit_started
                print(
                    f"SPI Stage-0: {processed_images}/{len(dataset)} | "
                    f"elapsed={elapsed:.1f}s | rate={processed_images / max(elapsed, 1e-9):.2f} img/s",
                    flush=True,
                )

    reports = _finalize_table(overall)
    event_rows = [_event_row(event, table) for event, table in sorted(events.items())]
    decision = build_stage0_decision(
        reports,  # type: ignore[arg-type]
        event_rows,
        minimum_valid_miou_gain_pp=args.minimum_valid_miou_gain_pp,
        minimum_building_macro_f1_gain_pp=args.minimum_building_macro_f1_gain_pp,
        minimum_damaged_f1_gain_pp=args.minimum_damaged_f1_gain_pp,
        minimum_nonnegative_events=args.minimum_nonnegative_events,
        maximum_intact_iou_drop_pp=args.maximum_intact_iou_drop_pp,
    )
    baseline = reports["e0_raw_p3"]
    gains = {
        name: _gain(report, baseline)  # type: ignore[arg-type]
        for name, report in reports.items()
        if name != "e0_raw_p3"
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_path = (
        Path(args.output_event_tsv)
        if args.output_event_tsv
        else output_path.with_suffix(".events.tsv")
    )
    payload = {
        "protocol": "SPI_STAGE0_same_split_no_training_v1",
        "split": "cvprw26_public_val",
        "test_accessed": False,
        "processed_images": processed_images,
        "execution": {
            "inference_device": str(device),
            "probability_pooling_device": str(device),
            "geometry_device": str(device),
            "elapsed_seconds": time.perf_counter() - audit_started,
        },
        "same_split_contract": True,
        "e3_available": include_rgid,
        "checkpoints": {
            "p3": {
                "path": str(Path(args.p3_checkpoint).resolve()),
                "sha256": p3_hash,
                "strict_load": True,
            },
            "a2": {
                "path": str(Path(args.a2_checkpoint).resolve()),
                "sha256": file_sha256(args.a2_checkpoint),
                "strict_load": True,
            },
        },
        "data_contract": {
            "validation_list": str(Path(args.val_data_list_path).resolve()),
            "official_instances": annotation_manifest,
            "rgid_predictions": (
                str(Path(args.rgid_predictions_json).resolve())
                if args.rgid_predictions_json
                else None
            ),
            "minimum_component_area": args.minimum_component_area,
            "match_iou": args.match_iou,
        },
        "variants": reports,
        "gains_vs_e0": gains,
        "events": event_rows,
        "decision": decision,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_event_tsv(event_path, event_rows)

    print("=" * 104)
    print("SPI Stage-0 | same split | no training | official instances are diagnostic only")
    for name, report in reports.items():
        semantic = report["semantic"]  # type: ignore[index]
        building = report["building"]  # type: ignore[index]
        geometry = report["geometry"]  # type: ignore[index]
        geometry_text = "n/a" if geometry is None else f"{geometry['f1']:.4f}"  # type: ignore[index]
        print(
            f"{name:<34} valid-mIoU={semantic['valid_miou']:.4f} "
            f"D-F1={semantic['damaged_f1']:.4f} "
            f"B-macro={building['macro_f1_present_gt']:.4f} "
            f"geom-F1={geometry_text}"
        )
    print(f"Decision : {decision['recommendation']}")
    print(f"Checks   : {decision['checks']}")
    print(f"Saved    : {output_path}")
    print(f"Events   : {event_path}")
    print("Test accessed: False")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
