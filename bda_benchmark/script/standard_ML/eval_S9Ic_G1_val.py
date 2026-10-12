#!/usr/bin/env python3
"""Evaluate S9-Ic/S9-Id center diagnostics and decoder behavior on validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"
for import_root in (PROJECT_ROOT, BDA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (
    build_training_bag_specs,
)
from bda_benchmark.model.instance_geometry.config import (
    GeometryDecoderConfig,
    geometry_preset,
)
from bda_benchmark.model.instance_geometry.decode import (
    decode_center_offset_instances,
    decode_center_watershed_instances,
)
from bda_benchmark.model.instance_geometry.metrics import (
    CenterMetricAccumulator,
    evaluate_center_mechanism,
)
from bda_benchmark.model.instance_geometry.s8_backbone import FrozenS8Backbone
from bda_benchmark.script.standard_ML.s9ic_shared import (
    EXPECTED_FEATURE_CHANNELS,
    EXPECTED_FEATURE_KEY,
    TARGET_DEFINITION,
    build_geometry_head,
)
from bda_benchmark.script.standard_ML.s9_instance_common import (
    InstanceMetricAccumulator,
    evaluate_instance_geometry,
    instances_from_bag_specs,
    instances_from_masks,
)
from bda_benchmark.script.standard_ML.train_S9_instance_geometry import (
    _find_key,
    _load_protocol_json,
    _make_loader,
    _resolve_data_argument,
)


def _safe_load(path: Path, allow_unsafe: bool):
    if allow_unsafe:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe geometry loading is unavailable; use --allow_unsafe_checkpoint "
            "only for a trusted checkpoint."
        ) from exc


def _aggregate_decoder(
    aggregate: Dict[str, float],
    report: Mapping[str, Any],
) -> None:
    for key, value in report.items():
        if isinstance(value, bool):
            aggregate.setdefault(key, float(value))
        elif isinstance(value, (int, float)):
            aggregate[key] = aggregate.get(key, 0.0) + float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate S9 geometry on validation")
    parser.add_argument("--s8_checkpoint", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--s9ib_json")
    parser.add_argument("--s9a2b_json")
    parser.add_argument("--s9i_json")
    parser.add_argument("--val_dataset_path")
    parser.add_argument("--val_data_list_path")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int)
    parser.add_argument(
        "--decoder_variant",
        choices=("g1", "g2", "g3", "id"),
        help="Decode the same checkpoint with a locked G1/G2/G3/Id preset.",
    )
    parser.add_argument("--instance_iou_threshold", type=float, default=0.5)
    parser.add_argument("--overlap_threshold", type=float, default=0.10)
    parser.add_argument(
        "--allow_legacy_target_checkpoint",
        action="store_true",
        help="Allow diagnostic evaluation of a checkpoint without the EDT-v2 target tag.",
    )
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    return parser


@torch.no_grad()
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = _load_protocol_json(args.s9ib_json)
    val_dataset_path = _resolve_data_argument(
        args.val_dataset_path,
        protocol,
        "val_dataset_path",
    )
    val_data_list_path = _resolve_data_argument(
        args.val_data_list_path,
        protocol,
        "val_data_list_path",
    )
    device = torch.device(args.device)
    backbone = FrozenS8Backbone.from_checkpoint(
        args.s8_checkpoint,
        feature_key=EXPECTED_FEATURE_KEY,
        expected_channels=EXPECTED_FEATURE_CHANNELS,
        allow_unsafe=args.allow_unsafe_checkpoint,
    ).to(device)

    geometry_path = Path(args.geometry_checkpoint).expanduser().resolve()
    payload = _safe_load(geometry_path, args.allow_unsafe_checkpoint)
    if not isinstance(payload, Mapping):
        raise TypeError("geometry checkpoint must be a mapping")
    metadata = payload.get("metadata", {})
    variant = str(metadata.get("variant", "g1"))
    target_definition = metadata.get("target_definition")
    if (
        target_definition != TARGET_DEFINITION
        and not args.allow_legacy_target_checkpoint
    ):
        raise RuntimeError(
            "geometry checkpoint was not trained with the locked source-scale "
            f"EDT target ({TARGET_DEFINITION}); got {target_definition!r}. "
            "Retrain, or pass --allow_legacy_target_checkpoint only for a "
            "clearly labeled diagnostic evaluation."
        )
    head_config = payload.get("head_config", {})
    if not isinstance(head_config, Mapping):
        raise TypeError("geometry checkpoint head_config must be a mapping")
    in_channels = int(
        head_config.get("in_channels", EXPECTED_FEATURE_CHANNELS)
    )
    hidden_channels = int(head_config.get("hidden_channels", 64))
    if in_channels != EXPECTED_FEATURE_CHANNELS:
        raise RuntimeError(
            f"geometry checkpoint expects {in_channels} S8 channels; "
            f"this integration is locked to {EXPECTED_FEATURE_CHANNELS}"
        )
    head = build_geometry_head(
        variant,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    ).to(device)
    state = payload.get("head_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("geometry checkpoint is missing head_state_dict")
    head.load_state_dict(state, strict=True)
    head.eval()

    decoder_payload = payload.get("decoder_config")
    checkpoint_decoder = (
        GeometryDecoderConfig(**dict(decoder_payload))
        if isinstance(decoder_payload, Mapping)
        else geometry_preset(variant)
    )
    decoder_variant = args.decoder_variant or variant
    if decoder_variant == "id" and variant != "id":
        raise ValueError("S9-Id decoding requires a checkpoint with an offset head")
    if variant == "id" and decoder_variant != "id":
        raise ValueError("an S9-Id checkpoint cannot use a watershed decoder")
    if args.decoder_variant is None:
        decoder = checkpoint_decoder
    else:
        preset = geometry_preset(decoder_variant)
        decoder = checkpoint_decoder.with_overrides(
            name=preset.name,
            watershed_enabled=preset.watershed_enabled,
            boundary_gate_enabled=preset.boundary_gate_enabled,
            rollback_enabled=preset.rollback_enabled,
        )
    loader = _make_loader(
        val_dataset_path,
        val_data_list_path,
        crop_size=1024,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train=False,
    )

    center_accumulator = CenterMetricAccumulator()
    instance_accumulator = InstanceMetricAccumulator()
    decoder_aggregate: Dict[str, float] = {}
    processed_images = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        pre, post, labels_loc, labels_damage, names = batch
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        frozen = backbone(pre, post)
        prediction = head(frozen.feature)
        center = torch.sigmoid(prediction["center_logits"])
        boundary = torch.sigmoid(prediction["boundary_logits"])
        offset = prediction.get("offset_logits")
        specs = build_training_bag_specs(
            labels_loc.long(),
            labels_damage.long(),
            min_area=32,
        )
        building = frozen.auxiliary["p_building"]

        for item_index in range(pre.shape[0]):
            item_specs = [
                spec
                for spec in specs
                if int(spec.batch_index) == item_index
            ]
            center_report = evaluate_center_mechanism(
                center[item_index, 0],
                [spec.mask for spec in item_specs],
            )
            center_accumulator.update(center_report)
            if offset is not None:
                decoded_masks, decoder_report = decode_center_offset_instances(
                    building[item_index, 0],
                    center[item_index, 0],
                    offset[item_index],
                    config=decoder,
                )
            else:
                decoded_masks, decoder_report = decode_center_watershed_instances(
                    building[item_index, 0],
                    center[item_index, 0],
                    boundary[item_index, 0],
                    config=decoder,
                )
            target_instances = instances_from_bag_specs(item_specs)
            predicted_instances = instances_from_masks(
                decoded_masks,
                batch_index=item_index,
            )
            instance_accumulator.update(
                evaluate_instance_geometry(
                    predicted_instances,
                    target_instances,
                    iou_threshold=args.instance_iou_threshold,
                    overlap_threshold=args.overlap_threshold,
                )
            )
            _aggregate_decoder(decoder_aggregate, decoder_report)
            processed_images += 1

    report = {
        "protocol": "validation_only",
        "variant": decoder_variant,
        "checkpoint_variant": variant,
        "target_definition": target_definition,
        "decoder_variant": decoder_variant,
        "processed_images": processed_images,
        "center_metrics": center_accumulator.compute(),
        "instance_geometry": instance_accumulator.compute(),
        "decoder_totals": decoder_aggregate,
        "decoder_config": decoder.to_dict(),
        "instance_evaluation_config": {
            "iou_threshold": args.instance_iou_threshold,
            "overlap_threshold": args.overlap_threshold,
            "target_source": "GT-derived pseudo-instance",
        },
        "checkpoints": {
            "s8": str(Path(args.s8_checkpoint).expanduser().resolve()),
            "geometry": str(geometry_path),
        },
        "provenance_json": {
            "s9ib": args.s9ib_json,
            "s9a2b": args.s9a2b_json,
            "s9i": args.s9i_json,
        },
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
