# -*- coding: utf-8 -*-
"""Independent full validation/test evaluator for A2-DCSM."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_A2_DCSM import (  # noqa: E402
    VALID_A2_DCSM_VARIANTS,
    build_a2_dcsm_damageformer,
)
from bda_benchmark.script.standard_ML.train_A2_DCSM import (  # noqa: E402
    configure_freeze_policy,
    keep_frozen_modules_in_eval,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    evaluate,
    fast_hist,
    metrics_line,
    set_seed,
)


VALID_PROTOCOLS = ("A2_DCSM_best_weights_v1", "A2_DCSM_resume_v1")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_damage_hist(histogram: torch.Tensor) -> dict[str, float]:
    histogram = histogram.double()
    intersection = torch.diag(histogram)
    union = histogram.sum(1) + histogram.sum(0) - intersection
    iou = intersection / union.clamp_min(1.0)
    return {
        "miou4": float(iou.mean().item()),
        "valid_miou": float(iou[1:].mean().item()),
        "background_iou": float(iou[0].item()),
        "intact_iou": float(iou[1].item()),
        "damaged_iou": float(iou[2].item()),
        "destroyed_iou": float(iou[3].item()),
    }


@torch.no_grad()
def eventwise_audit(model, loader, device, ignore_index: int):
    histograms: dict[str, torch.Tensor] = {}
    model.eval()
    for rgb, sar, _, damage, names in loader:
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        prediction = model(rgb, sar, return_dict=True)["output_dam"].argmax(1)
        for index, sample_name in enumerate(names):
            event = re.sub(r"_\d+$", "", str(sample_name))
            sample_hist = fast_hist(
                prediction[index : index + 1],
                damage[index : index + 1],
                4,
                ignore_index,
            ).cpu()
            histograms[event] = histograms.get(
                event, torch.zeros_like(sample_hist)
            ) + sample_hist
    events = {
        event: metrics_from_damage_hist(histogram)
        for event, histogram in sorted(histograms.items())
    }
    grouped = defaultdict(list)
    for event, metrics in events.items():
        disaster_type = event.rsplit("-", 1)[-1]
        grouped[disaster_type].append(metrics)
    disaster_types = {}
    for disaster_type, values in sorted(grouped.items()):
        disaster_types[disaster_type] = {
            "event_count": len(values),
            **{
                key: sum(item[key] for item in values) / len(values)
                for key in values[0]
            },
        }
    return events, disaster_types


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate A2-DCSM on validation or test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol == "A2_DCSM_pathway_v1":
        raise RuntimeError(
            "received pathway_final.pth, which contains only the intermediate "
            "pretrained SAR pathway and cannot be evaluated directly. Use the "
            "supervised experiment checkpoint: "
            "A2_M*/best_val_miou.pth (protocol A2_DCSM_best_weights_v1)."
        )
    if protocol not in VALID_PROTOCOLS:
        raise RuntimeError(f"unexpected A2-DCSM checkpoint protocol: {protocol!r}")
    if variant not in VALID_A2_DCSM_VARIANTS:
        raise RuntimeError(f"unexpected A2-DCSM variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("A2-DCSM checkpoint has no model state")

    model = build_a2_dcsm_damageformer(variant, pretrained_rgb=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load unexpectedly failed: {incompatible}")
    configure_freeze_policy(model)
    model.to(device).eval()
    keep_frozen_modules_in_eval(model)

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split=args.split,
        variant="h3",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    observed = evaluate(model, loader, device, args.ignore_index, max_batches=0)
    events, disaster_types = eventwise_audit(
        model, loader, device, args.ignore_index
    )
    stored = checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
    deltas = {}
    if args.split == "val" and isinstance(stored, Mapping):
        for key, value in observed.items():
            if key in stored and isinstance(stored[key], (int, float)):
                deltas[key] = float(value) - float(stored[key])

    payload = {
        "protocol": "A2_DCSM_independent_evaluation_v1",
        "split": args.split,
        "test_accessed": args.split == "test",
        "processed_images": int(observed.get("processed_images", 0)),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
            "protocol": protocol,
            "variant": variant,
            "step": int(checkpoint.get("step", 0)),
            "selector": checkpoint.get("selector"),
            "strict_load": True,
        },
        "stored_metrics": dict(stored) if isinstance(stored, Mapping) else {},
        "observed_metrics": observed,
        "event_metrics": events,
        "disaster_type_macro_metrics": disaster_types,
        "observed_minus_stored": deltas,
        "data": {
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "list_path": str(Path(args.data_list_path).resolve()),
            "sar_stats_path": str(Path(args.sar_stats_path).resolve()),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(metrics_line(f"Independent {args.split}", observed), flush=True)
    print("Strict checkpoint load: True", flush=True)
    print(f"Processed images: {payload['processed_images']}", flush=True)
    print("Event/disaster-type audit:", flush=True)
    for name, values in disaster_types.items():
        print(
            f"  {name:<12} N={values['event_count']:>2} | "
            f"mIoU4={100*values['miou4']:.4f} | "
            f"Dmg={100*values['damaged_iou']:.4f} | "
            f"Des={100*values['destroyed_iou']:.4f}",
            flush=True,
        )
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
