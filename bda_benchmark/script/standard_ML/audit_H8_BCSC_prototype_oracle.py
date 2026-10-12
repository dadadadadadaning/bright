# -*- coding: utf-8 -*-
"""Validation-only P0/P1/P2 audit for H8 building prototypes.

P0 disables the H8 prototype refinement, P1 forces the model-derived intact
building prototype, and P2 uses ground-truth intact pixels *only to estimate
the prototype*.  P2 is an analysis oracle and is forbidden on the test split.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H8_BCSC import (  # noqa: E402
    build_h8_bcsc_damageformer,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    fast_hist,
    metrics_from_hist,
    set_seed,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    configure_freeze_policy,
    keep_frozen_modules_in_eval,
)


VALID_PROTOCOLS = ("H8_BCSC_best_weights_v1", "H8_BCSC_resume_v1")


def _new_record(device: torch.device) -> Dict[str, object]:
    return {
        "loc_hist": torch.zeros(2, 2, dtype=torch.float64, device=device),
        "damage_hist": torch.zeros(4, 4, dtype=torch.float64, device=device),
        "event_hist": {},
        "processed": 0,
    }


def _update_record(
    record: Dict[str, object],
    outputs: Mapping[str, torch.Tensor],
    location: torch.Tensor,
    damage: torch.Tensor,
    names,
    ignore_index: int,
) -> None:
    loc_prediction = outputs["output_loc"].argmax(1)
    damage_prediction = outputs["output_dam"].argmax(1)
    record["loc_hist"] += fast_hist(  # type: ignore[operator]
        loc_prediction, location, 2, ignore_index
    )
    record["damage_hist"] += fast_hist(  # type: ignore[operator]
        damage_prediction, damage, 4, ignore_index
    )
    event_hist = record["event_hist"]
    for sample_index in range(damage.shape[0]):
        event = re.sub(r"_\d+$", "", str(names[sample_index]))
        histogram = fast_hist(
            damage_prediction[sample_index : sample_index + 1],
            damage[sample_index : sample_index + 1],
            4,
            ignore_index,
        )
        event_hist[event] = event_hist.get(
            event, torch.zeros_like(histogram)
        ) + histogram
    record["processed"] = int(record["processed"]) + int(damage.shape[0])


def _finalize(record: Mapping[str, object]) -> Dict[str, object]:
    metrics = metrics_from_hist(record["loc_hist"], record["damage_hist"])
    event_metrics: Dict[str, float] = {}
    for event, histogram in record["event_hist"].items():
        intersection = torch.diag(histogram)
        union = histogram.sum(1) + histogram.sum(0) - intersection
        event_metrics[event] = float(
            (intersection / union.clamp_min(1.0)).mean().item()
        )
    values = list(event_metrics.values())
    metrics.update(
        {
            "processed_images": int(record["processed"]),
            "event_count": len(values),
            "event_mean_official_style_4class_miou": (
                float(np.mean(values)) if values else 0.0
            ),
            "event_worst_official_style_4class_miou": (
                float(np.min(values)) if values else 0.0
            ),
        }
    )
    return {"metrics": metrics, "events": event_metrics}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("H8 validation-only prototype oracle audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--blend_values", type=float, nargs="+", default=(0.25, 0.50, 1.00)
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if "test" in Path(args.data_list_path).name.lower():
        raise RuntimeError("prototype oracle is validation-only; test is forbidden")
    blend_values = sorted(set(float(value) for value in args.blend_values))
    if not blend_values or any(not 0.0 < value <= 1.0 for value in blend_values):
        raise ValueError("blend_values must be in (0, 1]")

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol not in VALID_PROTOCOLS or variant not in ("c1", "c2"):
        raise RuntimeError(
            "prototype oracle requires an H8 C1/C2 best or resume checkpoint"
        )
    model = build_h8_bcsc_damageformer(variant, pretrained_rgb=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    configure_freeze_policy(model)
    model.to(device).eval()
    keep_frozen_modules_in_eval(model)

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split="val",
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

    records = {"P0_A2_base": _new_record(device)}
    for blend in blend_values:
        tag = f"{blend:.2f}".replace(".", "p")
        records[f"P1_proxy_blend_{tag}"] = _new_record(device)
        records[f"P2_oracle_blend_{tag}"] = _new_record(device)

    for batch_index, batch in enumerate(loader, start=1):
        rgb, sar, location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()

        outputs = model(
            rgb, sar, return_dict=True, audit_forced_blend=0.0
        )
        _update_record(
            records["P0_A2_base"], outputs, location, damage, names, args.ignore_index
        )
        oracle_intact = (damage == 1).unsqueeze(1).to(rgb.dtype)
        for blend in blend_values:
            tag = f"{blend:.2f}".replace(".", "p")
            proxy = model(
                rgb, sar, return_dict=True, audit_forced_blend=blend
            )
            oracle = model(
                rgb,
                sar,
                return_dict=True,
                audit_oracle_intact_mask=oracle_intact,
                audit_forced_blend=blend,
            )
            _update_record(
                records[f"P1_proxy_blend_{tag}"],
                proxy,
                location,
                damage,
                names,
                args.ignore_index,
            )
            _update_record(
                records[f"P2_oracle_blend_{tag}"],
                oracle,
                location,
                damage,
                names,
                args.ignore_index,
            )
        if batch_index % 50 == 0:
            print(f"H8 prototype oracle: {batch_index}/{len(loader)}", flush=True)

    results = {name: _finalize(record) for name, record in records.items()}
    base = results["P0_A2_base"]["metrics"]
    rows = []
    for name, result in results.items():
        metrics = result["metrics"]
        rows.append(
            {
                "name": name,
                "valid_miou": metrics["valid_miou"],
                "miou4": metrics["miou4"],
                "damaged_iou": metrics["damaged_iou"],
                "destroyed_iou": metrics["destroyed_iou"],
                "event_mean": metrics["event_mean_official_style_4class_miou"],
                "delta_valid_pp": 100.0 * (
                    metrics["valid_miou"] - base["valid_miou"]
                ),
            }
        )
    payload = {
        "protocol": "H8_BCSC_prototype_oracle_validation_v1",
        "validation_only": True,
        "test_accessed": False,
        "ground_truth_used_only_for_P2_prototype_estimation": True,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "protocol": protocol,
            "variant": variant,
            "step": int(checkpoint.get("step", 0)),
        },
        "blend_values": blend_values,
        "results": results,
        "summary": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"{'Mode':<30} {'Valid':>9} {'mIoU4':>9} {'Damaged':>9} "
        f"{'Destroyed':>10} {'Event':>9} {'dValid':>9}"
    )
    print("-" * 98)
    for row in rows:
        print(
            f"{row['name']:<30} {100*row['valid_miou']:9.4f} "
            f"{100*row['miou4']:9.4f} {100*row['damaged_iou']:9.4f} "
            f"{100*row['destroyed_iou']:10.4f} {100*row['event_mean']:9.4f} "
            f"{row['delta_valid_pp']:+9.4f}"
        )
    print(f"Saved: {output}", flush=True)
    print("Test accessed: False", flush=True)


if __name__ == "__main__":
    main()
