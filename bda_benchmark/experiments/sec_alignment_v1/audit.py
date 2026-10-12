# -*- coding: utf-8 -*-
"""Clean/shift accuracy and known-displacement response audit for SEC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.experiments.sec_alignment_v1.model import build_sec_model
from bda_benchmark.experiments.sec_alignment_v1.objectives import (
    STAGE_STRIDES,
    reference_validity_for_shift,
    shift_without_wrap,
)
from bda_benchmark.model.DamageFormer_SALT_CDE import build_salt_damageformer
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import (
    evaluate,
)


SHIFTS = {
    "none": (0, 0),
    "shift_x4": (4, 0),
    "shift_y4": (0, 4),
    "shift_x8": (8, 0),
    "shift_y8": (0, 8),
    "shift_x16": (16, 0),
    "shift_y16": (0, 16),
}


class ShiftedSARModel(nn.Module):
    def __init__(self, model: nn.Module, dx: int, dy: int) -> None:
        super().__init__()
        self.model, self.dx, self.dy = model, int(dx), int(dy)

    def forward(self, rgb, sar, return_dict=False):
        shifted, _ = shift_without_wrap(sar, self.dx, self.dy)
        return self.model(rgb, shifted, return_dict=return_dict)


@torch.inference_mode()
def offset_delta_response(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dx: int,
    dy: int,
    max_batches: int,
) -> Dict[str, object]:
    if dx == 0 and dy == 0:
        return {}
    totals: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        rgb = batch[0].float().to(device, non_blocking=True)
        sar = batch[1].float().to(device, non_blocking=True)
        location = batch[2].long().to(device, non_blocking=True)
        shifted_sar, _ = shift_without_wrap(sar, dx, dy)
        validity = reference_validity_for_shift(sar, dx, dy)
        clean, shifted = model.forward_counterfactual(rgb, sar, shifted_sar)
        building = (location == 1).float().unsqueeze(1) * validity
        for stage, stride in STAGE_STRIDES.items():
            x_key, y_key = f"stage{stage}_offset_x", f"stage{stage}_offset_y"
            if x_key not in clean or x_key not in shifted:
                continue
            delta_x = shifted[x_key].float() - clean[x_key].float()
            delta_y = shifted[y_key].float() - clean[y_key].float()
            support = F.interpolate(
                building, size=delta_x.shape[-2:], mode="area"
            ).clamp(0.0, 1.0)
            count = float(support.sum())
            if count <= 0:
                continue
            expected_x, expected_y = dx / float(stride), dy / float(stride)
            values = {
                "observed_delta_x": delta_x,
                "observed_delta_y": delta_y,
                "absolute_error_x": (delta_x - expected_x).abs(),
                "absolute_error_y": (delta_y - expected_y).abs(),
            }
            for label, value in values.items():
                key = f"stage{stage}.{label}"
                totals[key] = totals.get(key, 0.0) + float((value * support).sum())
                weights[key] = weights.get(key, 0.0) + count
    result: Dict[str, object] = {}
    for stage, stride in STAGE_STRIDES.items():
        prefix = f"stage{stage}."
        available = {k[len(prefix):]: totals[k] / weights[k] for k in totals if k.startswith(prefix)}
        if available:
            result[f"stage{stage}"] = {
                "stride": stride,
                "expected_delta_x": dx / float(stride),
                "expected_delta_y": dy / float(stride),
                **available,
                "mean_absolute_error": 0.5 * (
                    available["absolute_error_x"] + available["absolute_error_y"]
                ),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser("SEC validation-only audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--perturbation", choices=tuple(SHIFTS), default="none")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(checkpoint.get("variant", "")).lower()
    if variant not in (
        "d1", "s1", "s2", "s3", "s2r", "s2r0", "s2rp", "s2rr", "s2rcf"
    ):
        raise RuntimeError(f"not an SEC checkpoint: variant={variant!r}")
    device = torch.device(args.device)
    model = (
        build_salt_damageformer("d1", pretrained_rgb=False)
        if variant == "d1"
        else build_sec_model(variant, pretrained_rgb=False)
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path, args.data_list_path, 640, split="val", variant="h1",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    dx, dy = SHIFTS[args.perturbation]
    metrics = evaluate(
        ShiftedSARModel(model, dx, dy), loader, device, 255, args.max_batches
    )
    response = offset_delta_response(
        model, loader, device, dx, dy, args.max_batches
    )
    payload = {
        "protocol": "SEC_AUDIT_v1",
        "variant": variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "perturbation": args.perturbation,
        "synthetic_shift_pixels": {"dx": dx, "dy": dy},
        "metrics": metrics,
        "offset_delta_response": response,
        "processed_images": int(metrics.get("processed_images", 0)),
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
