# -*- coding: utf-8 -*-
"""Validation-only clean/perturbation audit for SALT-CDE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.model.DamageFormer_SALT_CDE import build_salt_damageformer
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import evaluate


def shift_zero(value: torch.Tensor, dx: int = 0, dy: int = 0) -> torch.Tensor:
    result = torch.zeros_like(value)
    height, width = value.shape[-2:]
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    result[..., target_y, target_x] = value[..., source_y, source_x]
    return result


def perturb_sar(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return value
    if mode == "zero":
        return torch.zeros_like(value)
    if mode == "pair_shuffle":
        return value.roll(1, 0) if value.shape[0] > 1 else shift_zero(value, 8, 0)
    for prefix, axis in (("shift_x", "x"), ("shift_y", "y")):
        if mode.startswith(prefix):
            amount = int(mode[len(prefix):])
            return shift_zero(value, dx=amount if axis == "x" else 0, dy=amount if axis == "y" else 0)
    raise ValueError(f"unknown perturbation {mode}")


class PerturbedModel(nn.Module):
    def __init__(self, model: nn.Module, mode: str) -> None:
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, rgb, sar, return_dict=False):
        return self.model(rgb, perturb_sar(sar, self.mode), return_dict=return_dict)


@torch.inference_mode()
def diagnostic_means(model, loader, device, mode: str, max_batches: int = 0) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        rgb, sar = batch[0].float().to(device), batch[1].float().to(device)
        output = model(rgb, perturb_sar(sar, mode), return_dict=True)
        for key, value in output.items():
            if any(token in key for token in ("entropy", "margin", "offset_x", "offset_y", "anchor_ring", "anchor_support", "sar_gate", "ring_strength")):
                totals[key] = totals.get(key, 0.0) + float(value.float().mean())
                counts[key] = counts.get(key, 0) + 1
        for stage in (2, 3):
            ring_key = f"stage{stage}_anchor_ring"
            for axis in ("x", "y"):
                offset_key = f"stage{stage}_offset_{axis}"
                if ring_key not in output or offset_key not in output:
                    continue
                ring = output[ring_key].float()
                weighted = float((output[offset_key].float() * ring).sum() / ring.sum().clamp_min(1.0))
                name = f"stage{stage}_offset_{axis}_on_anchor"
                totals[name] = totals.get(name, 0.0) + weighted
                counts[name] = counts.get(name, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals)}


def main() -> None:
    parser = argparse.ArgumentParser("SALT-CDE audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", default="")
    parser.add_argument("--perturbation", choices=("none", "shift_x4", "shift_y4", "shift_x8", "shift_y8", "shift_x16", "shift_y16", "pair_shuffle", "zero"), default="none")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(checkpoint.get("variant", checkpoint.get("args", {}).get("variant", ""))).lower()
    if variant not in ("d0", "d0n", "d1", "d2", "d3", "d2r", "d3v2"):
        raise RuntimeError(f"unsupported checkpoint variant {variant!r}")
    device = torch.device(args.device)
    model = build_salt_damageformer(variant, pretrained_rgb=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset_variant = "h0" if variant == "d0" else "h1"
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path, args.data_list_path, 640, split="val",
        variant=dataset_variant, sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    wrapped = PerturbedModel(model, args.perturbation)
    metrics = evaluate(wrapped, loader, device, 255, args.max_batches)
    diagnostics = (
        diagnostic_means(model, loader, device, args.perturbation, args.max_batches)
        if variant in ("d1", "d2", "d3", "d2r", "d3v2")
        else {}
    )
    offset_recovery = {}
    if args.perturbation.startswith(("shift_x", "shift_y")) and diagnostics:
        axis = "x" if args.perturbation.startswith("shift_x") else "y"
        pixels = int(args.perturbation.split(axis)[-1])
        for stage, stride in ((2, 8), (3, 16)):
            key = f"stage{stage}_offset_{axis}_on_anchor"
            if key in diagnostics:
                expected = pixels / float(stride)
                offset_recovery[f"stage{stage}"] = {
                    "axis": axis,
                    "expected_feature_offset": expected,
                    "observed_anchor_offset": diagnostics[key],
                    "absolute_error": abs(diagnostics[key] - expected),
                }
    payload = {
        "protocol": "SALT_CDE_AUDIT_v1",
        "variant": variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "perturbation": args.perturbation,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "offset_recovery": offset_recovery,
        "processed_images": metrics.get("processed_images", 0),
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
