#!/usr/bin/env python3
"""Clean and controlled-misalignment validation audit for D0/A2--A5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.model.DamageFormer_RGBSAR_Heterogeneous import OriginalDamageFormerControl
from bda_benchmark.model.DamageFormer_ROI_LCA import DamageFormerROILCA, LCA_VARIANTS
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_Heterogeneous import evaluate
from bda_benchmark.experiments.damageformer_roi_lca_v1.train import checkpoint_state, shift_zero


def perturb(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return value
    if mode == "zero":
        return torch.zeros_like(value)
    if mode == "pair_shuffle":
        return value.roll(1, 0) if value.shape[0] > 1 else shift_zero(value, 8, 0)
    if mode.startswith("shift_x"):
        return shift_zero(value, int(mode[7:]), 0)
    if mode.startswith("shift_y"):
        return shift_zero(value, 0, int(mode[7:]))
    raise ValueError(mode)


class Perturbed(nn.Module):
    def __init__(self, model: nn.Module, mode: str) -> None:
        super().__init__(); self.model = model; self.mode = mode

    def forward(self, rgb, sar, return_dict=False):
        return self.model(rgb, perturb(sar, self.mode), return_dict=return_dict)


@torch.inference_mode()
def diagnostics(model, loader, device, mode: str, max_batches: int) -> Dict[str, float]:
    if not isinstance(model, DamageFormerROILCA):
        return {}
    totals: Dict[str, float] = {}; count = 0
    model.eval()
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        rgb, sar = batch[0].float().to(device), batch[1].float().to(device)
        output = model(rgb, perturb(sar, mode), return_dict=True)
        roi = output["alignment_roi"].float()
        denominator = roi.sum().clamp_min(1.0)
        totals["mean_confidence_on_roi"] = totals.get("mean_confidence_on_roi", 0.0) + float(
            (output["alignment_confidence"].float() * roi).sum() / denominator
        )
        totals["mean_entropy_on_roi"] = totals.get("mean_entropy_on_roi", 0.0) + float(
            (output["alignment_entropy"].float() * roi).sum() / denominator
        )
        for axis, channel in (("x", 0), ("y", 1)):
            totals[f"mean_offset_{axis}_on_roi"] = totals.get(f"mean_offset_{axis}_on_roi", 0.0) + float(
                (output["alignment_offset"][:, channel:channel+1].float() * roi).sum() / denominator
            )
        count += 1
    values = {key: value / max(count, 1) for key, value in totals.items()}
    values["alignment_gain"] = float(torch.tanh(model.roi_lca.gain.detach()))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("d0", *LCA_VARIANTS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--perturbation", choices=(
        "none", "shift_x4", "shift_y4", "shift_x8", "shift_y8",
        "shift_x16", "shift_y16", "pair_shuffle", "zero"), default="none")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("ROI-LCA audit forbids Test")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.variant == "d0":
        model = OriginalDamageFormerControl(pretrained=False)
        model.load_state_dict(checkpoint_state(payload), strict=True)
    else:
        model = DamageFormerROILCA(args.variant, pretrained=False)
        model.load_state_dict(checkpoint_state(payload), strict=True)
    device = torch.device(args.device)
    model.to(device)
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path, args.val_list, 640, split="val", variant="h0"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    metrics = evaluate(Perturbed(model, args.perturbation), loader, device, 255, args.max_batches)
    diag = diagnostics(model, loader, device, args.perturbation, args.max_batches)
    offset_recovery = {}
    if args.perturbation.startswith(("shift_x", "shift_y")) and diag:
        axis = "x" if args.perturbation.startswith("shift_x") else "y"
        pixels = int(args.perturbation.split(axis)[-1])
        observed = diag[f"mean_offset_{axis}_on_roi"]
        expected = pixels / 4.0
        offset_recovery = {"axis": axis, "expected_feature_offset": expected,
                           "observed_feature_offset": observed,
                           "absolute_error": abs(observed - expected)}
    result = {
        "protocol": "DAMAGEFORMER_ROI_LCA_AUDIT_V1", "variant": args.variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(payload.get("step", 0)) if isinstance(payload, Mapping) else 0,
        "perturbation": args.perturbation, "metrics": metrics,
        "diagnostics": diag, "offset_recovery": offset_recovery,
        "processed_images": metrics.get("processed_images", len(dataset)), "test_accessed": False,
    }
    destination = Path(args.output_json); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
