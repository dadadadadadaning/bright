#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug check for BGCA_none path identity behavior."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
if BDA_ROOT not in sys.path:
    sys.path.insert(0, BDA_ROOT)

try:
    from model.DamageFormer_BGCA import DamageFormer_BGCA
except Exception:
    from bda_benchmark.model.DamageFormer_BGCA import DamageFormer_BGCA


def clone_bn_stats(model: DamageFormer_BGCA) -> Dict[str, torch.Tensor]:
    return {
        "rgb_mean": model.rgb_proj[1].running_mean.detach().clone(),
        "rgb_var": model.rgb_proj[1].running_var.detach().clone(),
        "sar_mean": model.sar_proj[1].running_mean.detach().clone(),
        "sar_var": model.sar_proj[1].running_var.detach().clone(),
    }


def max_mean_abs_diff(before: torch.Tensor, after: torch.Tensor) -> Tuple[float, float]:
    diff = (after.detach() - before.detach()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether DamageFormer_BGCA with roi_mode=none and use_align=False changes the main path."
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use_pretrained_swin",
        action="store_true",
        help="Enable torchvision pretrained Swin weights. Off by default to avoid downloads during debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = DamageFormer_BGCA(
        roi_mode="none",
        use_align=False,
        use_shared_diff=False,
        use_modality_prompt=False,
        use_reliability_gate=False,
        return_dict=True,
        use_pretrained_swin=args.use_pretrained_swin,
    ).to(device)
    model.train()

    pre = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    post = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    bn_before = clone_bn_stats(model)
    with torch.no_grad():
        outputs = model(pre, post)
    bn_after = clone_bn_stats(model)

    required_keys = ["roi_weight", "final_feat_fused", "final_feat_fused_roi"]
    missing_keys = [key for key in required_keys if key not in outputs]
    if missing_keys:
        raise KeyError(f"Missing required output keys: {missing_keys}. Available keys: {list(outputs.keys())}")

    roi_weight = outputs["roi_weight"].detach()
    final_feat_fused = outputs["final_feat_fused"].detach()
    final_feat_fused_roi = outputs["final_feat_fused_roi"].detach()
    feat_diff = (final_feat_fused - final_feat_fused_roi).abs()

    print("BGCA_none main-path debug check")
    print(f"device: {device}")
    print(f"model.training: {model.training}")
    print(f"input_shape: pre={tuple(pre.shape)}, post={tuple(post.shape)}")
    print("")
    print("ROI weight:")
    print(f"roi_weight.min():  {roi_weight.min().item():.12g}")
    print(f"roi_weight.max():  {roi_weight.max().item():.12g}")
    print(f"roi_weight.mean(): {roi_weight.mean().item():.12g}")
    print("")
    print("Fused feature identity:")
    print(f"max_abs_diff:  {feat_diff.max().item():.12g}")
    print(f"mean_abs_diff: {feat_diff.mean().item():.12g}")
    print("")
    print("Projection BN running-stat changes:")
    bn_changes = {}
    for name in ("rgb_mean", "rgb_var", "sar_mean", "sar_var"):
        max_diff, mean_diff = max_mean_abs_diff(bn_before[name], bn_after[name])
        bn_changes[name] = max_diff
        print(f"{name}: max_abs_change={max_diff:.12g}, mean_abs_change={mean_diff:.12g}")

    has_warning = False
    ones = torch.ones_like(roi_weight)
    if not torch.allclose(roi_weight, ones, rtol=0.0, atol=0.0):
        print("WARNING: roi_weight is not exactly all 1.")
        has_warning = True
    if feat_diff.max().item() != 0.0:
        print("WARNING: final_feat_fused and final_feat_fused_roi are not exactly identical.")
        has_warning = True
    for name, max_diff in bn_changes.items():
        if max_diff != 0.0:
            print(f"WARNING: {name} changed even though use_align=False.")
            has_warning = True

    if not has_warning:
        print("OK: roi_mode=none leaves roi_weight/fused features unchanged and projection BN stats untouched.")


if __name__ == "__main__":
    main()
