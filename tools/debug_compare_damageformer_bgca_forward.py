#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare forward outputs of DamageFormer and BGCA_none."""

from __future__ import annotations

import argparse
import importlib
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


def import_damageformer_module():
    try:
        return importlib.import_module("model.DamageFormer")
    except Exception:
        return importlib.import_module("bda_benchmark.model.DamageFormer")


def import_bgca_class():
    try:
        from model.DamageFormer_BGCA import DamageFormer_BGCA
    except Exception:
        from bda_benchmark.model.DamageFormer_BGCA import DamageFormer_BGCA
    return DamageFormer_BGCA


def copy_matching_state(src: torch.nn.Module, dst: torch.nn.Module) -> Tuple[int, int]:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = 0
    skipped = 0
    merged: Dict[str, torch.Tensor] = {}

    for key, value in dst_state.items():
        if key in src_state and src_state[key].shape == value.shape:
            merged[key] = src_state[key].detach().clone()
            copied += 1
        else:
            merged[key] = value
            skipped += 1

    dst.load_state_dict(merged, strict=False)
    return copied, skipped


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a.detach() - b.detach()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether DamageFormer_BGCA with all new modules disabled matches DamageFormer forward."
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use_pretrained_swin",
        action="store_true",
        help="Use torchvision pretrained Swin weights. Off by default to avoid downloads.",
    )
    parser.add_argument(
        "--train_mode",
        action="store_true",
        help="Run models in train mode. Default is eval mode for deterministic BatchNorm behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    damageformer_mod = import_damageformer_module()
    DamageFormer = damageformer_mod.DamageFormer
    DamageFormer_BGCA = import_bgca_class()

    if args.use_pretrained_swin:
        damageformer_mod.Swin_T_Weights = damageformer_mod.Swin_T_Weights.DEFAULT
    else:
        # The original DamageFormer hard-codes swin_t(weights=Swin_T_Weights).
        # Setting its module global to None makes construction download-free.
        damageformer_mod.Swin_T_Weights = None

    baseline = DamageFormer().to(device)
    bgca = DamageFormer_BGCA(
        roi_mode="none",
        use_align=False,
        use_shared_diff=False,
        use_modality_prompt=False,
        use_reliability_gate=False,
        return_dict=True,
        use_pretrained_swin=args.use_pretrained_swin,
    ).to(device)

    copied, skipped = copy_matching_state(baseline, bgca)
    if args.train_mode:
        baseline.train()
        bgca.train()
    else:
        baseline.eval()
        bgca.eval()

    pre = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    post = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    with torch.no_grad():
        base_loc, base_dam = baseline(pre, post)
        bgca_outputs = bgca(pre, post)

    bgca_loc = bgca_outputs["output_loc"]
    bgca_dam = bgca_outputs["output_dam"]
    roi_weight = bgca_outputs["roi_weight"]
    final_feat_fused = bgca_outputs["final_feat_fused"]
    final_feat_fused_roi = bgca_outputs["final_feat_fused_roi"]

    loc_max, loc_mean = diff_stats(base_loc, bgca_loc)
    dam_max, dam_mean = diff_stats(base_dam, bgca_dam)
    fused_max, fused_mean = diff_stats(final_feat_fused, final_feat_fused_roi)

    print("DamageFormer vs DamageFormer_BGCA_none forward comparison")
    print(f"device: {device}")
    print(f"mode: {'train' if args.train_mode else 'eval'}")
    print(f"input_shape: pre={tuple(pre.shape)}, post={tuple(post.shape)}")
    print(f"copied_same_shape_state_keys: {copied}")
    print(f"skipped_bgca_state_keys: {skipped}")
    print("")
    print("Output differences:")
    print(f"output_loc max_abs_diff:  {loc_max:.12g}")
    print(f"output_loc mean_abs_diff: {loc_mean:.12g}")
    print(f"output_dam max_abs_diff:  {dam_max:.12g}")
    print(f"output_dam mean_abs_diff: {dam_mean:.12g}")
    print("")
    print("BGCA ROI bypass:")
    print(f"roi_weight min:  {roi_weight.min().item():.12g}")
    print(f"roi_weight max:  {roi_weight.max().item():.12g}")
    print(f"roi_weight mean: {roi_weight.mean().item():.12g}")
    print(f"final_feat_fused vs final_feat_fused_roi max_abs_diff:  {fused_max:.12g}")
    print(f"final_feat_fused vs final_feat_fused_roi mean_abs_diff: {fused_mean:.12g}")

    if not torch.allclose(roi_weight, torch.ones_like(roi_weight), rtol=0.0, atol=0.0):
        print("WARNING: roi_weight is not exactly all 1.")
    if fused_max != 0.0:
        print("WARNING: final_feat_fused_roi is not exactly the same tensor value as final_feat_fused.")
    if loc_max > 1e-6 or dam_max > 1e-6:
        print("WARNING: BGCA_none forward is not numerically identical to DamageFormer.")


if __name__ == "__main__":
    main()
