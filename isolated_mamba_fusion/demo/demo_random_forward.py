"""Random forward demo for isolated mamba fusion modules.

Run:
    python isolated_mamba_fusion/demo/demo_random_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isolated_mamba_fusion.modules import BaseFusion, DiffMambaFusion, ExistGuidedMambaFusion  # noqa: E402


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def scale_summary(module: torch.nn.Module) -> str:
    if not hasattr(module, "scale"):
        return "N/A"
    scale = module.scale().detach()
    return f"shape={tuple(scale.shape)}, min={scale.min().item():.6f}, max={scale.max().item():.6f}"


def main() -> None:
    torch.manual_seed(2026)
    b, c, h, w = 2, 256, 40, 40
    f_rgb = torch.randn(b, c, h, w)
    f_sar = torch.randn(b, c, h, w)
    g_exist = torch.randn(b, 2, h // 2, w // 2)
    roi_mask = torch.rand(b, 1, h // 2, w // 2)

    base_module = BaseFusion(channels=c, base_mode="conv")
    diff_module = DiffMambaFusion(channels=c, gamma_init=0.05, gamma_scale=0.1, backend="auto")
    exist_module = ExistGuidedMambaFusion(
        channels=c,
        gamma_init=0.05,
        gamma_scale=0.1,
        backend="auto",
        base_mode="conv",
        gate_min=0.1,
        bg_alpha=0.05,
        detach_gate=False,
        gate_warmup=1000,
    )

    with torch.no_grad():
        y_base = base_module(f_rgb, f_sar)
        y_diff = diff_module(f_rgb, f_sar, roi_mask=roi_mask)
        y_exist_warmup = exist_module(f_rgb, f_sar, g_exist, roi_mask=roi_mask, cur_iter=0)
        y_exist = exist_module(f_rgb, f_sar, g_exist, roi_mask=roi_mask, cur_iter=2000)

    print("Input:")
    print(f"  F_rgb:   {tuple(f_rgb.shape)}")
    print(f"  F_sar:   {tuple(f_sar.shape)}")
    print(f"  G_exist: {tuple(g_exist.shape)}")
    print(f"  roi:     {tuple(roi_mask.shape)}")
    print("Output:")
    print(f"  BaseFusion:             {tuple(y_base.shape)}")
    print(f"  DiffMambaFusion:        {tuple(y_diff.shape)}")
    print(f"  ExistGuided warmup:     {tuple(y_exist_warmup.shape)}")
    print(f"  ExistGuided after warm: {tuple(y_exist.shape)}")
    print("Parameters:")
    print(f"  BaseFusion:             {count_parameters(base_module):,}")
    print(f"  DiffMambaFusion:        {count_parameters(diff_module):,}")
    print(f"  ExistGuidedMambaFusion: {count_parameters(exist_module):,}")
    print("Gamma / bounded scale:")
    print(f"  Diff raw gamma shape:   {tuple(diff_module.gamma.shape)}")
    print(f"  Diff effective scale:   {scale_summary(diff_module)}")
    print(f"  Exist raw gamma shape:  {tuple(exist_module.gamma.shape)}")
    print(f"  Exist effective scale:  {scale_summary(exist_module)}")
    print("Backend:")
    print(f"  DiffMambaFusion block:        {diff_module.mamba_block.active_backend}")
    print(f"  ExistGuidedMambaFusion block: {exist_module.diff_fusion.mamba_block.active_backend}")


if __name__ == "__main__":
    main()

