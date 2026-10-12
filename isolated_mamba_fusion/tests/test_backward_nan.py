"""Backward and NaN checks for isolated fusion modules.

Run:
    python isolated_mamba_fusion/tests/test_backward_nan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isolated_mamba_fusion.modules import DiffMambaFusion, ExistGuidedMambaFusion  # noqa: E402


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_module(name: str, module: torch.nn.Module, f_rgb: torch.Tensor, f_sar: torch.Tensor, g_exist: torch.Tensor, roi_mask: torch.Tensor) -> None:
    module.train()
    f_rgb.grad = None
    f_sar.grad = None
    module.zero_grad(set_to_none=True)

    out = module(f_rgb, f_sar, g_exist, roi_mask=roi_mask, cur_iter=100)
    assert not torch.isnan(out).any(), f"{name}: output has NaN"
    loss = out.mean()
    loss.backward()

    assert f_rgb.grad is not None, f"{name}: F_rgb.grad is None"
    assert f_sar.grad is not None, f"{name}: F_sar.grad is None"
    assert not torch.isnan(f_rgb.grad).any(), f"{name}: F_rgb.grad has NaN"
    assert not torch.isnan(f_sar.grad).any(), f"{name}: F_sar.grad has NaN"

    for param_name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"{name}: parameter {param_name} grad is None"
        assert not torch.isnan(param.grad).any(), f"{name}: parameter {param_name} grad has NaN"

    if hasattr(module, "scale"):
        scale = module.scale().detach()
        scale_msg = f"scale shape={tuple(scale.shape)}, min={scale.min().item():.6f}, max={scale.max().item():.6f}"
    else:
        scale_msg = "scale=N/A"

    print(f"{name} backward passed")
    print(f"  output shape: {tuple(out.shape)}")
    print(f"  parameter count: {count_parameters(module):,}")
    print(f"  gamma {scale_msg}")


def run_device(device: torch.device) -> None:
    torch.manual_seed(2026)
    b, c, h, w = 2, 64, 20, 20
    f_rgb = torch.randn(b, c, h, w, device=device, requires_grad=True)
    f_sar = torch.randn(b, c, h, w, device=device, requires_grad=True)
    g_exist = torch.randn(b, 2, h, w, device=device)
    roi_mask = torch.rand(b, 1, h, w, device=device)

    diff = DiffMambaFusion(channels=c, backend="fallback", gamma_init=0.05).to(device)
    exist = ExistGuidedMambaFusion(channels=c, backend="fallback", gamma_init=0.05, detach_gate=True).to(device)

    check_module("DiffMambaFusion", diff, f_rgb, f_sar, g_exist, roi_mask)
    check_module("ExistGuidedMambaFusion", exist, f_rgb, f_sar, g_exist, roi_mask)
    print(f"test_backward_nan.py passed on {device}")


def main() -> None:
    run_device(torch.device("cpu"))
    if torch.cuda.is_available():
        run_device(torch.device("cuda"))
    else:
        print("CUDA not available; GPU backward test skipped.")


if __name__ == "__main__":
    main()

