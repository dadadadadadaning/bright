"""CPU forward/backward smoke test.

Run:
    python isolated_mamba_fusion/tests/test_forward_cpu.py
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


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    b, c, h, w = 1, 64, 16, 16
    f_rgb = torch.randn(b, c, h, w, device=device, requires_grad=True)
    f_sar = torch.randn(b, c, h, w, device=device, requires_grad=True)
    g_exist = torch.randn(b, 2, h, w, device=device)
    roi_mask = torch.rand(b, 1, h, w, device=device)

    diff_module = DiffMambaFusion(channels=c, backend="fallback").to(device)
    exist_module = ExistGuidedMambaFusion(channels=c, backend="fallback", detach_gate=True).to(device)

    y1 = diff_module(f_rgb, f_sar, roi_mask=roi_mask)
    y2 = exist_module(f_rgb, f_sar, g_exist, roi_mask=roi_mask)
    loss = y1.mean() + y2.mean()
    loss.backward()

    assert f_rgb.grad is not None, "No gradient for F_rgb"
    assert f_sar.grad is not None, "No gradient for F_sar"
    assert y1.shape == f_rgb.shape
    assert y2.shape == f_rgb.shape

    print("test_forward_cpu.py passed")
    print(f"loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

