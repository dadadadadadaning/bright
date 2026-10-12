"""Shape tests for isolated mamba fusion modules.

Run:
    python isolated_mamba_fusion/tests/test_shapes.py
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


def assert_no_nan(name: str, tensor: torch.Tensor) -> None:
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"


def scale_range(module: torch.nn.Module) -> str:
    scale = module.scale().detach()
    return f"shape={tuple(scale.shape)}, min={scale.min().item():.6f}, max={scale.max().item():.6f}"


def main() -> None:
    torch.manual_seed(3407)
    b, c, h, w = 2, 256, 40, 40
    f_rgb = torch.randn(b, c, h, w)
    f_sar = torch.randn(b, c, h, w)
    g_exist_1 = torch.randn(b, 1, h, w)
    g_exist_2 = torch.randn(b, 2, h // 2, w // 2)
    roi = torch.rand(b, 1, h, w)
    roi_small = torch.rand(b, 1, h // 2, w // 2)

    base_module = BaseFusion(channels=c, base_mode="conv")
    y_base = base_module(f_rgb, f_sar)
    assert y_base.shape == f_rgb.shape, f"BaseFusion shape mismatch: {y_base.shape}"
    assert_no_nan("BaseFusion", y_base)

    diff_module = DiffMambaFusion(channels=c, backend="fallback", gamma_init=0.05)
    for idx, mask in enumerate([None, roi, roi_small]):
        y_diff = diff_module(f_rgb, f_sar, roi_mask=mask)
        assert y_diff.shape == f_rgb.shape, f"DiffMambaFusion shape mismatch at roi case {idx}: {y_diff.shape}"
        assert_no_nan(f"DiffMambaFusion roi case {idx}", y_diff)

    exist_module = ExistGuidedMambaFusion(channels=c, backend="fallback", gamma_init=0.05, detach_gate=True)
    exist_cases = [
        ("none", None, None),
        ("one_channel", g_exist_1, None),
        ("two_channel_resized", g_exist_2, None),
        ("roi", None, roi),
        ("roi_resized", g_exist_2, roi_small),
        ("warmup", g_exist_2, roi),
    ]
    for name, gate, mask in exist_cases:
        cur_iter = 0 if name == "warmup" else None
        exist_module.gate_warmup = 10 if name == "warmup" else 0
        y_exist = exist_module(f_rgb, f_sar, gate, roi_mask=mask, cur_iter=cur_iter)
        assert y_exist.shape == f_rgb.shape, f"ExistGuidedMambaFusion {name} shape mismatch: {y_exist.shape}"
        assert_no_nan(f"ExistGuidedMambaFusion {name}", y_exist)

    for mode, expected in [
        ("rgb", f_rgb),
        ("sar", f_sar),
        ("avg", 0.5 * (f_rgb + f_sar)),
    ]:
        disabled = ExistGuidedMambaFusion(channels=c, backend="fallback", base_mode=mode, enable=False)
        y_disabled = disabled(f_rgb, f_sar, g_exist_2, roi_mask=roi)
        assert torch.allclose(y_disabled, expected), f"enable=False base_mode={mode} is not identity/no-op"

    disabled_conv = ExistGuidedMambaFusion(channels=c, backend="fallback", base_mode="conv", enable=False)
    y_disabled_conv = disabled_conv(f_rgb, f_sar, g_exist_2, roi_mask=roi)
    assert y_disabled_conv.shape == f_rgb.shape
    assert_no_nan("enable=False base_mode=conv", y_disabled_conv)

    for mode, expected in [
        ("rgb", f_rgb),
        ("sar", f_sar),
        ("avg", 0.5 * (f_rgb + f_sar)),
    ]:
        disabled_diff = DiffMambaFusion(channels=c, backend="fallback", base_mode=mode, enable=False)
        y_disabled_diff = disabled_diff(f_rgb, f_sar, g_exist_2, roi_mask=roi)
        assert torch.allclose(y_disabled_diff, expected), f"Diff enable=False base_mode={mode} is not identity/no-op"

    print("test_shapes.py passed")
    print(f"Base output:  {tuple(y_base.shape)}")
    print(f"Diff output:  {tuple(y_diff.shape)}")
    print(f"Exist output: {tuple(y_exist.shape)}")
    print(f"Diff raw gamma shape:  {tuple(diff_module.gamma.shape)}")
    print(f"Diff scale range:      {scale_range(diff_module)}")
    print(f"Exist raw gamma shape: {tuple(exist_module.gamma.shape)}")
    print(f"Exist scale range:     {scale_range(exist_module)}")


if __name__ == "__main__":
    main()

