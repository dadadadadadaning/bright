"""Comprehensive forward/backward tests for isolated fusion modules.

Run:
    python isolated_mamba_fusion/tests/test_comprehensive.py

This script intentionally does not depend on pytest.  It enumerates a broad
set of channel counts, spatial sizes, base modes, enable flags, G_exist shapes,
and ROI-mask shapes, then runs forward + backward checks for both
DiffMambaFusion and ExistGuidedMambaFusion.
"""

from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isolated_mamba_fusion.modules import DiffMambaFusion, ExistGuidedMambaFusion  # noqa: E402


CHANNELS = [96, 128, 192, 256, 384, 768]
SPATIAL_SIZES = [20, 40, 64]
BASE_MODES = ["rgb", "sar", "avg", "conv"]
ENABLE_FLAGS = [True, False]
G_EXIST_CASES = ["none", "1ch", "2ch", "1ch_half", "2ch_half"]
ROI_CASES = ["none", "full", "half"]


def make_g_exist(case: str, batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if case == "none":
        return None
    if case == "1ch":
        return torch.randn(batch, 1, height, width, device=device, dtype=dtype)
    if case == "2ch":
        return torch.randn(batch, 2, height, width, device=device, dtype=dtype)
    if case == "1ch_half":
        return torch.randn(batch, 1, max(1, height // 2), max(1, width // 2), device=device, dtype=dtype)
    if case == "2ch_half":
        return torch.randn(batch, 2, max(1, height // 2), max(1, width // 2), device=device, dtype=dtype)
    raise ValueError(f"Unknown G_exist case: {case}")


def make_roi(case: str, batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if case == "none":
        return None
    if case == "full":
        return torch.rand(batch, 1, height, width, device=device, dtype=dtype)
    if case == "half":
        return torch.rand(batch, 1, max(1, height // 2), max(1, width // 2), device=device, dtype=dtype)
    raise ValueError(f"Unknown ROI case: {case}")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def grad_status(module: torch.nn.Module) -> Tuple[int, int]:
    """Return (trainable_param_count, none_grad_param_count)."""

    trainable = 0
    none_grad = 0
    for param in module.parameters():
        if param.requires_grad:
            trainable += 1
            if param.grad is None:
                none_grad += 1
    return trainable, none_grad


def check_no_nan_grads(module: torch.nn.Module) -> None:
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            raise AssertionError(f"Parameter grad is None: {name}")
        if torch.isnan(param.grad).any():
            raise AssertionError(f"Parameter grad has NaN: {name}")


def run_one_case(
    *,
    module_name: str,
    channels: int,
    height: int,
    width: int,
    base_mode: str,
    enable: bool,
    g_case: str,
    roi_case: str,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    batch = 2
    torch.manual_seed(1000 + channels + height + len(base_mode))
    f_rgb = torch.randn(batch, channels, height, width, device=device, dtype=dtype, requires_grad=True)
    f_sar = torch.randn(batch, channels, height, width, device=device, dtype=dtype, requires_grad=True)
    g_exist = make_g_exist(g_case, batch, height, width, device, dtype)
    roi_mask = make_roi(roi_case, batch, height, width, device, dtype)

    if module_name == "DiffMambaFusion":
        module = DiffMambaFusion(
            channels=channels,
            backend="fallback",
            base_mode=base_mode,
            enable=enable,
            gamma_init=0.05,
            gamma_scale=0.1,
        ).to(device=device, dtype=dtype)
    elif module_name == "ExistGuidedMambaFusion":
        module = ExistGuidedMambaFusion(
            channels=channels,
            backend="fallback",
            base_mode=base_mode,
            enable=enable,
            gamma_init=0.05,
            gamma_scale=0.1,
            detach_gate=True,
            gate_warmup=5,
        ).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown module: {module_name}")

    module.train()
    out = module(f_rgb, f_sar, g_exist, roi_mask=roi_mask, cur_iter=10)
    if out.shape != f_rgb.shape:
        raise AssertionError(f"Output shape mismatch: expected {tuple(f_rgb.shape)}, got {tuple(out.shape)}")
    if torch.isnan(out).any():
        raise AssertionError("Output contains NaN")

    loss = out.mean()
    loss.backward()

    if f_rgb.grad is None:
        raise AssertionError("F_rgb.grad is None")
    if f_sar.grad is None:
        raise AssertionError("F_sar.grad is None")
    if torch.isnan(f_rgb.grad).any():
        raise AssertionError("F_rgb.grad contains NaN")
    if torch.isnan(f_sar.grad).any():
        raise AssertionError("F_sar.grad contains NaN")
    check_no_nan_grads(module)

    trainable, none_grad = grad_status(module)
    if hasattr(module, "scale"):
        scale = module.scale().detach()
        scale_msg = f"scale=[{scale.min().item():.4f},{scale.max().item():.4f}]"
    else:
        scale_msg = "scale=N/A"

    return (
        f"PASS {module_name} C={channels} HW={height}x{width} base={base_mode} "
        f"enable={enable} G={g_case} roi={roi_case} params={count_parameters(module)} "
        f"trainable_tensors={trainable} none_grad_tensors={none_grad} {scale_msg}"
    )


def main() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    modules = ["DiffMambaFusion", "ExistGuidedMambaFusion"]
    total = 0
    passed = 0
    failed = 0

    for module_name, channels, size, base_mode, enable, g_case, roi_case in itertools.product(
        modules,
        CHANNELS,
        SPATIAL_SIZES,
        BASE_MODES,
        ENABLE_FLAGS,
        G_EXIST_CASES,
        ROI_CASES,
    ):
        total += 1
        try:
            msg = run_one_case(
                module_name=module_name,
                channels=channels,
                height=size,
                width=size,
                base_mode=base_mode,
                enable=enable,
                g_case=g_case,
                roi_case=roi_case,
                device=device,
                dtype=dtype,
            )
            passed += 1
            print(msg)
        except Exception as exc:
            failed += 1
            print(
                "FAIL "
                f"module={module_name} C={channels} HW={size}x{size} base={base_mode} "
                f"enable={enable} G={g_case} roi={roi_case}"
            )
            print(f"  error: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    print("=" * 80)
    print(f"Comprehensive test summary: total={total}, passed={passed}, failed={failed}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


