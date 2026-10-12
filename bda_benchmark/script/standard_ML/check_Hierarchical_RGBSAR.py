#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment and forward-interface check for the unified S5-S8 code."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", action="store_true", help="Run 64x64 CPU forward checks")
    args = parser.parse_args()

    print("Python:", sys.executable)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Resolved BRIGHT root:", BDA_ROOT)

    required = [
        BDA_ROOT / "model" / "DamageFormer_Hierarchical_RGBSAR.py",
        BDA_ROOT / "script" / "standard_ML" / "train_Hierarchical_RGBSAR.py",
        BDA_ROOT / "script" / "standard_ML" / "eval_Hierarchical_RGBSAR.py",
    ]
    failed = False
    for path in required:
        exists = path.is_file()
        print(("PASS" if exists else "MISSING"), path)
        failed |= not exists
    if failed:
        raise SystemExit(2)

    for module_name in ("torchvision", "numpy", "imageio", "cv2"):
        try:
            module = importlib.import_module(module_name)
            print("IMPORT PASS", module_name, getattr(module, "__version__", ""))
        except Exception as exc:
            print("IMPORT FAIL", module_name, repr(exc))
            failed = True

    try:
        from dataset.make_data_loader import MultimodalDamageAssessmentDatset  # noqa: F401
        print("IMPORT PASS dataset.make_data_loader")
    except Exception as exc:
        print("IMPORT FAIL dataset.make_data_loader", repr(exc))
        failed = True

    from model.DamageFormer_Hierarchical_RGBSAR import DamageFormer_Hierarchical_RGBSAR
    print("MODEL IMPORT PASS")

    if args.forward:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for variant in ("S5", "S6", "S7", "S8"):
                model = DamageFormer_Hierarchical_RGBSAR(
                    variant=variant,
                    use_pretrained_swin=False,
                    return_dict=True,
                ).eval()
                x = torch.randn(1, 3, 64, 64)
                with torch.no_grad():
                    out = model(x, x)
                assert out["output_loc"].shape == (1, 2, 64, 64)
                assert out["output_dam"].shape == (1, 4, 64, 64)
                if variant == "S8":
                    assert "ia_rgb_s3" in out and "ia_rgb_s4" in out
                print("FORWARD PASS", variant, tuple(out["output_dam"].shape))
                del model, x, out
        finally:
            torch.set_num_threads(old_threads)

    if failed:
        raise SystemExit(3)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
