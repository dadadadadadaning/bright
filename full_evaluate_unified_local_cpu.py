#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows-local CPU-only entry point for the unified BRIGHT evaluator.

This wrapper intentionally leaves ``full_evaluate_unified.py`` unchanged.  It
forces CPU execution and disables torchvision pretrained-weight downloads
before delegating all dataset, inference, official metric, and event-wise work
to the audited unified evaluator.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "bright_mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torch

# The historical dataset loader only needs imageio.imread.  Some lightweight
# CPU environments omit imageio while already providing Pillow, so supply a
# local read-only compatibility shim instead of requiring a network install.
try:
    import imageio  # noqa: F401
except ModuleNotFoundError:
    import numpy as np
    from PIL import Image

    imageio_shim = types.ModuleType("imageio")

    def _imread(path):
        with Image.open(path) as image:
            return np.asarray(image)

    imageio_shim.imread = _imread
    sys.modules["imageio"] = imageio_shim

import full_evaluate_unified as evaluator


def import_model_class_cpu(model_type: str):
    """Import a supported model without requesting pretrained Swin weights."""
    model_cls = evaluator._ORIGINAL_IMPORT_MODEL_CLASS(model_type)
    module = sys.modules.get(model_cls.__module__)
    if module is not None and hasattr(module, "Swin_T_Weights"):
        # DamageFormer historically calls swin_t(weights=Swin_T_Weights).
        # Replacing the module symbol makes that call equivalent to weights=None.
        module.Swin_T_Weights = None
    return model_cls


evaluator._ORIGINAL_IMPORT_MODEL_CLASS = evaluator.import_model_class
evaluator.import_model_class = import_model_class_cpu


def main() -> None:
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        raise RuntimeError("Local CPU evaluator unexpectedly detected CUDA.")

    parser = evaluator.build_arg_parser()
    args = parser.parse_args()
    if args.num_workers != 0:
        print("WARNING: forcing --num_workers 0 for Windows CPU evaluation.")
        args.num_workers = 0
    evaluator.evaluate(args)


if __name__ == "__main__":
    main()
