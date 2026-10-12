#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated 4k-step runner for S8-P6 A4 + Soft-Critical MICB.

The mature P4 trainer owns the dataset, losses, checkpointing and validation
protocol.  This thin entry point replaces only the student model factory while
leaving the accepted a0 P3 teacher untouched.  Keeping the adapter here avoids
adding an ambiguous variant to the P4 registry.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.DamageFormer_S8_P4_NativeSAR_MICB import (  # noqa: E402
    DamageFormer_S8_P4_NativeSAR_MICB,
    P4_LATE_TRAINABLE_PREFIXES,
    SAR_INPUT_PREFIX,
    lock_frozen_batch_norm,
)
from model.DamageFormer_S8_P6_SoftCritical import (  # noqa: E402
    DamageFormer_S8_P6_SoftCritical,
)
from script.standard_ML import train_S8_P4_native_sar_micb as runtime  # noqa: E402


def _argument_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value after {name}")
    return sys.argv[index + 1]


class _P6Factory:
    """Dispatch the frozen a0 teacher and the isolated a4 P6 student."""

    def __new__(cls, variant: str = "a4", **kwargs: Any):
        key = str(variant).lower()
        if key == "a0":
            return DamageFormer_S8_P4_NativeSAR_MICB(variant="a0", **kwargs)
        if key != "a4":
            raise ValueError("S8-P6 training accepts only student a4 and teacher a0")
        temperature = float(os.environ.get("OCCURRENCE_TEMPERATURE", "0.5"))
        return DamageFormer_S8_P6_SoftCritical(
            variant="a4",
            occurrence_temperature=temperature,
            **kwargs,
        )


def _module_for_prefix(model: nn.Module, prefix: str) -> nn.Module:
    module = model
    for name in prefix.rstrip(".").split("."):
        module = module[int(name)] if name.isdigit() else getattr(module, name)
    return module


def configure_p6_trainable(model: nn.Module, input_only: bool):
    """Two non-overlapping phases: adapt A4, then lock it and adapt the core."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = (SAR_INPUT_PREFIX,) if input_only else P4_LATE_TRAINABLE_PREFIXES
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
    lock_frozen_batch_norm(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def set_p6_training_mode(model: nn.Module, input_only: bool) -> None:
    model.eval()
    prefixes = (SAR_INPUT_PREFIX,) if input_only else P4_LATE_TRAINABLE_PREFIXES
    for prefix in prefixes:
        _module_for_prefix(model, prefix).train()
    lock_frozen_batch_norm(model)


def main() -> None:
    variant = (_argument_value("--variant") or "").lower()
    if variant != "a4":
        raise ValueError("S8-P6 is locked to --variant a4")
    output_value = _argument_value("--output_dir")
    if not output_value:
        raise ValueError("S8-P6 requires --output_dir")
    output_dir = Path(output_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "family": "S8-P6 NativeSAR SoftCritical MICB",
        "base": "accepted P3",
        "student_variant": "a4",
        "occurrence_temperature": float(
            os.environ.get("OCCURRENCE_TEMPERATURE", "0.5")
        ),
        "shared_cross_attention": False,
        "pixel_occurrence_head": False,
        "mil_residual_scale": 0.0,
        "phase_1_trainable": ["a4_sar_patch_embed"],
        "phase_2_trainable": ["fusion_layer_3", "clf_2"],
        "a4_input_locked_after_step": 1000,
        "test_access": "forbidden during training",
    }
    with open(output_dir / "experiment_contract.json", "w", encoding="utf-8") as file:
        json.dump(contract, file, ensure_ascii=False, indent=2)

    runtime.DamageFormer_S8_P4_NativeSAR_MICB = _P6Factory
    runtime.configure_p4_trainable = configure_p6_trainable
    runtime.set_p4_training_mode = set_p6_training_mode
    runtime.main()

    summary_path = output_dir / "training_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)
        summary["experiment_contract"] = contract
        temporary = summary_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        os.replace(temporary, summary_path)


if __name__ == "__main__":
    main()
