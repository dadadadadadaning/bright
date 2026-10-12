#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated P6-C1 occurrence-only Soft-Critical continuation.

The run must resume from the validated P6/A4 step-1000 checkpoint.  Native SAR
input, both encoders, the pixel classifier, and the frozen MICB scorer remain
locked.  Only ``fusion_layer_3`` receives the low-variance occurrence gradient.
The mature P4 runtime still owns data loading, checkpointing, distillation, and
validation; command-line loss weights declare the occurrence-only objective.
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
    SAR_INPUT_PREFIX,
    lock_frozen_batch_norm,
)
from model.DamageFormer_S8_P6_SoftCritical import (  # noqa: E402
    DamageFormer_S8_P6_SoftCritical,
)
from script.standard_ML import train_S8_P4_native_sar_micb as runtime  # noqa: E402

P6C1_TRAINABLE_PREFIXES = ("fusion_layer_3.",)


def _argument_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value after {name}")
    return sys.argv[index + 1]


class _P6C1Factory:
    def __new__(cls, variant: str = "a4", **kwargs: Any):
        key = str(variant).lower()
        if key == "a0":
            return DamageFormer_S8_P4_NativeSAR_MICB(variant="a0", **kwargs)
        if key != "a4":
            raise ValueError("P6-C1 accepts only student a4 and teacher a0")
        return DamageFormer_S8_P6_SoftCritical(
            variant="a4",
            occurrence_temperature=float(
                os.environ.get("OCCURRENCE_TEMPERATURE", "0.5")
            ),
            **kwargs,
        )


def _module_for_prefix(model: nn.Module, prefix: str) -> nn.Module:
    module = model
    for name in prefix.rstrip(".").split("."):
        module = module[int(name)] if name.isdigit() else getattr(module, name)
    return module


def configure_p6c1_trainable(model: nn.Module, input_only: bool):
    """Resume-only policy: never reopen the A4 head or pixel classifier."""
    if input_only:
        raise RuntimeError(
            "P6-C1 is a step-1000 continuation; input_only must be false"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(P6C1_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    lock_frozen_batch_norm(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def set_p6c1_training_mode(model: nn.Module, input_only: bool) -> None:
    if input_only:
        raise RuntimeError("P6-C1 cannot enter input-only training mode")
    model.eval()
    for prefix in P6C1_TRAINABLE_PREFIXES:
        _module_for_prefix(model, prefix).train()
    lock_frozen_batch_norm(model)


def main() -> None:
    if (_argument_value("--variant") or "").lower() != "a4":
        raise ValueError("P6-C1 is locked to --variant a4")
    resume = _argument_value("--resume")
    expected_step = int(_argument_value("--expected_resume_step") or "-1")
    if not resume or expected_step != 1000:
        raise ValueError(
            "P6-C1 requires --resume and --expected_resume_step 1000"
        )
    output_value = _argument_value("--output_dir")
    if not output_value:
        raise ValueError("P6-C1 requires --output_dir")
    output_dir = Path(output_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "family": "S8-P6C1 occurrence-only SoftCritical",
        "resume_step": 1000,
        "phase_2_trainable": ["fusion_layer_3"],
        "frozen": [
            "native_sar_input",
            "rgb_encoder",
            "sar_encoder",
            "clf_2",
            "micb_scorer",
        ],
        "loss_contract": {
            "three_class_mil": 0.0,
            "severity": 0.0,
            "occurrence": 0.05,
            "intact_guard": 0.02,
            "ramp_steps": 500,
        },
        "memory_retrieval": "forbidden",
        "test_access": "forbidden",
    }
    (output_dir / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    runtime.DamageFormer_S8_P4_NativeSAR_MICB = _P6C1Factory
    runtime.configure_p4_trainable = configure_p6c1_trainable
    runtime.set_p4_training_mode = set_p6c1_training_mode
    runtime.main()

    summary_path = output_dir / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["experiment_contract"] = contract
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, summary_path)


if __name__ == "__main__":
    main()
