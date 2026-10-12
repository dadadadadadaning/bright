#!/usr/bin/env python3
"""Isolated adapter that trains the hierarchical decoder with the P5 runtime."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for path in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.DamageFormer_S8_P5_DCMF import (  # noqa: E402
    P5_LATE_PREFIXES, P5_NEW_PREFIXES, _extract_state_dict,
)
from model.DamageFormer_S8_P5_Hierarchical import (  # noqa: E402
    DamageFormer_S8_P5_Hierarchical,
)
from script.standard_ML import train_S8_P5_dcmf as runtime  # noqa: E402

HIERARCHY_PREFIX = "hierarchical_decoder."


def load_source(model: nn.Module, checkpoint_path: str) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(checkpoint)
    transferable = {
        key: value for key, value in state.items() if not key.startswith(P5_NEW_PREFIXES)
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    allowed = (*P5_NEW_PREFIXES, HIERARCHY_PREFIX)
    illegal = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
    if illegal or incompatible.unexpected_keys:
        raise RuntimeError(
            f"hierarchical P5 migration failed: missing={illegal}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "new_prefixes": list(P5_NEW_PREFIXES) + [HIERARCHY_PREFIX],
        "prediction_safe_zero_transition": True,
    }


def configure(model: nn.Module, warmup: bool) -> Dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    active = (*P5_NEW_PREFIXES, HIERARCHY_PREFIX)
    if not warmup:
        active = (*active, *P5_LATE_PREFIXES)
    for name, parameter in model.named_parameters():
        if name.startswith(active):
            parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def lock(model: nn.Module, warmup: bool) -> None:
    model.eval()
    model.encoder_2.train()
    model.fusion_layer_2.train()
    model.fusion_layer_3.train()
    model.hierarchical_decoder.train()
    if not warmup:
        model.clf_2.train()
        model.micb_dsmil.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def optimizer(model, sar_learning_rate, fusion_learning_rate,
              core_learning_rate, weight_decay, recovery_heads=None,
              recovery_learning_rate=1e-4):
    if recovery_heads is not None:
        raise ValueError("hierarchical P5 does not combine C2 recovery heads")
    groups = {"sar": [], "fusion": [], "core": []}
    for name, parameter in model.named_parameters():
        if name.startswith(("encoder_2.", "fusion_layer_2.")):
            groups["sar"].append(parameter)
        elif name.startswith(("fusion_layer_3.", HIERARCHY_PREFIX)):
            groups["fusion"].append(parameter)
        elif name.startswith(("clf_2.", "micb_dsmil.")):
            groups["core"].append(parameter)
    return torch.optim.AdamW([
        {"params": groups["sar"], "lr": sar_learning_rate, "name": "native_sar"},
        {"params": groups["fusion"], "lr": fusion_learning_rate, "name": "dcmf_hierarchy"},
        {"params": groups["core"], "lr": core_learning_rate, "name": "p3_heads"},
    ], weight_decay=weight_decay)


def main() -> None:
    runtime.DamageFormer_S8_P5_DCMF = DamageFormer_S8_P5_Hierarchical
    runtime.load_p3_checkpoint_for_p5 = load_source
    runtime.configure_p5_trainable = configure
    runtime.lock_p5_frozen_modules = lock
    runtime.build_optimizer = optimizer
    print(
        "P5-H contract: original train/val lists; native SAR + DCMF retained; "
        "soft occurrence/severity decoder is inside the main network; no meta split "
        "and no test-set model selection.", flush=True
    )
    runtime.main()


if __name__ == "__main__":
    main()
