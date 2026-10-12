# -*- coding: utf-8 -*-
"""Train CCU-BDA with the audited heterogeneous RGB/SAR pipeline.

This wrapper deliberately reuses the established crop, augmentation,
optimiser, CE/Lovasz, checkpoint and validation code.  It changes only the
model family, forces native one-channel SAR data, and records controlled
transfer provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset as BaseRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_CCU_BDA import (  # noqa: E402
    CCU_VARIANTS,
    VALID_CCU_VARIANTS,
    build_ccu_bda,
    load_controlled_variant_transfer,
    load_registration_transfer,
    zero_extra_regularization,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_Heterogeneous as base_train,
)
from bda_benchmark.script.standard_ML.ccu_bda_runtime import (  # noqa: E402
    evaluate_ccu,
    metrics_line,
)


TRANSFER_PROVENANCE: Dict[str, object] = {"kind": "none"}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeCCUDataset(BaseRGBSARDataset):
    """Map C0--C4 to the validated H3 native-SAR preprocessing contract."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["variant"] = "h3"
        super().__init__(*args, **kwargs)


def build_ccu_with_optional_transfer(variant: str, pretrained_rgb: bool = True):
    global TRANSFER_PROVENANCE
    model = build_ccu_bda(variant, pretrained_rgb=pretrained_rgb)
    controlled = os.environ.get("CCU_SOURCE_CKPT", "").strip()
    registration = os.environ.get("REGISTRATION_CKPT", "").strip()
    if controlled and registration:
        raise RuntimeError(
            "CCU_SOURCE_CKPT and REGISTRATION_CKPT are mutually exclusive; "
            "run them as separate ablations"
        )
    if variant in ("e1", "e2", "e3") and not controlled:
        raise RuntimeError(
            f"{variant.upper()} requires CCU_SOURCE_CKPT pointing to the locked "
            "H6-A2 best checkpoint"
        )
    if variant in ("e1", "e2", "e3") and registration:
        raise RuntimeError("E1/E2/E3 do not accept registration transfer")
    if controlled:
        details = load_controlled_variant_transfer(model, controlled)
        TRANSFER_PROVENANCE = {
            "kind": "controlled_variant",
            "sha256": sha256(controlled),
            **details,
        }
    elif registration:
        details = load_registration_transfer(model, registration)
        TRANSFER_PROVENANCE = {
            "kind": "registration_transfer",
            "sha256": sha256(registration),
            **details,
        }
    else:
        TRANSFER_PROVENANCE = {"kind": "none"}
    return model


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_metrics: Mapping[str, float],
    args,
) -> Dict[str, object]:
    return {
        "protocol": "CCU_BDA_resume_v1",
        "experiment": "CCU_BDA_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "provenance": dict(TRANSFER_PROVENANCE),
        "args": vars(args),
        "teacher": False,
        "distillation": False,
        "extra_supervised_losses": False,
        "test_accessed": False,
    }


def best_weights_payload(
    model: nn.Module,
    step: int,
    best_metrics: Mapping[str, float],
    args,
) -> Dict[str, object]:
    return {
        "protocol": "CCU_BDA_best_weights_v1",
        "experiment": "CCU_BDA_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "selector": "validation_valid_miou",
        "step": int(step),
        "model": model.state_dict(),
        "best_metrics": dict(best_metrics),
        "provenance": dict(TRANSFER_PROVENANCE),
        "args": vars(args),
        "teacher": False,
        "distillation": False,
        "extra_supervised_losses": False,
        "test_accessed": False,
    }


def load_ccu_resume(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    require_optimizer: bool,
):
    """Resume without losing the original controlled-transfer provenance."""

    global TRANSFER_PROVENANCE
    if not path:
        return 0, {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "CCU_BDA_resume_v1":
        raise RuntimeError("unexpected CCU-BDA resume protocol")
    if checkpoint.get("variant") != getattr(model, "variant", None):
        raise RuntimeError("CCU-BDA resume variant mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
    elif require_optimizer:
        raise RuntimeError("CCU-BDA resume checkpoint has no optimizer state")
    TRANSFER_PROVENANCE = dict(checkpoint.get("provenance", {"kind": "none"}))
    return int(checkpoint.get("step", 0)), dict(checkpoint.get("best_metrics", {}))


def rewrite_summary() -> None:
    try:
        output_dir = Path(sys.argv[sys.argv.index("--output_dir") + 1])
    except (ValueError, IndexError):
        return
    path = output_dir / "training_summary.json"
    if not path.is_file():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update(
        {
            "protocol": "CCU_BDA_training_summary_v1",
            "experiment": "CCU_BDA_RGBSAR_DamageAssessment",
            "prediction_unit": "pixel",
            "rgb_localization": "rgb_only",
            "sar_input": "native_one_channel_train_only_statistics",
            "provenance": dict(TRANSFER_PROVENANCE),
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "test_accessed": False,
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    base_train.VALID_VARIANTS = VALID_CCU_VARIANTS
    base_train.VARIANTS = CCU_VARIANTS
    base_train.build_heterogeneous_damageformer = build_ccu_with_optional_transfer
    base_train.common_unique_regularization = zero_extra_regularization
    base_train.HeterogeneousRGBSARDataset = NativeCCUDataset
    base_train.evaluate = evaluate_ccu
    base_train.metrics_line = metrics_line
    base_train.checkpoint_payload = checkpoint_payload
    base_train.best_weights_payload = best_weights_payload
    base_train.load_resume = load_ccu_resume
    print(
        "CCU-BDA contract: E1/E2/E3 strictly inherit and freeze the same H6-A2 "
        "base. E1 is same-position difference, E2 is multi-scale 3D "
        "spatio-temporal difference, and E3 is 5x5 soft local matching. "
        "All use a zero residual transition. "
        "E0-unsupported stage3/entropy/margin/offset/cycle signals are excluded. "
        "No teacher, KD, prototype memory, translation, or additional supervised "
        "loss. Test is never used for selection.",
        flush=True,
    )
    base_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
