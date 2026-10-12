# -*- coding: utf-8 -*-
"""Controlled trainer for H10 hierarchical 3D-STA damage assessment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (  # noqa: E402
    H10_VARIANTS,
    VALID_H10_VARIANTS,
    build_h10_h3d_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    hierarchical_damage_nll,
    metrics_line,
)


H10_NEW_MODULE_TOKENS = (
    "affected_head",
    "affected_transition",
    "severity_head",
    "severity_transition",
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    value = checkpoint.get("model", checkpoint)
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint does not contain a model state dictionary")
    return value  # type: ignore[return-value]


def load_locked_h10_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    """Load H6-A2 for H0-H3 or a trained H10-H3 for H4."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target_variant = str(getattr(model, "variant", "")).lower()
    source_variant = str(checkpoint.get("variant", "")).lower()
    protocol = str(checkpoint.get("protocol", ""))
    if target_variant == "h4":
        if source_variant != "h3" or not protocol.startswith("H10_H3D_"):
            raise RuntimeError(
                "H10-H4 requires a validation-selected H10-H3 checkpoint; "
                f"received protocol={protocol!r}, variant={source_variant!r}"
            )
    elif source_variant != "a2":
        raise RuntimeError(
            "H10-H0/H1/H2/H3 must start from the locked H6-A2 checkpoint; "
            f"received variant={source_variant!r}"
        )

    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key for key in missing if any(token in key for token in H10_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or target_variant in ("h0", "h1"))
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "controlled H10 source migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if target_variant == "h4" and any("affected_" in key for key in missing):
        raise RuntimeError("H10-H4 source did not provide the trained H3 occurrence branch")
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": checkpoint.get("protocol"),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "source_selector": checkpoint.get("selector"),
        "missing_new_keys": sorted(allowed_missing),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
    }


def no_optional_pretraining(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    del model, args, provenance


def configure_h10_freeze_policy(model: nn.Module) -> None:
    """Train one decision at a time and freeze the complete H6-A2 base."""

    variant = str(getattr(model, "variant", "")).lower()
    for name, parameter in model.named_parameters():
        trainable = False
        if variant in ("h2", "h3"):
            trainable = name.startswith(("affected_head", "affected_transition"))
        elif variant == "h4":
            trainable = name.startswith(("severity_head", "severity_transition"))
        elif variant in ("h0", "h1"):
            # The validated H4 driver constructs an optimizer before entering
            # validation-only mode.  This already-existing scalar is enabled
            # solely to satisfy that interface; H0/H1 launchers forbid train.
            trainable = name == "hierarchy_transition"
        parameter.requires_grad_(trainable)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_h10_frozen_in_eval(model: nn.Module) -> None:
    """Prevent stochastic/state drift in every inherited A2 module."""

    model.eval()
    variant = str(getattr(model, "variant", "")).lower()
    if variant in ("h2", "h3") and hasattr(model, "affected_head"):
        model.affected_head.train()
    if variant == "h4" and hasattr(model, "severity_head"):
        model.severity_head.train()


def evaluate_h10_and_restore(
    model,
    loader,
    device,
    ignore_index: int,
    max_batches: int = 0,
):
    metrics = evaluate_h10(model, loader, device, ignore_index, max_batches)
    keep_h10_frozen_in_eval(model)
    return metrics


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_metrics: Mapping[str, float],
    selection_records: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
    args,
) -> Dict[str, object]:
    return {
        "protocol": "H10_H3D_resume_v1",
        "experiment": "H10_H3D_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "selection_records": {key: dict(value) for key, value in selection_records.items()},
        "provenance": dict(provenance),
        "objective": "single_hierarchical_building_damage_nll",
        "teacher": False,
        "distillation": False,
        "test_accessed": False,
        "args": vars(args),
    }


def weights_payload(
    model: nn.Module,
    step: int,
    metrics: Mapping[str, float],
    provenance: Mapping[str, object],
    args,
    selector: str,
) -> Dict[str, object]:
    return {
        "protocol": "H10_H3D_best_weights_v1",
        "experiment": "H10_H3D_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "objective": "single_hierarchical_building_damage_nll",
        "teacher": False,
        "distillation": False,
        "test_accessed": False,
        "args": vars(args),
    }


def load_h10_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H10_H3D_resume_v1":
        raise RuntimeError("unexpected H10-H3D resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H10-H3D resume variant mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("step", 0)),
        dict(checkpoint.get("best_metrics", {})),
        {
            key: dict(value)
            for key, value in checkpoint.get("selection_records", {}).items()
        },
        dict(checkpoint.get("provenance", {})),
    )


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
            "protocol": "H10_H3D_training_summary_v1",
            "experiment": "H10_H3D_RGBSAR_DamageAssessment",
            "prediction_unit": "pixel_soft_hierarchy",
            "objective": "single_hierarchical_building_damage_nll",
            "rgb_localization": "frozen_p3",
            "base_expert": "locked_h6_a2",
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "test_accessed": False,
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    h4_train.VALID_H4_VARIANTS = VALID_H10_VARIANTS
    h4_train.H4_VARIANTS = H10_VARIANTS
    h4_train.build_h4_damageformer = build_h10_h3d_damageformer
    h4_train.load_h3_source = load_locked_h10_source
    h4_train.load_optional_pretraining = no_optional_pretraining
    h4_train.configure_freeze_policy = configure_h10_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_h10_frozen_in_eval
    h4_train.evaluate = evaluate_h10_and_restore
    h4_train.metrics_line = metrics_line
    h4_train.NEW_MODULE_TOKENS = H10_NEW_MODULE_TOKENS
    h4_train.EXTRA_SUPERVISED_LOSS = hierarchical_damage_nll
    h4_train.EXTRA_SUPERVISED_USES_NAMES = False
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.SELECT_INITIAL_CHECKPOINT = True
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h10_resume
    print(
        "H10 contract: H0 is locked H6-A2; H1 exposes an exactly equivalent "
        "soft hierarchy; H2 and H3 train only the intact-vs-affected residual "
        "using matched 2-D or Lite-3D-STA Stage4 evidence; H4 freezes the "
        "selected H3 occurrence branch and trains only conditional "
        "damaged-vs-destroyed severity. P3 localisation and the complete A2 "
        "base remain frozen. One hierarchical likelihood is optimised; no "
        "teacher, KD, FCA, prototype, contrastive, or alignment loss is used. "
        "Test is never used for model selection.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
