# -*- coding: utf-8 -*-
"""Controlled C0/C1 trainer for OA-HMD."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_OA_HMD import (  # noqa: E402
    OA_HMD_VARIANTS,
    VALID_OA_HMD_VARIANTS,
    build_oa_hmd_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    hierarchical_damage_nll,
    metrics_line,
)


OA_NEW_MODULE_TOKENS = (
    "p3_anchor",
    "stage3_difference_head",
    "affected_transition",
    "oa_imagenet_mean",
    "oa_imagenet_std",
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


def load_locked_a2_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    """Load the complete native-SAR feature path from locked H6-A2."""
    del require_exact
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(
            "OA-HMD native source must be H6-A2; "
            f"received variant={checkpoint.get('variant')!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key
        for key in missing
        if key.startswith(
            (
                "p3_anchor.",
                "stage3_difference_head.",
                "affected_transition",
                "oa_imagenet_mean",
                "oa_imagenet_std",
                "oa_native_sar_mean",
                "oa_native_sar_std",
            )
        )
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    if unexpected or illegal_missing:
        raise RuntimeError(
            "H6-A2-to-OA-HMD migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
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


def load_p3_anchor(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    path = str(args.joint_pretrained_checkpoint)
    if not path:
        raise RuntimeError("OA-HMD requires --joint_pretrained_checkpoint=P3")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _state(checkpoint)
    incompatible = model.p3_anchor.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 anchor load failed: {incompatible}")
    provenance["p3_anchor"] = {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": checkpoint.get("protocol"),
        "experiment": checkpoint.get("experiment"),
        "step": int(checkpoint.get("step", 0)),
        "variant": checkpoint.get("variant"),
        "selector": checkpoint.get("selector"),
        "metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
        "strict_load": True,
    }


def configure_freeze_policy(model: nn.Module) -> None:
    variant = str(getattr(model, "variant", "")).lower()
    for name, parameter in model.named_parameters():
        if variant == "c1":
            trainable = name.startswith(
                ("stage3_difference_head", "affected_transition")
            )
        else:
            # Validation-only C0 still needs one parameter for the inherited
            # trainer's optimizer construction contract.
            trainable = name == "hierarchy_transition"
        parameter.requires_grad_(trainable)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_frozen_in_eval(model: nn.Module) -> None:
    model.eval()
    if str(getattr(model, "variant", "")).lower() == "c1":
        model.stage3_difference_head.train()


def evaluate_and_restore(model, loader, device, ignore_index: int, max_batches: int = 0):
    metrics = evaluate_h10(model, loader, device, ignore_index, max_batches)
    keep_frozen_in_eval(model)
    return metrics


def update_selection_records(
    records: MutableMapping[str, Dict[str, object]],
    metrics: Mapping[str, float],
    step: int,
) -> Sequence[str]:
    selectors = {
        "valid": float(metrics["valid_miou"]),
        "event_worst": float(
            metrics["event_worst_official_style_4class_miou"]
        ),
        "event": float(metrics["event_mean_official_style_4class_miou"]),
        "damaged": float(metrics["damaged_iou"]),
    }
    improved = []
    for name, score in selectors.items():
        previous = float(records.get(name, {}).get("score", -math.inf))
        if score > previous:
            records[name] = {
                "score": score,
                "step": int(step),
                "metrics": dict(metrics),
            }
            improved.append(name)
    return improved


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
        "protocol": "OA_HMD_resume_v1",
        "experiment": "OA_HMD_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "selection_records": {
            key: dict(value) for key, value in selection_records.items()
        },
        "provenance": dict(provenance),
        "objective": "single_hierarchical_building_damage_nll",
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
        "protocol": "OA_HMD_best_weights_v1",
        "experiment": "OA_HMD_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "objective": "single_hierarchical_building_damage_nll",
        "test_accessed": False,
        "args": vars(args),
    }


def load_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "OA_HMD_resume_v1":
        raise RuntimeError("unexpected OA-HMD resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("OA-HMD resume variant mismatch")
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
            "protocol": "OA_HMD_training_summary_v1",
            "experiment": "OA_HMD_RGBSAR_DamageAssessment",
            "rgb_localization": "locked_p3",
            "main_prediction": "locked_p3_plus_safe_occurrence_residual",
            "sar_representation": "locked_h6_a2_native_raw_stage3",
            "difference": "R,S,absolute_difference,product",
            "fca": False,
            "flow": False,
            "conv3d": False,
            "router": False,
            "test_accessed": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    h4_train.VALID_H4_VARIANTS = VALID_OA_HMD_VARIANTS
    h4_train.H4_VARIANTS = OA_HMD_VARIANTS
    h4_train.build_h4_damageformer = build_oa_hmd_damageformer
    h4_train.load_h3_source = load_locked_a2_source
    h4_train.load_optional_pretraining = load_p3_anchor
    h4_train.configure_freeze_policy = configure_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_frozen_in_eval
    h4_train.evaluate = evaluate_and_restore
    h4_train.metrics_line = metrics_line
    h4_train.NEW_MODULE_TOKENS = OA_NEW_MODULE_TOKENS
    h4_train.EXTRA_SUPERVISED_LOSS = hierarchical_damage_nll
    h4_train.EXTRA_SUPERVISED_USES_NAMES = False
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.SELECT_INITIAL_CHECKPOINT = True
    h4_train.SECONDARY_BEST_SELECTORS = (
        "event_worst",
        "event",
        "damaged",
    )
    h4_train._update_selection_records = update_selection_records
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_resume
    print(
        "OA-HMD contract: C0 is the exact frozen P3 guard. C1 reads locked "
        "H6-A2 native-SAR/RGB raw encoder Stage3 features, applies independent "
        "projections and explicit [R,S,|R-S|,R*S] relations, and adds only "
        "an exactly-zero-initialised Intact-vs-Affected residual to P3. "
        "Damaged-vs-Destroyed odds and P3 localisation are preserved. No "
        "FCA, Flow, Conv3D, router, prototype, graph, or test selection.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
