# -*- coding: utf-8 -*-
"""Controlled trainer for H14 unique-aware weak-damage correction."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H14_UWDC import (  # noqa: E402
    H14_VARIANTS,
    VALID_H14_VARIANTS,
    build_h14_uwdc_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.h12_dgb_runtime import (  # noqa: E402
    EventBalancedSampler,
    evaluate_h10,
    hierarchical_damage_nll,
    metrics_line,
    update_h12_selection_records,
)
from bda_benchmark.script.standard_ML.h14_uwdc_runtime import (  # noqa: E402
    error_conditioned_occurrence_bce,
)


H14_NEW_MODULE_TOKENS = ("weak_damage_pyramid", "affected_transition")


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


def build_h14_from_environment(variant: str, pretrained_rgb: bool = True):
    return build_h14_uwdc_damageformer(variant, pretrained_rgb=pretrained_rgb)


def load_locked_h14_source(model: nn.Module, path: str, require_exact: bool):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_variant = str(checkpoint.get("variant", "")).lower()
    protocol = str(checkpoint.get("protocol", ""))
    if source_variant != "a2" or not protocol.startswith("H6_SDCR_"):
        raise RuntimeError(
            "H14 must start from locked H6-A2; "
            f"received protocol={protocol!r}, variant={source_variant!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key for key in missing if any(token in key for token in H14_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "u0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "controlled H6-A2-to-H14 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    config = getattr(model, "h14_config")
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": protocol,
        "source_variant": source_variant,
        "source_step": int(checkpoint.get("step", 0)),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
        "missing_new_keys": sorted(allowed_missing),
        "evidence_stages": list(config.evidence_stages),
        "error_conditioned_loss": bool(config.error_conditioned_loss),
    }


def no_optional_pretraining(model, args, provenance: MutableMapping[str, object]):
    del model, args, provenance


def configure_h14_freeze_policy(model: nn.Module) -> None:
    variant = str(getattr(model, "variant", "")).lower()
    for name, parameter in model.named_parameters():
        if variant == "u0":
            trainable = name == "hierarchy_transition"
        else:
            trainable = name.startswith(H14_NEW_MODULE_TOKENS)
        parameter.requires_grad_(trainable)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_h14_frozen_in_eval(model: nn.Module) -> None:
    model.eval()
    if getattr(model, "variant", "") != "u0" and hasattr(
        model, "weak_damage_pyramid"
    ):
        model.weak_damage_pyramid.train()


def evaluate_h14_and_restore(model, loader, device, ignore_index, max_batches=0):
    metrics = evaluate_h10(model, loader, device, ignore_index, max_batches)
    keep_h14_frozen_in_eval(model)
    return metrics


def make_event_balanced_sampler(dataset, generator):
    if os.environ.get("H14_EVENT_BALANCE", "1") != "1":
        return None
    return EventBalancedSampler(dataset, generator=generator)


def _objective(variant: str) -> str:
    return (
        "error_conditioned_occurrence_bce"
        if H14_VARIANTS[variant].error_conditioned_loss
        else "single_hierarchical_building_damage_nll"
    )


def checkpoint_payload(
    model,
    optimizer,
    scaler,
    step,
    best_metrics,
    selection_records,
    provenance,
    args,
) -> Dict[str, object]:
    return {
        "protocol": "H14_UWDC_resume_v1",
        "experiment": "H14_UWDC_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "selection_records": {k: dict(v) for k, v in selection_records.items()},
        "provenance": dict(provenance),
        "objective": _objective(args.variant),
        "selection_protocol": "event_worst_then_event_mean_then_damaged_then_valid",
        "test_accessed": False,
        "args": vars(args),
    }


def weights_payload(model, step, metrics, provenance, args, selector):
    return {
        "protocol": "H14_UWDC_best_weights_v1",
        "experiment": "H14_UWDC_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "objective": _objective(args.variant),
        "selection_protocol": "event_worst_then_event_mean_then_damaged_then_valid",
        "test_accessed": False,
        "args": vars(args),
    }


def load_h14_resume(path, model, optimizer, scaler, variant):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H14_UWDC_resume_v1":
        raise RuntimeError("unexpected H14-UWDC resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H14-UWDC resume variant mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("step", 0)),
        dict(checkpoint.get("best_metrics", {})),
        {k: dict(v) for k, v in checkpoint.get("selection_records", {}).items()},
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
    variant = str(summary.get("args", {}).get("variant", "u1"))
    config = H14_VARIANTS[variant]
    summary.update(
        {
            "protocol": "H14_UWDC_training_summary_v1",
            "experiment": "H14_UWDC_RGBSAR_DamageAssessment",
            "objective": _objective(variant),
            "base_expert": "locked_h6_a2",
            "evidence_stages": list(config.evidence_stages),
            "uses_common_features": True,
            "uses_unique_features": True,
            "uses_superpixels": False,
            "uses_graph": False,
            "severity_odds_preserved": True,
            "test_accessed": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    variant = "u1"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1].lower()
    os.environ["H14_VARIANT"] = variant
    config = H14_VARIANTS[variant]
    h4_train.VALID_H4_VARIANTS = VALID_H14_VARIANTS
    h4_train.H4_VARIANTS = H14_VARIANTS
    h4_train.build_h4_damageformer = build_h14_from_environment
    h4_train.load_h3_source = load_locked_h14_source
    h4_train.load_optional_pretraining = no_optional_pretraining
    h4_train.configure_freeze_policy = configure_h14_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_h14_frozen_in_eval
    h4_train.evaluate = evaluate_h14_and_restore
    h4_train.metrics_line = metrics_line
    h4_train.NEW_MODULE_TOKENS = H14_NEW_MODULE_TOKENS
    h4_train.EXTRA_SUPERVISED_LOSS = (
        error_conditioned_occurrence_bce
        if config.error_conditioned_loss
        else hierarchical_damage_nll
    )
    h4_train.EXTRA_SUPERVISED_USES_NAMES = False
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.SELECT_INITIAL_CHECKPOINT = True
    h4_train.SECONDARY_BEST_SELECTORS = ("event", "balance", "damaged")
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h14_resume
    h4_train._update_selection_records = update_h12_selection_records
    h4_train.TRAIN_SAMPLER_FACTORY = make_event_balanced_sampler
    h4_train.ZERO_LOSS_REFERENCE_KEY = "output_loc"
    print(
        "H14 contract: locked H6-A2; no superpixels/graph/flow/3D; "
        f"stages={config.evidence_stages}; common+unique evidence; "
        f"objective={_objective(variant)}; full soft RGB building support; "
        "only Intact/Affected odds change; test never selects.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
