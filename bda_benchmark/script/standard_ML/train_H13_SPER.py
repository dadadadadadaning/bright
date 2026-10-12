# -*- coding: utf-8 -*-
"""Controlled trainer for H13 superpixel evidence reasoning."""
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

from bda_benchmark.model.DamageFormer_RGBSAR_H13_SPER import (  # noqa: E402
    H13_VARIANTS,
    VALID_H13_VARIANTS,
    build_h13_sper_damageformer,
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


H13_NEW_MODULE_TOKENS = ("region_head", "affected_transition")


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


def superpixel_parameters() -> Dict[str, float | int]:
    return {
        "cell_size": int(os.environ.get("H13_CELL_SIZE", "2")),
        "color_temperature": float(os.environ.get("H13_COLOR_TEMPERATURE", "0.50")),
        "compactness": float(os.environ.get("H13_COMPACTNESS", "0.75")),
    }


def build_h13_from_environment(variant: str, pretrained_rgb: bool = True):
    return build_h13_sper_damageformer(
        variant,
        pretrained_rgb=pretrained_rgb,
        **superpixel_parameters(),
    )


def load_locked_h13_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_variant = str(checkpoint.get("variant", "")).lower()
    protocol = str(checkpoint.get("protocol", ""))
    if source_variant != "a2" or not protocol.startswith("H6_SDCR_"):
        raise RuntimeError(
            "H13 must start from the locked H6-A2 checkpoint; "
            f"received protocol={protocol!r}, variant={source_variant!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key for key in missing if any(token in key for token in H13_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "s0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "controlled H6-A2-to-H13 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    config = getattr(model, "h13_config")
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
        "assignment_mode": config.assignment_mode,
        "graph_refinement": bool(config.graph_refinement),
        "superpixel_parameters": superpixel_parameters(),
    }


def no_optional_pretraining(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    del model, args, provenance


def configure_h13_freeze_policy(model: nn.Module) -> None:
    variant = str(getattr(model, "variant", "")).lower()
    for name, parameter in model.named_parameters():
        if variant == "s0":
            trainable = name == "hierarchy_transition"
        else:
            trainable = name.startswith(H13_NEW_MODULE_TOKENS)
        parameter.requires_grad_(trainable)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_h13_frozen_in_eval(model: nn.Module) -> None:
    model.eval()
    if getattr(model, "variant", "") != "s0" and hasattr(model, "region_head"):
        model.region_head.train()


def evaluate_h13_and_restore(
    model,
    loader,
    device,
    ignore_index: int,
    max_batches: int = 0,
):
    metrics = evaluate_h10(model, loader, device, ignore_index, max_batches)
    keep_h13_frozen_in_eval(model)
    return metrics


def make_event_balanced_sampler(dataset, generator):
    if os.environ.get("H13_EVENT_BALANCE", "1") != "1":
        return None
    return EventBalancedSampler(dataset, generator=generator)


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
        "protocol": "H13_SPER_resume_v1",
        "experiment": "H13_SPER_RGBSAR_DamageAssessment",
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
        "selection_protocol": "event_worst_then_event_mean_then_damaged_iou_then_valid_miou",
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
        "protocol": "H13_SPER_best_weights_v1",
        "experiment": "H13_SPER_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "objective": "single_hierarchical_building_damage_nll",
        "selection_protocol": "event_worst_then_event_mean_then_damaged_iou_then_valid_miou",
        "teacher": False,
        "distillation": False,
        "test_accessed": False,
        "args": vars(args),
    }


def load_h13_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H13_SPER_resume_v1":
        raise RuntimeError("unexpected H13-SPER resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H13-SPER resume variant mismatch")
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
            "protocol": "H13_SPER_training_summary_v1",
            "experiment": "H13_SPER_RGBSAR_DamageAssessment",
            "prediction_unit": "soft_superpixel_evidence_pixel_output",
            "objective": "single_hierarchical_building_damage_nll",
            "base_expert": "locked_h6_a2",
            "rgb_localization": "frozen_p3_final_residual_support_only",
            "selection_protocol": "event_worst -> event_mean -> damaged_iou -> valid_miou",
            "superpixel_parameters": superpixel_parameters(),
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "test_accessed": False,
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    variant = "s2"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1].lower()
    os.environ["H13_VARIANT"] = variant
    h4_train.VALID_H4_VARIANTS = VALID_H13_VARIANTS
    h4_train.H4_VARIANTS = H13_VARIANTS
    h4_train.build_h4_damageformer = build_h13_from_environment
    h4_train.load_h3_source = load_locked_h13_source
    h4_train.load_optional_pretraining = no_optional_pretraining
    h4_train.configure_freeze_policy = configure_h13_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_h13_frozen_in_eval
    h4_train.evaluate = evaluate_h13_and_restore
    h4_train.metrics_line = metrics_line
    h4_train.NEW_MODULE_TOKENS = H13_NEW_MODULE_TOKENS
    h4_train.EXTRA_SUPERVISED_LOSS = hierarchical_damage_nll
    h4_train.EXTRA_SUPERVISED_USES_NAMES = False
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.SELECT_INITIAL_CHECKPOINT = True
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h13_resume
    h4_train._update_selection_records = update_h12_selection_records
    h4_train.TRAIN_SAMPLER_FACTORY = make_event_balanced_sampler
    h4_train.ZERO_LOSS_REFERENCE_KEY = "output_loc"
    config = H13_VARIANTS[variant]
    print(
        "H13 contract: S0 is exact H6-A2; S1 replaces pixel evidence with "
        "regular region pooling; S2 uses one pre-RGB-guided soft partition "
        "for both modalities; S3 adds one boundary-gated adjacency step. "
        "Regions cover the scene, but only the frozen RGB building support "
        "may receive an occurrence residual. Severity odds remain unchanged.",
        flush=True,
    )
    print(
        f"H13 active design: assignment={config.assignment_mode} | "
        f"graph={config.graph_refinement} | params={superpixel_parameters()} | "
        "one hierarchical NLL | no auxiliary superpixel loss | test never selects",
        flush=True,
    )
    h4_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
