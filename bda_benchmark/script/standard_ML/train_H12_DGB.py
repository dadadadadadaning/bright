# -*- coding: utf-8 -*-
"""Controlled trainer for H12 Damage-Gate Bypass."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H12_DGB import (  # noqa: E402
    H12_VARIANTS,
    OCCURRENCE_CANDIDATE_BOUND,
    VALID_H12_VARIANTS,
    build_h12_dgb_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.h12_dgb_runtime import (  # noqa: E402
    EventBalancedSampler,
    evaluate_h10,
    fn_aware_hierarchical_nll,
    hierarchical_damage_nll,
    metrics_line,
    update_h12_selection_records,
)


H12_NEW_MODULE_TOKENS = ("bypass_heads", "affected_transition")


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


def parse_bypass_stages() -> Sequence[str] | None:
    raw = os.environ.get("H12_BYPASS_STAGES", "").strip()
    if not raw:
        return None
    return tuple(raw.replace(",", " ").split())


def build_h12_from_environment(
    variant: str, pretrained_rgb: bool = True
):
    return build_h12_dgb_damageformer(
        variant, pretrained_rgb=pretrained_rgb, bypass_stages=parse_bypass_stages()
    )


def load_locked_h12_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_variant = str(checkpoint.get("variant", "")).lower()
    protocol = str(checkpoint.get("protocol", ""))
    if source_variant != "a2" or not protocol.startswith("H6_SDCR_"):
        raise RuntimeError(
            "H12 must start from the locked H6-A2 checkpoint; "
            f"received protocol={protocol!r}, variant={source_variant!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key for key in missing if any(token in key for token in H12_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "d0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "controlled H6-A2-to-H12 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": protocol,
        "source_variant": source_variant,
        "source_step": int(checkpoint.get("step", 0)),
        "source_selector": checkpoint.get("selector"),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
        "missing_new_keys": sorted(allowed_missing),
        "bypass_stages": list(getattr(model, "h12_config").bypass_stages),
        "fn_aware_occurrence": bool(getattr(model, "h12_config").fn_aware_occurrence),
    }


def no_optional_pretraining(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    del model, args, provenance


def configure_h12_freeze_policy(model: nn.Module) -> None:
    variant = str(getattr(model, "variant", "")).lower()
    for name, parameter in model.named_parameters():
        trainable = False
        if variant in ("d1", "d2", "d3"):
            trainable = name.startswith(H12_NEW_MODULE_TOKENS)
        elif variant == "d0":
            trainable = name == "hierarchy_transition"
        parameter.requires_grad_(trainable)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_h12_frozen_in_eval(model: nn.Module) -> None:
    model.eval()
    variant = str(getattr(model, "variant", "")).lower()
    if variant in ("d1", "d2", "d3") and hasattr(model, "bypass_heads"):
        model.bypass_heads.train()


def evaluate_h12_and_restore(
    model,
    loader,
    device,
    ignore_index: int,
    max_batches: int = 0,
):
    metrics = evaluate_h10(model, loader, device, ignore_index, max_batches)
    keep_h12_frozen_in_eval(model)
    return metrics


def make_extra_supervised(variant: str):
    use_fn = H12_VARIANTS[variant].fn_aware_occurrence

    def loss_fn(outputs, labels, ignore_index):
        if use_fn:
            return fn_aware_hierarchical_nll(outputs, labels, ignore_index)
        return hierarchical_damage_nll(outputs, labels, ignore_index)

    return loss_fn


def make_event_balanced_sampler(dataset, generator):
    if os.environ.get("H12_EVENT_BALANCE", "1") != "1":
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
        "protocol": "H12_DGB_resume_v1",
        "experiment": "H12_DGB_RGBSAR_DamageAssessment",
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
        "protocol": "H12_DGB_best_weights_v1",
        "experiment": "H12_DGB_RGBSAR_DamageAssessment",
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


def load_h12_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H12_DGB_resume_v1":
        raise RuntimeError("unexpected H12-DGB resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H12-DGB resume variant mismatch")
    nonfinite_keys = [
        key
        for key, value in checkpoint["model"].items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if nonfinite_keys:
        raise RuntimeError(
            "H12-DGB resume contains non-finite model tensors and cannot be "
            f"continued: {nonfinite_keys}"
        )
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
            "protocol": "H12_DGB_training_summary_v1",
            "experiment": "H12_DGB_RGBSAR_DamageAssessment",
            "prediction_unit": "pixel_soft_hierarchy",
            "objective": "single_hierarchical_building_damage_nll",
            "rgb_localization": "frozen_p3",
            "base_expert": "locked_h6_a2",
            "selection_protocol": (
                "event_worst -> event_mean -> damaged_iou -> valid_miou"
            ),
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "test_accessed": False,
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    variant = "d1"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1].lower()
    os.environ["H12_VARIANT"] = variant
    h4_train.VALID_H4_VARIANTS = VALID_H12_VARIANTS
    h4_train.H4_VARIANTS = H12_VARIANTS
    h4_train.build_h4_damageformer = build_h12_from_environment
    h4_train.load_h3_source = load_locked_h12_source
    h4_train.load_optional_pretraining = no_optional_pretraining
    h4_train.configure_freeze_policy = configure_h12_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_h12_frozen_in_eval
    h4_train.evaluate = evaluate_h12_and_restore
    h4_train.metrics_line = metrics_line
    h4_train.NEW_MODULE_TOKENS = H12_NEW_MODULE_TOKENS
    h4_train.EXTRA_SUPERVISED_LOSS = make_extra_supervised(variant)
    h4_train.EXTRA_SUPERVISED_USES_NAMES = False
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.SELECT_INITIAL_CHECKPOINT = True
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h12_resume
    h4_train._update_selection_records = update_h12_selection_records
    h4_train.TRAIN_SAMPLER_FACTORY = make_event_balanced_sampler
    h4_train.ZERO_LOSS_REFERENCE_KEY = "output_loc"
    print(
        "H12 contract: D0 is locked H6-A2; D1 adds a Stage2 Intact/Affected "
        "bypass; D2 adds Stage2+Stage3; D3 only replaces the Affected term "
        "with FN-aware focal loss. Localisation, native SAR, fusion and the "
        "Damaged/Destroyed odds stay frozen. Checkpoints are selected by "
        "event-worst, then event-mean, then Damaged IoU, then valid mIoU. "
        "Test is never used for model selection.",
        flush=True,
    )
    print(
        "H12 numeric contract v3: A2 no_grad | fp32 stable moments | "
        f"candidate bound={OCCURRENCE_CANDIDATE_BOUND} | zero output layer | "
        "bounded positive transition | fail-fast non-finite checks | "
        f"model={__file__}",
        flush=True,
    )
    h4_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
