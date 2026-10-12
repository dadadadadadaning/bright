# -*- coding: utf-8 -*-
"""Train H6-SDCR by reusing the validated H4 data/metric pipeline."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    H6_VARIANTS,
    VALID_H6_VARIANTS,
    build_h6_sdcr_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)


H6_NEW_MODULE_TOKENS = (
    "p3_spatial_anchor",
    "p3_full_anchor",
    "sensor_damage_adapters",
    "safe_residual_router",
    "native_sar_mean",
    "native_sar_std",
    "imagenet_mean",
    "imagenet_std",
)

_BASE_H4_EVALUATE = h4_train.evaluate


def event_name(sample_name: str) -> str:
    """Return the disaster-event id without the chip index."""

    return re.sub(r"_\d+$", "", str(sample_name))


def tempered_event_balanced_sampler(
    dataset,
    generator: torch.Generator,
    power: float = 0.5,
) -> WeightedRandomSampler:
    """Reduce event-size bias without forcing an artificial uniform split.

    ``power=0`` is ordinary image sampling and ``power=1`` is full inverse
    event-frequency sampling.  The square-root default is deliberately mild:
    BRIGHT train/validation/test event frequencies are already similar, while
    full inverse-frequency sampling over-represents very small events.
    """

    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("event balance power must lie in [0, 1]")
    names = [event_name(name) for name in dataset.data_list]
    counts = Counter(names)
    weights = torch.tensor(
        [counts[name] ** (-power) for name in names], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(names),
        replacement=True,
        generator=generator,
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


def load_h4_a4_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a4":
        raise RuntimeError(
            "H6 source must be H4-A4, "
            f"received variant={checkpoint.get('variant')!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key
        for key in missing
        if any(token in key for token in H6_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "a0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "H4-A4-to-H6 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "source_selector": checkpoint.get("selector"),
        "missing_new_keys": sorted(allowed_missing),
        "unexpected_keys": sorted(unexpected),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
    }


def _extract_prefixed(
    state: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def load_p3_rgb_anchor(model: nn.Module, path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _state(checkpoint)
    encoder = _extract_prefixed(state, "encoder_1.swin_transformer.features.")
    fusion = _extract_prefixed(state, "fusion_layer_1.")
    head = _extract_prefixed(state, "clf_1.")
    if not encoder or not fusion or not head:
        raise RuntimeError(
            "P3 checkpoint lacks encoder_1/fusion_layer_1/clf_1 RGB anchor keys"
        )
    if not hasattr(model, "p3_spatial_anchor"):
        raise RuntimeError("current H6 variant does not declare the P3 RGB anchor")
    model.rgb_encoder.features.load_state_dict(encoder, strict=True)
    model.p3_spatial_anchor.fusion.load_state_dict(fusion, strict=True)
    model.p3_spatial_anchor.head.load_state_dict(head, strict=True)
    full_anchor_keys = 0
    if hasattr(model, "p3_full_anchor"):
        incompatible = model.p3_full_anchor.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "P3 full safety-anchor strict load failed: "
                f"{incompatible}"
            )
        full_anchor_keys = len(state)
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "experiment": checkpoint.get("experiment"),
        "step": int(checkpoint.get("step", 0)),
        "metrics": dict(checkpoint.get("metrics", {})),
        "loaded_encoder_keys": len(encoder),
        "loaded_fusion_keys": len(fusion),
        "loaded_head_keys": len(head),
        "loaded_full_anchor_keys": full_anchor_keys,
        "teacher": False,
        "distillation": False,
    }


def load_a2_native_expert(model: nn.Module, path: str) -> Dict[str, object]:
    """Warm-start the native expert from the validated A2 checkpoint.

    Only keys declared by the destination model and sharing the exact shape
    are imported.  P3 safety-anchor and residual-router keys are necessarily
    absent from A2 and remain at their safe initialisation.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(
            "H6-B native expert source must be A2, "
            f"received variant={checkpoint.get('variant')!r}"
        )
    source = _state(checkpoint)
    destination = model.state_dict()
    selected = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    if not selected:
        raise RuntimeError("A2 checkpoint has no compatible native-expert keys")
    incompatible = model.load_state_dict(selected, strict=False)
    illegal_unexpected = list(incompatible.unexpected_keys)
    if illegal_unexpected:
        raise RuntimeError(
            f"A2 native-expert migration produced unexpected keys: {illegal_unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "source_selector": checkpoint.get("selector"),
        "loaded_keys": len(selected),
        "excluded_keys": len(source) - len(selected),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
    }


def load_optional_p3(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    provenance["sar_freeze_mode"] = resolve_sar_freeze_mode()
    if args.variant == "a0":
        return
    if args.variant in ("b1", "b2", "b1r"):
        if not args.sar_pretrained_checkpoint:
            raise RuntimeError("H6-B requires the locked H6-A2 expert checkpoint")
        provenance["a2_native_expert"] = load_a2_native_expert(
            model, args.sar_pretrained_checkpoint
        )
    # The shared H4 parser already exposes this neutral checkpoint argument.
    # H6's launcher names it P3_RGB_CKPT and passes it through here.
    if not args.joint_pretrained_checkpoint:
        raise RuntimeError("H6 non-control variants require the P3 anchor checkpoint")
    provenance["p3_rgb_anchor"] = load_p3_rgb_anchor(
        model, args.joint_pretrained_checkpoint
    )


VALID_SAR_FREEZE_MODES = ("f0_shallow_frozen", "f1_stage2_to_4", "f2_all_sar")


def resolve_sar_freeze_mode(value: str | None = None) -> str:
    mode = value or os.environ.get("H6_SAR_FREEZE_MODE", "f0_shallow_frozen")
    if mode not in VALID_SAR_FREEZE_MODES:
        raise ValueError(
            f"unknown H6 SAR freeze mode {mode!r}; expected {VALID_SAR_FREEZE_MODES}"
        )
    return mode


def _frozen_sar_prefixes(mode: str) -> tuple[str, ...]:
    if mode == "f0_shallow_frozen":
        return (
            "sar_encoder.stem",
            "sar_encoder.stages.0",
            "sar_encoder.stages.1",
            "sar_encoder.downsamples.0",
            "sar_encoder.downsamples.1",
        )
    if mode == "f1_stage2_to_4":
        # Preserve the lowest-level speckle/edge extractor, while allowing the
        # stage-1 -> stage-2 projection and every later SAR representation to adapt.
        return ("sar_encoder.stem", "sar_encoder.stages.0")
    return ()


def configure_freeze_policy(model: nn.Module, mode: str | None = None) -> None:
    """Freeze RGB geometry and select one controlled SAR adaptation depth."""

    router_only = getattr(model, "variant", "") == "b1r"
    sar_freeze_mode = resolve_sar_freeze_mode(mode)
    frozen_sar_prefixes = _frozen_sar_prefixes(sar_freeze_mode)
    setattr(model, "_h6_sar_freeze_mode", sar_freeze_mode)
    for name, parameter in model.named_parameters():
        permanently_frozen = (
            name.startswith("rgb_encoder")
            or name.startswith("localization_decoder")
            or name.startswith("localization_head")
            or name.startswith("p3_spatial_anchor")
            or name.startswith("p3_full_anchor")
            or name.startswith(frozen_sar_prefixes)
        )
        parameter.requires_grad_(
            name.startswith("safe_residual_router")
            if router_only
            else not permanently_frozen
        )
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_frozen_modules_in_eval(model: nn.Module) -> None:
    model.rgb_encoder.eval()
    model.localization_decoder.eval()
    model.localization_head.eval()
    if hasattr(model, "p3_spatial_anchor"):
        model.p3_spatial_anchor.eval()
    if hasattr(model, "p3_full_anchor"):
        model.p3_full_anchor.eval()
    sar_freeze_mode = resolve_sar_freeze_mode(
        getattr(model, "_h6_sar_freeze_mode", None)
    )
    frozen_sar_prefixes = _frozen_sar_prefixes(sar_freeze_mode)
    named_modules = dict(model.named_modules())
    for prefix in frozen_sar_prefixes:
        module = named_modules.get(prefix)
        if module is not None:
            module.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def evaluate_h6(model, loader, device, ignore_index: int, max_batches: int = 0):
    """Run the shared evaluator and immediately restore the freeze contract."""

    metrics = _BASE_H4_EVALUATE(
        model, loader, device, ignore_index, max_batches
    )
    keep_frozen_modules_in_eval(model)
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
        "protocol": "H6_SDCR_resume_v1",
        "experiment": "H6_SDCR_RGBSAR_DamageFormer",
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
        "args": vars(args),
        "test_accessed": bool(getattr(args, "test_every_validation", False)),
        "test_monitor_policy": (
            "after_every_validation_log_only_never_selector"
            if getattr(args, "test_every_validation", False)
            else "not_accessed"
        ),
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
        "protocol": "H6_SDCR_best_weights_v1",
        "experiment": "H6_SDCR_RGBSAR_DamageFormer",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "args": vars(args),
        "test_accessed": bool(getattr(args, "test_every_validation", False)),
        "test_monitor_policy": (
            "after_every_validation_log_only_never_selector"
            if getattr(args, "test_every_validation", False)
            else "not_accessed"
        ),
    }


def load_h6_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H6_SDCR_resume_v1":
        raise RuntimeError("unexpected H6 resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H6 resume variant mismatch")
    provenance = checkpoint.get("provenance", {})
    stored_freeze_mode = (
        provenance.get("sar_freeze_mode", "f0_shallow_frozen")
        if isinstance(provenance, Mapping)
        else "f0_shallow_frozen"
    )
    active_freeze_mode = resolve_sar_freeze_mode()
    if stored_freeze_mode != active_freeze_mode:
        raise RuntimeError(
            "H6 resume SAR freeze-mode mismatch: "
            f"checkpoint={stored_freeze_mode!r}, active={active_freeze_mode!r}"
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


def rewrite_summary_identity() -> None:
    try:
        output_dir = Path(sys.argv[sys.argv.index("--output_dir") + 1])
    except (ValueError, IndexError):
        return
    path = output_dir / "training_summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update(
        {
            "protocol": "H6_SDCR_training_summary_v1",
            "experiment": "H6_SDCR_RGBSAR_DamageFormer",
            "prediction_unit": "pixel",
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "sar_freeze_mode": resolve_sar_freeze_mode(),
            "test_accessed": bool(summary.get("test_accessed", False)),
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    event_balance_power = float(os.environ.get("H6_EVENT_BALANCE_POWER", "0"))
    if not 0.0 <= event_balance_power <= 1.0:
        raise ValueError("H6_EVENT_BALANCE_POWER must lie in [0, 1]")
    sar_freeze_mode = resolve_sar_freeze_mode()
    h4_train.VALID_H4_VARIANTS = VALID_H6_VARIANTS
    h4_train.H4_VARIANTS = H6_VARIANTS
    h4_train.build_h4_damageformer = build_h6_sdcr_damageformer
    h4_train.load_h3_source = load_h4_a4_source
    h4_train.load_optional_pretraining = load_optional_p3
    h4_train.configure_freeze_policy = configure_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_frozen_modules_in_eval
    h4_train.evaluate = evaluate_h6
    h4_train.NEW_MODULE_TOKENS = H6_NEW_MODULE_TOKENS
    h4_train.TRAIN_SAMPLER_FACTORY = (
        (
            lambda dataset, generator: tempered_event_balanced_sampler(
                dataset, generator, power=event_balance_power
            )
        )
        if event_balance_power > 0.0
        else None
    )
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h6_resume
    print(
        "H6 contract: H4-A4 control; P3 RGB anchor is direct weight inheritance "
        "(no teacher); A2/A3 estimate sensor nuisance only at stages 3-4; "
        "B0/B1/B2/B1R retain the complete frozen P3 inference anchor and inject "
        "native-SAR evidence through an exactly-zero class residual router. "
        "B1R freezes the complete native expert and trains only a low-open-rate, "
        "class-conditional router. "
        f"Event-balance power={event_balance_power:.2f}; "
        f"SAR-freeze mode={sar_freeze_mode}; "
        "Test monitoring is never a checkpoint selector.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary_identity()


if __name__ == "__main__":
    main()
