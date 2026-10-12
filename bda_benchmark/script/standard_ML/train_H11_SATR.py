# -*- coding: utf-8 -*-
"""Controlled trainer for H11 safe SAR representation transfer."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H11_SATR import (  # noqa: E402
    H11_VARIANTS,
    VALID_H11_VARIANTS,
    SARMAEResidualPyramid,
    build_h11_satr_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    metrics_line,
)


H11_NEW_MODULE_TOKENS = (
    "transfer_adapters",
    "transfer_transition",
)
H11_MIGRATION_TOKENS = H11_NEW_MODULE_TOKENS + ("sarmae_encoder",)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    value = checkpoint.get(
        "model",
        checkpoint.get("state_dict", checkpoint.get("encoder", checkpoint)),
    )
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint does not contain a state dictionary")
    return value  # type: ignore[return-value]


def load_locked_h11_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    protocol = str(checkpoint.get("protocol", ""))
    source_variant = str(checkpoint.get("variant", "")).lower()
    if source_variant != "a2" or not protocol.startswith("H6_SDCR_"):
        raise RuntimeError(
            "H11 requires the validation-selected H6-A2 checkpoint; "
            f"received protocol={protocol!r}, variant={source_variant!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key for key in missing if any(token in key for token in H11_MIGRATION_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "t0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "controlled H6-A2-to-H11 migration failed: "
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
        "missing_transfer_keys": sorted(allowed_missing),
    }


def _strip_encoder_prefix(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "base_encoder.",
        "encoder.",
        # Official SARMAE joint-pretraining checkpoints are saved from
        # SAROpticalPatchAlignment, whose MAE branch is registered as
        # ``sar_encoder``.  Keep this distinct from our downstream
        # ``sarmae_encoder`` module name.
        "sar_encoder.",
        "sarmae_encoder.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _resize_position_embedding(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if source.shape == target.shape:
        return source
    if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != target.shape[-1]:
        raise RuntimeError(
            f"cannot adapt SARMAE position embedding {tuple(source.shape)} "
            f"to {tuple(target.shape)}"
        )
    source_patch = source[:, 1:]
    target_patch = target[:, 1:]
    source_side = int(round(source_patch.shape[1] ** 0.5))
    target_side = int(round(target_patch.shape[1] ** 0.5))
    if (
        source_side * source_side != source_patch.shape[1]
        or target_side * target_side != target_patch.shape[1]
    ):
        raise RuntimeError("SARMAE position embeddings must use square patch grids")
    resized = source_patch.reshape(
        1, source_side, source_side, source.shape[-1]
    ).permute(0, 3, 1, 2)
    resized = F.interpolate(
        resized.float(),
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).reshape(1, target_side * target_side, source.shape[-1])
    return torch.cat([source[:, :1].float(), resized], dim=1).to(target.dtype)


def migrate_sarmae_encoder_state(
    encoder: nn.Module,
    checkpoint: Mapping[str, object],
) -> Dict[str, object]:
    """Migrate a common MAE/SARMAE ViT-B encoder without its decoder/head."""

    source = _state(checkpoint)
    target = encoder.state_dict()
    selected: Dict[str, torch.Tensor] = {}
    ignored = []
    for raw_key, value in source.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = _strip_encoder_prefix(str(raw_key))
        if key not in target:
            ignored.append(str(raw_key))
            continue
        if key == "patch_embed.proj.weight" and value.shape != target[key].shape:
            if (
                value.ndim == 4
                and value.shape[0] == target[key].shape[0]
                and value.shape[2:] == target[key].shape[2:]
                and target[key].shape[1] == 1
            ):
                value = value.float().mean(dim=1, keepdim=True).to(target[key].dtype)
        if key == "pos_embed" and value.shape != target[key].shape:
            value = _resize_position_embedding(value, target[key])
        if value.shape == target[key].shape:
            selected[key] = value.to(dtype=target[key].dtype)

    block_ids = {
        int(key.split(".")[1])
        for key in selected
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    required_blocks = set(range(len(getattr(encoder, "blocks"))))
    essential = {
        "patch_embed.proj.weight",
        "cls_token",
        "pos_embed",
        "norm.weight",
        "norm.bias",
    }
    missing_essential = sorted(essential - set(selected))
    if block_ids != required_blocks or missing_essential:
        raise RuntimeError(
            "SARMAE checkpoint is not a compatible full encoder: "
            f"blocks={sorted(block_ids)}, missing={missing_essential}"
        )
    incompatible = encoder.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"SARMAE migration produced unexpected keys: {incompatible.unexpected_keys}"
        )
    # Bias-free patch embeddings are accepted, but every transformer and norm
    # parameter must have been provided by the external checkpoint.
    illegal_missing = [
        key
        for key in incompatible.missing_keys
        if key != "patch_embed.proj.bias"
    ]
    if illegal_missing:
        raise RuntimeError(f"SARMAE migration is incomplete: {illegal_missing}")
    return {
        "loaded_keys": len(selected),
        "ignored_decoder_or_head_keys": len(ignored),
        "encoder_blocks": len(block_ids),
        "patch_input_channels": int(target["patch_embed.proj.weight"].shape[1]),
        "strict_encoder_coverage": True,
    }


def load_optional_sarmae(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    if getattr(model, "variant", "") == "t0":
        return
    if not args.sar_pretrained_checkpoint:
        raise RuntimeError(
            "H11-T1/T2 require --sar_pretrained_checkpoint with SARMAE encoder weights"
        )
    if not isinstance(model.sar_encoder, SARMAEResidualPyramid):
        raise RuntimeError("H11 transfer variant did not construct the SARMAE pathway")
    checkpoint = torch.load(
        args.sar_pretrained_checkpoint, map_location="cpu", weights_only=False
    )
    report = migrate_sarmae_encoder_state(
        model.sar_encoder.sarmae_encoder, checkpoint
    )
    provenance["sarmae_transfer"] = {
        "path": str(Path(args.sar_pretrained_checkpoint).resolve()),
        "sha256": file_sha256(args.sar_pretrained_checkpoint),
        "source_protocol": checkpoint.get("protocol"),
        "source_epoch": checkpoint.get("epoch"),
        "source_test_accessed": bool(checkpoint.get("test_accessed", False)),
        **report,
    }


def configure_h11_freeze_policy(model: nn.Module) -> None:
    """Freeze H6-A2; train transfer adapters and optionally MAE blocks 10-11."""

    variant = str(getattr(model, "variant", "")).lower()
    for name, parameter in model.named_parameters():
        trainable = any(token in name for token in H11_NEW_MODULE_TOKENS)
        if variant in ("t2", "t2e"):
            trainable = trainable or name.startswith(
                ("sar_encoder.sarmae_encoder.blocks.10", "sar_encoder.sarmae_encoder.blocks.11")
            ) or name.startswith("sar_encoder.sarmae_encoder.norm")
        if variant == "t0":
            # Validation-only T0 still needs one parameter for the shared
            # trainer's optimizer construction.
            trainable = name == "hierarchy_transition"
        parameter.requires_grad_(trainable)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_h11_frozen_in_eval(model: nn.Module) -> None:
    """Keep the inherited expert frozen while preserving trainable H11 modules."""

    model.eval()
    if not isinstance(model.sar_encoder, SARMAEResidualPyramid):
        return
    model.sar_encoder.transfer_adapters.train()
    variant = str(getattr(model, "variant", "")).lower()
    if variant in ("t2", "t2e"):
        model.sar_encoder.sarmae_encoder.blocks[10].train()
        model.sar_encoder.sarmae_encoder.blocks[11].train()
        model.sar_encoder.sarmae_encoder.norm.train()


def evaluate_h11_and_restore(
    model,
    loader,
    device,
    ignore_index: int,
    max_batches: int = 0,
):
    metrics = evaluate_h10(model, loader, device, ignore_index, max_batches)
    keep_h11_frozen_in_eval(model)
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
        "protocol": "H11_SATR_resume_v1",
        "experiment": "H11_SATR_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "selection_records": {key: dict(value) for key, value in selection_records.items()},
        "provenance": dict(provenance),
        "objective": "unchanged_h6_a2_supervised_objective",
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
        "protocol": "H11_SATR_best_weights_v1",
        "experiment": "H11_SATR_RGBSAR_DamageAssessment",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "objective": "unchanged_h6_a2_supervised_objective",
        "teacher": False,
        "distillation": False,
        "test_accessed": False,
        "args": vars(args),
    }


def load_h11_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H11_SATR_resume_v1":
        raise RuntimeError("unexpected H11-SATR resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H11-SATR resume variant mismatch")
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
    try:
        variant = str(sys.argv[sys.argv.index("--variant") + 1]).lower()
    except (ValueError, IndexError):
        variant = str(summary.get("variant", "")).lower()
    exact_input = variant in ("t1e", "t2e")
    summary.update(
        {
            "protocol": "H11_SATR_training_summary_v1",
            "experiment": "H11_SATR_RGBSAR_DamageAssessment",
            "base_expert": "locked_h6_a2",
            "rgb_localization": "frozen_p3",
            "transfer_stages": [3, 4],
            "sarmae_input_contract": (
                "official_3ch_imagenet_photometric"
                if exact_input
                else "naive_1ch_train_stats"
            ),
            "sarmae_patch_embed_channels": 3 if exact_input else 1,
            "dataset_variant": "h4" if exact_input else "h3",
            "objective": "unchanged_h6_a2_supervised_objective",
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "test_accessed": False,
            "test_metrics_are_selector": False,
        }
    )
    initial_path = output_dir / "initial_validation.json"
    initial = (
        json.loads(initial_path.read_text(encoding="utf-8"))
        if initial_path.is_file()
        else {}
    )
    records = summary.get("selection_records", {})
    candidate_files = {
        "valid": "best_val_miou.pth",
        "event": "best_event.pth",
        "balance": "best_balance.pth",
    }
    gate_records = []
    for selector, filename in candidate_files.items():
        record = records.get(selector, {}) if isinstance(records, Mapping) else {}
        metrics = record.get("metrics", {}) if isinstance(record, Mapping) else {}
        checks = {}
        if isinstance(metrics, Mapping) and isinstance(initial, Mapping):
            checks = {
                "valid_gain_at_least_0p10pp": float(metrics.get("valid_miou", -1.0))
                >= float(initial.get("valid_miou", 0.0)) + 0.001,
                "damaged_not_lower": float(metrics.get("damaged_iou", -1.0))
                >= float(initial.get("damaged_iou", 0.0)),
                "affected_recall_not_lower": float(metrics.get("affected_recall", -1.0))
                >= float(initial.get("affected_recall", 0.0)),
                "damaged_gate_miss_not_higher": float(
                    metrics.get("damaged_gate_miss", 2.0)
                )
                <= float(initial.get("damaged_gate_miss", 1.0)),
                "event_mean_not_lower": float(
                    metrics.get("event_mean_official_style_4class_miou", -1.0)
                )
                >= float(initial.get("event_mean_official_style_4class_miou", 0.0)),
                "event_worst_not_lower": float(
                    metrics.get("event_worst_official_style_4class_miou", -1.0)
                )
                >= float(initial.get("event_worst_official_style_4class_miou", 0.0)),
            }
        gate_records.append(
            {
                "selector": selector,
                "checkpoint": filename,
                "step": record.get("step") if isinstance(record, Mapping) else None,
                "checks": checks,
                "pass": bool(checks) and all(checks.values()),
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
            }
        )
    passing = [record for record in gate_records if record["pass"]]
    recommended = (
        max(
            passing,
            key=lambda record: float(record["metrics"].get("valid_miou", -1.0)),
        )
        if passing
        else None
    )
    gate_payload = {
        "protocol": "H11_SATR_validation_gate_v1",
        "test_accessed": False,
        "baseline": dict(initial) if isinstance(initial, Mapping) else {},
        "candidates": gate_records,
        "decision": "PROCEED" if recommended is not None else "STOP",
        "recommended_selector": (
            recommended["selector"] if recommended is not None else None
        ),
        "recommended_checkpoint": (
            recommended["checkpoint"] if recommended is not None else None
        ),
    }
    summary["validation_gate"] = gate_payload
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "validation_gate.json").write_text(
        json.dumps(gate_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "H11 validation gate: "
        f"{gate_payload['decision']} | recommended="
        f"{gate_payload['recommended_checkpoint']}",
        flush=True,
    )


def main() -> None:
    h4_train.VALID_H4_VARIANTS = VALID_H11_VARIANTS
    h4_train.H4_VARIANTS = H11_VARIANTS
    h4_train.build_h4_damageformer = build_h11_satr_damageformer
    h4_train.load_h3_source = load_locked_h11_source
    h4_train.load_optional_pretraining = load_optional_sarmae
    h4_train.configure_freeze_policy = configure_h11_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_h11_frozen_in_eval
    h4_train.evaluate = evaluate_h11_and_restore
    h4_train.metrics_line = metrics_line
    h4_train.NEW_MODULE_TOKENS = H11_NEW_MODULE_TOKENS
    h4_train.EXTRA_SUPERVISED_LOSS = None
    h4_train.EXTRA_SUPERVISED_USES_NAMES = False
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.SELECT_INITIAL_CHECKPOINT = True
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h11_resume
    print(
        "H11 contract: T0 is exact H6-A2; T1/T2 retain the completed naive "
        "one-channel ablation; T1E/T2E preserve the official three-channel "
        "SARMAE PatchEmbed and reproduce grayscale-repeat plus ImageNet "
        "photometric normalization from the raw display SAR channel. T1/T1E "
        "freeze the complete SARMAE encoder and train only zero-initialised "
        "Stage3/4 transfer residuals; T2/T2E additionally fine-tune only "
        "SARMAE blocks 10-11 after warmup. "
        "P3 RGB localisation, the complete native A2 expert, fusion and heads "
        "remain frozen. The original H6-A2 supervised objective is unchanged; "
        "no teacher, KD, severity head, prototype, contrastive or alignment "
        "loss is added. Test is never used for model selection.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary()


if __name__ == "__main__":
    main()
