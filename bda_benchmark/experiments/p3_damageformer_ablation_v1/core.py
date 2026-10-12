#!/usr/bin/env python3
"""Shared contracts for the controlled P3 lineage ablation."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .variants import VARIANTS, VariantSpec, get_variant, variant_manifest
except ImportError:
    from variants import VARIANTS, VariantSpec, get_variant, variant_manifest


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_state(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                payload = value
                break
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint does not contain a model state mapping")
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }


def migrate_original_damageformer_names(
    state: Mapping[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Map the isolated D0 control names onto the historical P3 names.

    The modules are structurally equivalent, but the isolated SALT control
    names its two Swin wrappers ``rgb_encoder`` and ``sar_encoder`` whereas
    the historical DamageFormer/P3 files use ``encoder_1`` and ``encoder_2``.
    Shape validation is still performed by ``load_state_dict`` afterwards.
    """

    if not any(key.startswith("rgb_encoder.features.") for key in state):
        return dict(state), {}
    replacements = (
        ("rgb_encoder.features.", "encoder_1.swin_transformer.features."),
        ("sar_encoder.features.", "encoder_2.swin_transformer.features."),
        ("localization_head.", "clf_1."),
        ("damage_head.", "clf_2."),
    )
    migrated: Dict[str, torch.Tensor] = {}
    report: Dict[str, str] = {}
    for source, value in state.items():
        target = source
        for old, new in replacements:
            if source.startswith(old):
                target = new + source[len(old) :]
                break
        if target in migrated:
            raise RuntimeError(f"checkpoint key migration collision: {source} -> {target}")
        migrated[target] = value
        if target != source:
            report[source] = target
    return migrated, report


def load_parent(model: torch.nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_state(payload)
    state, name_migration = migrate_original_damageformer_names(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # A historical DamageFormer checkpoint naturally lacks both later modules;
    # the S3 prototype is inherited only for checkpoint compatibility and is
    # explicitly bypassed/frozen by P3.
    allowed_prefixes = ("micb_dsmil.", "damage_proto_memory.")
    allowed_missing = [key for key in missing if key.startswith(allowed_prefixes)]
    forbidden_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
    if forbidden_missing:
        raise RuntimeError(
            "parent checkpoint is not compatible with the P3 pixel path; "
            f"first missing keys={forbidden_missing[:12]}"
        )
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, Mapping) else -1,
        "allowed_later_module_missing_keys": len(allowed_missing),
        "unexpected_keys": list(unexpected),
        "name_migration_count": len(name_migration),
        "name_migration_examples": list(name_migration.items())[:12],
    }


def parse_batch(batch: Sequence[object], device: torch.device):
    if len(batch) < 5:
        raise ValueError("expected (pre, post, location, damage, name)")
    pre, post, location, damage, names = batch[:5]
    return (
        pre.to(device, non_blocking=True).float(),
        post.to(device, non_blocking=True).float(),
        location.to(device, non_blocking=True).long(),
        damage.to(device, non_blocking=True).long(),
        list(names),
    )


def event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def bag_losses(bag_result: Mapping[str, object] | None, reference: torch.Tensor):
    zero = reference.float().sum() * 0.0
    output = {"mil": zero, "exist": zero, "severity": zero, "guard": zero}
    if not bag_result or int(bag_result.get("bag_count", 0)) == 0:
        return output
    labels = bag_result["labels"]
    if not isinstance(labels, torch.Tensor):
        raise TypeError("bag labels must be a tensor")
    labels = labels.long()
    output["mil"] = F.nll_loss(bag_result["log_probs3"].float(), labels - 1)
    affected = (labels > 1).long()
    output["exist"] = F.cross_entropy(bag_result["exist_logits"].float(), affected)
    mask = labels > 1
    if bool(mask.any().item()):
        output["severity"] = F.cross_entropy(
            bag_result["severity_logits"].float()[mask], labels[mask] - 2
        )
    guard = bag_result["intact_guard_logits"]
    if isinstance(guard, torch.Tensor) and guard.numel():
        output["guard"] = F.binary_cross_entropy_with_logits(
            guard.float(), torch.zeros_like(guard, dtype=torch.float32)
        )
    return output


def augment_metrics(metrics: Dict[str, float], loc_hist: torch.Tensor, dam_hist: torch.Tensor):
    tp = loc_hist[1, 1].double()
    metrics["loc_precision"] = float((tp / loc_hist[:, 1].sum().clamp_min(1.0)).item())
    metrics["loc_recall"] = float((tp / loc_hist[1].sum().clamp_min(1.0)).item())
    affected_total = dam_hist[2:4].sum().clamp_min(1.0)
    metrics["affected_recall"] = float((dam_hist[2:4, 2:4].sum() / affected_total).item())
    metrics["damaged_gate_miss"] = float(
        (dam_hist[2, 0:2].sum() / dam_hist[2].sum().clamp_min(1.0)).item()
    )
    return metrics


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
