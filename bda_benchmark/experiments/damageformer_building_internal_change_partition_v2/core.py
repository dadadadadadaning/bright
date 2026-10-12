"""Checkpoint/reproducibility helpers for V2."""
from __future__ import annotations

import os
from typing import Dict

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import (
    atomic_torch_save,
    augment_metrics,
    event_name,
    extract_state,
    migrate_original_damageformer_names,
    parse_batch,
    seed_everything,
    seed_worker,
    sha256_file,
    write_json,
)


def load_a1_parent(model: torch.nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_state(payload)
    state, migration = migrate_original_damageformer_names(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    forbidden = [key for key in missing if not key.startswith("biscp.")]
    if forbidden:
        raise RuntimeError(
            "checkpoint does not reproduce the A1 parent; "
            f"first forbidden missing keys={forbidden[:12]}"
        )
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "biscp_missing_keys": len(missing),
        "unexpected_keys": list(unexpected),
        "migration_count": len(migration),
    }


def load_v2_checkpoint(model: torch.nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "plugin_model" in payload:
        missing, unexpected = model.biscp.load_state_dict(payload["plugin_model"], strict=True)
    else:
        state = extract_state(payload)
        missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict V2 load failed: missing={missing}, unexpected={unexpected}")
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "stage": payload.get("stage", "") if isinstance(payload, dict) else "",
        "variant": payload.get("variant", "") if isinstance(payload, dict) else "",
    }


__all__ = [
    "atomic_torch_save",
    "augment_metrics",
    "event_name",
    "extract_state",
    "load_a1_parent",
    "load_v2_checkpoint",
    "parse_batch",
    "seed_everything",
    "seed_worker",
    "sha256_file",
    "write_json",
]
