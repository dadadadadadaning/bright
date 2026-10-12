"""Checkpoint and reproducibility contracts for the isolated experiment."""

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
    forbidden = [key for key in missing if not key.startswith("cmsp.")]
    if forbidden:
        raise RuntimeError(
            "checkpoint does not reproduce the A1 parent; "
            f"first forbidden missing keys={forbidden[:12]}"
        )
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "cmsp_missing_keys": len(missing),
        "unexpected_keys": list(unexpected),
        "migration_count": len(migration),
    }


def load_experiment_checkpoint(model: torch.nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(extract_state(payload), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint load failed: missing={missing}, unexpected={unexpected}")
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
    }


__all__ = [
    "atomic_torch_save", "augment_metrics", "event_name", "extract_state", "load_a1_parent",
    "load_experiment_checkpoint", "parse_batch", "seed_everything", "seed_worker",
    "sha256_file", "write_json",
]
