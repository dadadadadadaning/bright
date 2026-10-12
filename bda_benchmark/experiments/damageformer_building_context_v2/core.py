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
    forbidden = [
        key for key in missing
        if not (key.startswith("cmsp.") or key.startswith("bir."))
    ]
    if forbidden or unexpected:
        raise RuntimeError(
            f"A1 load mismatch: forbidden_missing={forbidden[:16]}, "
            f"unexpected={list(unexpected)[:16]}"
        )

    return {
        "kind": "a1",
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "allowed_missing": len(missing),
        "migration_count": len(migration),
    }


def load_old_c5(model: torch.nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_state(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)

    forbidden = [key for key in missing if not key.startswith("bir.")]
    if forbidden or unexpected:
        raise RuntimeError(
            f"C5 load mismatch: forbidden_missing={forbidden[:16]}, "
            f"unexpected={list(unexpected)[:16]}"
        )

    return {
        "kind": "old_c5",
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "bir_missing_keys": len(missing),
    }


__all__ = [
    "atomic_torch_save",
    "augment_metrics",
    "event_name",
    "extract_state",
    "load_a1_parent",
    "load_old_c5",
    "parse_batch",
    "seed_everything",
    "seed_worker",
    "write_json",
]
