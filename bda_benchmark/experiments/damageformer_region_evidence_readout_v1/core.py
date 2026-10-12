from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import (
    atomic_torch_save, augment_metrics, event_name, extract_state,
    migrate_original_damageformer_names, parse_batch, seed_everything,
    seed_worker, sha256_file, write_json,
)


def load_frozen_sources(model: torch.nn.Module, a1_path: str, stage1_path: str,
                        minimum_stage1_step: int = 3000,
                        expected_stage1_sha256: str = "") -> Dict[str, object]:
    a1_payload = torch.load(a1_path, map_location="cpu", weights_only=False)
    state, migration = migrate_original_damageformer_names(extract_state(a1_payload))
    missing, unexpected = model.load_state_dict(state, strict=False)
    forbidden = [k for k in missing if not k.startswith(("biscp.", "readout."))]
    if forbidden:
        raise RuntimeError(f"A1 reproduction failed; missing={forbidden[:12]}")

    stage1_payload = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if stage1_payload.get("stage") != "change":
        raise RuntimeError("region evidence checkpoint must be V2 Stage1/change")
    if stage1_payload.get("variant") != "p_sc_residual":
        raise RuntimeError("region evidence checkpoint must use p_sc_residual")
    step = int(stage1_payload.get("step", -1))
    if step < minimum_stage1_step:
        raise RuntimeError(f"stale Stage1 checkpoint: step={step}, require >= {minimum_stage1_step}")
    stage1_sha = sha256_file(stage1_path)
    if expected_stage1_sha256 and stage1_sha != expected_stage1_sha256:
        raise RuntimeError("Stage1 SHA256 differs from the locked checkpoint")
    model.biscp.load_state_dict(stage1_payload["plugin_model"], strict=True)
    return {
        "a1": {"path": os.path.abspath(a1_path), "sha256": sha256_file(a1_path),
               "step": int(a1_payload.get("step", -1)), "migration_count": len(migration),
               "unexpected_keys": list(unexpected)},
        "stage1": {"path": os.path.abspath(stage1_path), "sha256": stage1_sha,
                   "step": step, "stage": "change", "variant": "p_sc_residual"},
    }


def load_readout(model: torch.nn.Module, path: str) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1":
        raise RuntimeError("not a Region Evidence Readout V1 checkpoint")
    if payload.get("variant") != model.readout_variant:
        raise RuntimeError("readout variant mismatch")
    model.readout.load_state_dict(payload["readout_model"], strict=True)
    return payload


def verify_checkpoint_sources(payload: Mapping[str, object], current: Mapping[str, object]) -> None:
    saved = payload.get("sources", {})
    for key in ("a1", "stage1"):
        if saved.get(key, {}).get("sha256") != current.get(key, {}).get("sha256"):
            raise RuntimeError(f"{key} source SHA256 differs from the readout training source")


__all__ = ["atomic_torch_save", "augment_metrics", "event_name", "load_frozen_sources",
           "load_readout", "verify_checkpoint_sources", "parse_batch", "seed_everything", "seed_worker",
           "sha256_file", "write_json"]
