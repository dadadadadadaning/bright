#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the identity and stored provenance of the two S8 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


HISTORICAL_P3_STEP = 8000
HISTORICAL_P3_MIOU4_PERCENT = 69.7612


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect S8 checkpoint identity")
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--ia_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint has no state-dict mapping: {type(checkpoint)!r}")
    return {
        (key[7:] if isinstance(key, str) and key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def first_number(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def as_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * value if abs(value) <= 1.0 else value


def identify_architecture(keys: set[str]) -> str:
    has_micb = any(key.startswith("micb_dsmil.") for key in keys)
    has_hcdm = any(key.startswith("hcdm.") for key in keys)
    has_ia = any(
        key.startswith(("ia_stage3.", "ia_stage4."))
        for key in keys
    )
    if has_micb and not (has_hcdm or has_ia):
        return "DamageFormer_S8_MICB_DSMIL"
    if has_hcdm and has_ia and not has_micb:
        return "DamageFormer_S8_IA_MLA"
    return "unknown_or_mixed"


def inspect_checkpoint(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    keys = set(str(key) for key in state)
    metadata = checkpoint if isinstance(checkpoint, Mapping) else {}
    metrics = json_mapping(metadata.get("metrics"))
    args = json_mapping(metadata.get("args"))
    architecture = identify_architecture(keys)
    if architecture == "DamageFormer_S8_IA_MLA":
        has_legacy_relation = any(
            key.startswith("hcdm.stage_blocks.") and ".relation." in key
            for key in keys
        )
        has_current_relation_fuse = any(
            key.startswith("hcdm.stage_blocks.") and ".relation_fuse." in key
            for key in keys
        )
        if has_legacy_relation and not has_current_relation_fuse:
            state_schema = "legacy_relation"
        elif has_current_relation_fuse and not has_legacy_relation:
            state_schema = "current_relation_fuse"
        else:
            state_schema = "mixed_or_unknown_relation"
    else:
        state_schema = "native"
    miou4 = first_number(
        metrics,
        ("miou4", "official_style_4class_miou", "final_miou", "miou"),
    )
    valid_miou = first_number(
        metrics,
        ("valid_miou", "target_3class_miou"),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "architecture": architecture,
        "state_schema": state_schema,
        "experiment": metadata.get("experiment"),
        "step": metadata.get("step"),
        "state_key_count": len(state),
        "stored_miou4_percent": as_percent(miou4),
        "stored_valid_miou_percent": as_percent(valid_miou),
        "best_val_miou_percent": as_percent(
            first_number(metadata, ("best_val_miou", "best_valid_miou"))
        ),
        "selected_args": {
            key: args.get(key)
            for key in (
                "roi_alpha",
                "token_dim",
                "mil_topk",
                "max_tokens_per_bag",
                "mil_residual_scale",
                "use_asymmetric_regions",
                "diff_channels",
                "align_dim",
                "gamma_init",
                "seed",
            )
            if key in args
        },
    }


def main() -> None:
    args = parse_args()
    p3 = inspect_checkpoint(args.p3_checkpoint)
    ia = inspect_checkpoint(args.ia_checkpoint)

    if p3["architecture"] != "DamageFormer_S8_MICB_DSMIL":
        raise RuntimeError(
            "The first checkpoint is not a pure MICB-DSMIL P3 state dict: "
            f"{p3['architecture']}"
        )
    if ia["architecture"] != "DamageFormer_S8_IA_MLA":
        raise RuntimeError(
            "The second checkpoint is not an IA-MLA state dict: "
            f"{ia['architecture']}"
        )

    p3_step = p3.get("step")
    p3_miou4 = p3.get("stored_miou4_percent")
    p3_matches_historical = (
        p3_step == HISTORICAL_P3_STEP
        and p3_miou4 is not None
        and abs(float(p3_miou4) - HISTORICAL_P3_MIOU4_PERCENT) <= 1e-4
    )
    report = {
        "p3": p3,
        "ia_mla": ia,
        "historical_p3_record": {
            "step": HISTORICAL_P3_STEP,
            "miou4_percent": HISTORICAL_P3_MIOU4_PERCENT,
            "current_file_matches": p3_matches_historical,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("Checkpoint identity audit")
    print("=" * 72)
    print(
        "P3 current file : "
        f"arch={p3['architecture']} | schema={p3['state_schema']} | "
        f"experiment={p3['experiment']} | "
        f"step={p3['step']} | stored_mIoU4={p3['stored_miou4_percent']} | "
        f"sha256={p3['sha256']}"
    )
    print(
        "IA-MLA file     : "
        f"arch={ia['architecture']} | schema={ia['state_schema']} | "
        f"experiment={ia['experiment']} | "
        f"step={ia['step']} | stored_mIoU4={ia['stored_miou4_percent']} | "
        f"sha256={ia['sha256']}"
    )
    print(
        "Historical P3  : "
        f"step={HISTORICAL_P3_STEP} | mIoU4={HISTORICAL_P3_MIOU4_PERCENT} | "
        f"current_file_matches={p3_matches_historical}"
    )
    print(f"Saved audit JSON: {output_path}")


if __name__ == "__main__":
    main()
