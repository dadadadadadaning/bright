# -*- coding: utf-8 -*-
"""Train H5 building-instance-query RGB/SAR DamageFormer variants.

The mature H4 training/evaluation protocol is reused deliberately.  This file
only replaces the model factory, A4 checkpoint migration, new-module warmup
tokens, and checkpoint identity.  No additional supervised loss is introduced.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H5_BIQ import (  # noqa: E402
    H5_VARIANTS,
    VALID_H5_VARIANTS,
    build_h5_biq_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)


H5_NEW_MODULE_TOKENS = (
    "instance_query_head",
    "instance_sparse_fusions",
)


def _h5_provenance(provenance: Mapping[str, object]) -> Dict[str, object]:
    value = dict(provenance)
    if "h3_source" in value and "h4_a4_source" not in value:
        value["h4_a4_source"] = value.pop("h3_source")
    return value


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_h4_a4_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a4":
        raise RuntimeError(
            "H5 source must be H4-A4, "
            f"received variant={checkpoint.get('variant')!r}"
        )
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("H4-A4 checkpoint has no model state dictionary")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = [
        key
        for key in missing
        if any(token in key for token in H5_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    if unexpected or illegal_missing or (require_exact and missing):
        raise RuntimeError(
            "H4-A4-to-H5 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": _file_sha256(path),
        "protocol": checkpoint.get("protocol"),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "source_selector": checkpoint.get("selector"),
        "missing_new_keys": sorted(allowed_missing),
        "unexpected_keys": sorted(unexpected),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
    }


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
        "protocol": "H5_BIQ_resume_v1",
        "experiment": "H5_BuildingInstanceQuery_RGBSAR_DamageFormer",
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "selection_records": {
            key: dict(value) for key, value in selection_records.items()
        },
        "provenance": _h5_provenance(provenance),
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
        "protocol": "H5_BIQ_best_weights_v1",
        "experiment": "H5_BuildingInstanceQuery_RGBSAR_DamageFormer",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": _h5_provenance(provenance),
        "args": vars(args),
    }


def load_h5_resume(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    variant: str,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H5_BIQ_resume_v1":
        raise RuntimeError("unexpected H5 BIQ resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H5 BIQ resume variant mismatch")
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
        _h5_provenance(checkpoint.get("provenance", {})),
    )


def _rewrite_summary_identity() -> None:
    try:
        output_index = sys.argv.index("--output_dir") + 1
        output_dir = Path(sys.argv[output_index])
    except (ValueError, IndexError):
        return
    path = output_dir / "training_summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["protocol"] = "H5_BIQ_training_summary_v1"
    summary["experiment"] = "H5_BuildingInstanceQuery_RGBSAR_DamageFormer"
    summary["extra_supervised_losses"] = False
    summary["provenance"] = _h5_provenance(
        summary.get("provenance", {})
    )
    path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # The H4 driver resolves these names at call time.  Rebinding them keeps
    # the proven data, metric and checkpoint-selection pipeline unchanged.
    h4_train.VALID_H4_VARIANTS = VALID_H5_VARIANTS
    h4_train.H4_VARIANTS = H5_VARIANTS
    h4_train.build_h4_damageformer = build_h5_biq_damageformer
    h4_train.load_h3_source = load_h4_a4_source
    h4_train.NEW_MODULE_TOKENS = H5_NEW_MODULE_TOKENS
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h5_resume
    print(
        "H5 contract: one query per frozen-RGB building component; "
        "Q2 sparse deformable fusion only at stages 3-4; "
        "no additional supervised loss.",
        flush=True,
    )
    h4_train.main()
    _rewrite_summary_identity()


if __name__ == "__main__":
    main()
