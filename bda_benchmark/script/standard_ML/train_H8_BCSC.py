# -*- coding: utf-8 -*-
"""Train H8-BCSC through the validated H4/H6 data and metric pipeline."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H8_BCSC import (  # noqa: E402
    H8_VARIANTS,
    VALID_H8_VARIANTS,
    build_h8_bcsc_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    configure_freeze_policy,
    evaluate_h6,
    keep_frozen_modules_in_eval,
)


H8_NEW_MODULE_TOKENS = (
    "prototype_blend",
    "reliability_gate",
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


def load_locked_a2_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(
            "H8 source must be the locked H6-A2 checkpoint; "
            f"received variant={checkpoint.get('variant')!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = [
        key
        for key in missing
        if any(token in key for token in H8_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "c0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "H6-A2-to-H8 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "source_selector": checkpoint.get("selector"),
        "missing_new_keys": sorted(allowed_missing),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
    }


def no_optional_pretraining(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    del model, args, provenance


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
        "protocol": "H8_BCSC_resume_v1",
        "experiment": "H8_BCSC_RGBSAR_DamageFormer",
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
        "test_accessed": False,
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
        "protocol": "H8_BCSC_best_weights_v1",
        "experiment": "H8_BCSC_RGBSAR_DamageFormer",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "args": vars(args),
        "test_accessed": False,
    }


def load_h8_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H8_BCSC_resume_v1":
        raise RuntimeError("unexpected H8 resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H8 resume variant mismatch")
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
            "protocol": "H8_BCSC_training_summary_v1",
            "experiment": "H8_BCSC_RGBSAR_DamageFormer",
            "prediction_unit": "pixel",
            "teacher": False,
            "distillation": False,
            "extra_supervised_losses": False,
            "test_accessed": False,
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    h4_train.VALID_H4_VARIANTS = VALID_H8_VARIANTS
    h4_train.H4_VARIANTS = H8_VARIANTS
    h4_train.build_h4_damageformer = build_h8_bcsc_damageformer
    h4_train.load_h3_source = load_locked_a2_source
    h4_train.load_optional_pretraining = no_optional_pretraining
    h4_train.configure_freeze_policy = configure_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_frozen_modules_in_eval
    h4_train.evaluate = evaluate_h6
    h4_train.NEW_MODULE_TOKENS = H8_NEW_MODULE_TOKENS
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h8_resume
    print(
        "H8 contract: C0 is the exact locked H6-A2 control; C1 refines the "
        "stage3/4 sensor prototype using detached high-confidence intact "
        "building evidence; C2 bounds that refinement with a low-open spatial "
        "reliability gate. RGB/localisation/BN remain frozen. No teacher, KD, "
        "ground-truth prototype, or additional supervised loss is used.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary_identity()


if __name__ == "__main__":
    main()
