# -*- coding: utf-8 -*-
"""Train controlled H5-B sparse fusion and H5-C pretrained-adapter variants."""
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

from bda_benchmark.model.DamageFormer_RGBSAR_H5_BC import (  # noqa: E402
    H5_BC_VARIANTS,
    VALID_H5_BC_VARIANTS,
    build_h5_bc_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)


H5_BC_NEW_MODULE_TOKENS = (
    "building_sparse_fusions",
    "cross_modal_adapters",
)


def file_sha256(path: str) -> str:
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
            "H5-B/C source must be H4-A4, "
            f"received variant={checkpoint.get('variant')!r}"
        )
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("H4-A4 checkpoint has no model state dictionary")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = [
        key
        for key in missing
        if any(token in key for token in H5_BC_NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") == "b0")
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "H4-A4-to-H5-B/C migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
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


def load_joint_pretraining(
    model: nn.Module,
    path: str,
    source: Mapping[str, object],
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H5_C_joint_adapters_v1":
        raise RuntimeError(
            "unexpected H5-C joint-pretraining protocol: "
            f"{checkpoint.get('protocol')!r}"
        )
    checkpoint_source = checkpoint.get("source_a4", {})
    if (
        not isinstance(checkpoint_source, Mapping)
        or checkpoint_source.get("sha256") != source.get("sha256")
    ):
        raise RuntimeError(
            "H5-C pretraining and supervised fine-tuning use different A4 sources"
        )
    state = checkpoint.get("cross_modal_adapters")
    if not isinstance(state, Mapping):
        raise RuntimeError("H5-C checkpoint has no cross_modal_adapters state")
    if not hasattr(model, "cross_modal_adapters"):
        raise RuntimeError("current model does not declare C1 adapters")
    model.cross_modal_adapters.load_state_dict(state, strict=True)
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": checkpoint.get("protocol"),
        "step": int(checkpoint.get("step", 0)),
        "source_a4": dict(checkpoint_source),
        "pretraining_contract": checkpoint.get("pretraining_contract", {}),
    }


def load_optional_pretraining(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    if args.variant != "c1":
        return
    if not args.joint_pretrained_checkpoint:
        raise RuntimeError("C1 requires --joint_pretrained_checkpoint")
    provenance["joint_pretraining"] = load_joint_pretraining(
        model,
        args.joint_pretrained_checkpoint,
        provenance["h3_source"],
    )


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
        "protocol": "H5_BC_resume_v1",
        "experiment": "H5_BuildingSparse_JointPretrain_RGBSAR",
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
        "protocol": "H5_BC_best_weights_v1",
        "experiment": "H5_BuildingSparse_JointPretrain_RGBSAR",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "args": vars(args),
    }


def load_h5_bc_resume(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    variant: str,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H5_BC_resume_v1":
        raise RuntimeError("unexpected H5-B/C resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H5-B/C resume variant mismatch")
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
        output_index = sys.argv.index("--output_dir") + 1
        output_dir = Path(sys.argv[output_index])
    except (ValueError, IndexError):
        return
    path = output_dir / "training_summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["protocol"] = "H5_BC_training_summary_v1"
    summary["experiment"] = "H5_BuildingSparse_JointPretrain_RGBSAR"
    summary["hard_instance_decode"] = False
    summary["extra_supervised_losses"] = False
    path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    h4_train.VALID_H4_VARIANTS = VALID_H5_BC_VARIANTS
    h4_train.H4_VARIANTS = H5_BC_VARIANTS
    h4_train.build_h4_damageformer = build_h5_bc_damageformer
    h4_train.load_h3_source = load_h4_a4_source
    h4_train.load_optional_pretraining = load_optional_pretraining
    h4_train.NEW_MODULE_TOKENS = H5_BC_NEW_MODULE_TOKENS
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h5_bc_resume
    print(
        "H5-B/C contract: B1 uses building-restricted K-point sampling at "
        "stages 3-4; C1 loads train-only joint-pretrained symmetric adapters; "
        "standard A4 pixel decoder; no hard instance decode; no new supervised loss.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary_identity()


if __name__ == "__main__":
    main()
