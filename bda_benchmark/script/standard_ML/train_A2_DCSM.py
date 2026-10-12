# -*- coding: utf-8 -*-
"""Supervised fine-tuning for the controlled A2-DCSM experiments.

The validated H4/H6 data, loss, metric, validation, and checkpoint loop is
reused verbatim.  This wrapper changes only five contracts:

1. the starting point is the locked H6-A2 native-SAR expert;
2. M1-M3 must load a provenance-matched DCSM pathway checkpoint;
3. RGB/localisation/BN and the migrated shallow SAR pathway stay frozen;
4. training samples are event balanced and crops favour building damage;
5. checkpoint metadata records the migration scope explicitly.

No pretraining objective is active during supervised fine-tuning and no new
supervised loss is introduced here.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset as _BaseDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_A2_DCSM import (  # noqa: E402
    A2_DCSM_VARIANTS,
    VALID_A2_DCSM_VARIANTS,
    build_a2_dcsm_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)


PATHWAY_PREFIXES = {
    "m1": ("sar_encoder.",),
    "m2": ("structural_adapter.", "sar_encoder."),
    "m3": (
        "structural_adapter.",
        "sar_encoder.",
        "joint_stages.2.sar_common.",
        "joint_stages.2.sar_unique.",
        "joint_stages.3.sar_common.",
        "joint_stages.3.sar_unique.",
    ),
}


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
    model: nn.Module, path: str, require_exact: bool
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(
            "A2-DCSM must start from the locked H6-A2 expert; "
            f"received variant={checkpoint.get('variant')!r}"
        )
    missing, unexpected = model.load_state_dict(_state(checkpoint), strict=False)
    allowed_missing = sorted(
        key for key in missing if key.startswith("structural_adapter.")
    )
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    exact = bool(require_exact or getattr(model, "variant", "") in ("m0", "m1"))
    if unexpected or illegal_missing or (exact and missing):
        raise RuntimeError(
            "locked H6-A2 migration is not strict: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "source_selector": checkpoint.get("selector"),
        "source_metrics": dict(
            checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
        ),
        "allowed_new_keys": allowed_missing,
    }


def load_dcsm_pretraining(
    model: nn.Module, args, provenance: MutableMapping[str, object]
) -> None:
    variant = str(args.variant).lower()
    if variant == "m0":
        if args.sar_pretrained_checkpoint:
            raise RuntimeError("M0 is the locked A2 control and cannot load DCSM pretraining")
        return
    if not args.sar_pretrained_checkpoint:
        raise RuntimeError(f"{variant.upper()} requires --sar_pretrained_checkpoint")

    checkpoint = torch.load(
        args.sar_pretrained_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("protocol") != "A2_DCSM_pathway_v1":
        raise RuntimeError(
            f"unexpected DCSM pathway protocol: {checkpoint.get('protocol')!r}"
        )
    if str(checkpoint.get("variant", "")).lower() != variant:
        raise RuntimeError("DCSM pathway variant does not match supervised variant")
    source = checkpoint.get("source_a2", {})
    if not isinstance(source, Mapping):
        raise RuntimeError("DCSM pathway lacks source-A2 provenance")
    locked_source = provenance.get("h3_source", {})
    if (
        not isinstance(locked_source, Mapping)
        or source.get("sha256") != locked_source.get("sha256")
    ):
        raise RuntimeError("pretraining and supervised fine-tuning use different A2 sources")

    pathway = checkpoint.get("pathway")
    if not isinstance(pathway, Mapping) or not pathway:
        raise RuntimeError("DCSM checkpoint has no pathway state")
    prefixes = PATHWAY_PREFIXES[variant]
    destination = model.state_dict()
    expected = {
        key for key in destination if any(key.startswith(prefix) for prefix in prefixes)
    }
    supplied = set(pathway)
    if supplied != expected:
        raise RuntimeError(
            "DCSM migration scope mismatch: "
            f"missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}"
        )
    incompatible = model.load_state_dict(pathway, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"DCSM migration produced unexpected keys: {incompatible.unexpected_keys}"
        )
    provenance["dcsm_pretraining"] = {
        "path": str(Path(args.sar_pretrained_checkpoint).resolve()),
        "sha256": file_sha256(args.sar_pretrained_checkpoint),
        "protocol": checkpoint.get("protocol"),
        "variant": checkpoint.get("variant"),
        "step": int(checkpoint.get("step", 0)),
        "source_a2": dict(source),
        "migrated_scope": list(prefixes),
        "objective_active_during_supervision": False,
    }


def configure_freeze_policy(model: nn.Module) -> None:
    """Freeze spatial anchors and the migrated low-level SAR contract."""

    for name, parameter in model.named_parameters():
        permanently_frozen = (
            name.startswith("rgb_encoder")
            or name.startswith("localization_decoder")
            or name.startswith("localization_head")
            or name.startswith("p3_spatial_anchor")
            or name.startswith("p3_full_anchor")
            or name.startswith("structural_adapter")
            or name.startswith("sar_encoder.stem")
            or name.startswith("sar_encoder.stages.0")
            or name.startswith("sar_encoder.stages.1")
            or name.startswith("sar_encoder.downsamples.0")
            or name.startswith("sar_encoder.downsamples.1")
        )
        parameter.requires_grad_(not permanently_frozen)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_frozen_modules_in_eval(model: nn.Module) -> None:
    model.rgb_encoder.eval()
    model.localization_decoder.eval()
    model.localization_head.eval()
    if hasattr(model, "p3_spatial_anchor"):
        model.p3_spatial_anchor.eval()
    model.structural_adapter.eval()
    model.sar_encoder.stem.eval()
    model.sar_encoder.stages[0].eval()
    model.sar_encoder.stages[1].eval()
    model.sar_encoder.downsamples[0].eval()
    model.sar_encoder.downsamples[1].eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _event_name(sample_name: str) -> str:
    return sample_name.rsplit("_", 1)[0]


def event_balanced_sampler(dataset, generator: torch.Generator):
    names = list(dataset.data_list)
    counts = Counter(_event_name(name) for name in names)
    weights = torch.as_tensor(
        [1.0 / counts[_event_name(name)] for name in names], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights, num_samples=len(names), replacement=True, generator=generator
    )


def dcsm_dataset(*args, **kwargs):
    if str(kwargs.get("split", "train")).lower() == "train":
        kwargs.setdefault("focus_probability", 0.75)
        kwargs.setdefault("focus_class_weights", (0.0, 1.0, 2.0, 2.0))
    try:
        return _BaseDataset(*args, **kwargs)
    except TypeError as error:
        if "focus_probability" not in str(error):
            raise
        raise RuntimeError(
            "A2-DCSM dataset interface is outdated. Upload the matching "
            "bda_benchmark/dataset/make_data_loader_heterogeneous_rgbsar.py; "
            "its HeterogeneousRGBSARDataset.__init__ must declare "
            "focus_probability and focus_class_weights. The run is stopped "
            "instead of silently disabling damage-focused sampling."
        ) from error


def validate_dataset_interface() -> None:
    parameters = inspect.signature(_BaseDataset.__init__).parameters
    required = {"focus_probability", "focus_class_weights"}
    missing = sorted(required - set(parameters))
    if missing:
        raise RuntimeError(
            "A2-DCSM dataset interface mismatch; missing constructor arguments "
            f"{missing}. Upload the matching file: "
            "bda_benchmark/dataset/make_data_loader_heterogeneous_rgbsar.py"
        )


def checkpoint_payload(
    model, optimizer, scaler, step, best_metrics, selection_records,
    provenance, args,
) -> Dict[str, object]:
    return {
        "protocol": "A2_DCSM_resume_v1",
        "experiment": "A2_DCSM_NativeSAR_Expert",
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
        "test_accessed": bool(getattr(args, "test_every_validation", False)),
        "test_metrics_are_selector": False,
    }


def weights_payload(
    model, step, metrics, provenance, args, selector: str
) -> Dict[str, object]:
    return {
        "protocol": "A2_DCSM_best_weights_v1",
        "experiment": "A2_DCSM_NativeSAR_Expert",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "args": vars(args),
        "test_accessed": bool(getattr(args, "test_every_validation", False)),
        "test_metrics_are_selector": False,
    }


def load_resume(path: str, model, optimizer, scaler, variant: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "A2_DCSM_resume_v1":
        raise RuntimeError("unexpected A2-DCSM resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("A2-DCSM resume variant mismatch")
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
            "protocol": "A2_DCSM_training_summary_v1",
            "experiment": "A2_DCSM_NativeSAR_Expert",
            "prediction_unit": "pixel",
            "teacher": False,
            "distillation": False,
            "pretraining_loss_active_during_supervision": False,
            "event_balanced_sampling": True,
            "damage_focused_crops": True,
            "test_metrics_are_selector": False,
        }
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    validate_dataset_interface()
    h4_train.VALID_H4_VARIANTS = VALID_A2_DCSM_VARIANTS
    h4_train.H4_VARIANTS = A2_DCSM_VARIANTS
    h4_train.build_h4_damageformer = build_a2_dcsm_damageformer
    h4_train.load_h3_source = load_locked_a2_source
    h4_train.load_optional_pretraining = load_dcsm_pretraining
    h4_train.configure_freeze_policy = configure_freeze_policy
    h4_train.keep_frozen_modules_in_eval = keep_frozen_modules_in_eval
    h4_train.HeterogeneousRGBSARDataset = dcsm_dataset
    h4_train.TRAIN_SAMPLER_FACTORY = event_balanced_sampler
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_resume
    print(
        "A2-DCSM contract: locked H6-A2 control; train-only event-balanced "
        "damage-conditioned feature pretraining; M2 adds one display-domain "
        "Scharr structural view; M3 migrates the stage3/4 SAR projection neck. "
        "RGB/localisation/BN and the migrated shallow SAR pathway remain frozen. "
        "No pretraining objective or additional supervised loss is used here.",
        flush=True,
    )
    h4_train.main()
    rewrite_summary_identity()


if __name__ == "__main__":
    main()
