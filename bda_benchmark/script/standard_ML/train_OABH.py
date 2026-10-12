# -*- coding: utf-8 -*-
"""Train OABH by reusing the audited heterogeneous-training runtime.

The wrapper changes only the model factory, declared variants and optional
initialisation.  Data, metrics, checkpoint semantics and the no-test training
contract remain identical to the established BRIGHT runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_OABH import (
    OABH_VARIANTS,
    VALID_OABH_VARIANTS,
    build_oabh_damageformer,
    load_full_oabh_initialisation,
    load_h6_encoder_initialisation,
)
from bda_benchmark.script.standard_ML import (
    train_DamageFormer_RGBSAR_Heterogeneous as runtime,
)


_NATIVE_SAR_DATASET = runtime.HeterogeneousRGBSARDataset


def _oabh_native_sar_dataset(*args, **kwargs):
    """Decouple model-ablation names from the native-SAR input contract.

    The shared dataset only needs to distinguish pseudo-RGB ``h0`` from a
    native one-channel SAR experiment.  OA-HSDNet variants are architecture
    names, so they are deliberately mapped to the already-audited ``x1``
    native-SAR contract instead of requiring every dataset copy to know them.
    """
    model_variant = str(kwargs.get("variant", "")).lower()
    if model_variant not in VALID_OABH_VARIANTS:
        raise ValueError(f"unexpected OA-HSDNet dataset variant: {model_variant}")
    kwargs["variant"] = "x1"
    dataset = _NATIVE_SAR_DATASET(*args, **kwargs)
    dataset.model_variant = model_variant
    return dataset


def _zero_regularisation(outputs, labels, **kwargs):
    reference = outputs["output_dam"]
    zero = reference.sum() * 0.0
    return {
        "representation_loss": zero,
        "common_unchanged_loss": zero,
        "common_unique_orthogonality_loss": zero,
    }


def _factory(variant: str, pretrained_rgb: bool = True):
    full_init = os.environ.get("OABH_FULL_INIT_CKPT", "").strip()
    h6_init = os.environ.get("OABH_H6_INIT_CKPT", "").strip()
    if full_init and h6_init:
        raise RuntimeError("set only OABH_FULL_INIT_CKPT or OABH_H6_INIT_CKPT")
    # A supplied checkpoint fully determines encoder weights; avoid an
    # unnecessary ImageNet download before replacing them.
    style_mode = os.environ.get("OABH_SAR_STYLE_MODE", "none").strip().lower()
    mix_probability = float(os.environ.get("OABH_SAR_MIX_PROBABILITY", "0.5"))
    mix_alpha = float(os.environ.get("OABH_SAR_MIX_ALPHA", "0.1"))
    model = build_oabh_damageformer(
        variant,
        pretrained_rgb=pretrained_rgb and not (full_init or h6_init),
        sar_style_mode=style_mode,
        sar_mix_probability=mix_probability,
        sar_mix_alpha=mix_alpha,
    )
    if full_init:
        if not Path(full_init).is_file():
            raise FileNotFoundError(f"missing OABH full initialisation: {full_init}")
        load_full_oabh_initialisation(model, full_init)
        print(f"OABH full initialisation: {full_init}", flush=True)
    elif h6_init:
        if not Path(h6_init).is_file():
            raise FileNotFoundError(f"missing H6 encoder initialisation: {h6_init}")
        report = load_h6_encoder_initialisation(model, h6_init)
        print(f"OABH H6 encoder initialisation: {report}", flush=True)
    phase = os.environ.get("OABH_PHASE", "joint").strip().lower()
    model.set_training_phase(phase)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"OABH phase={phase} | sar_style={style_mode} | "
        f"trainable/total={trainable}/{total}",
        flush=True,
    )
    return model


def main() -> None:
    runtime.VALID_VARIANTS = VALID_OABH_VARIANTS
    runtime.VARIANTS = OABH_VARIANTS
    runtime.build_heterogeneous_damageformer = _factory
    runtime.common_unique_regularization = _zero_regularisation
    runtime.HeterogeneousRGBSARDataset = _oabh_native_sar_dataset
    runtime.main()


if __name__ == "__main__":
    main()
