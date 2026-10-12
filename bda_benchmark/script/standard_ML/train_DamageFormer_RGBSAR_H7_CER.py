# -*- coding: utf-8 -*-
"""Train the damaged-only H7 counterfactual router with frozen experts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
import torch.nn.functional as F

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H7_CER import (  # noqa: E402
    build_h7_cer_damageformer,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_H4 as h4_train,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    H6_NEW_MODULE_TOKENS,
    checkpoint_payload as h6_checkpoint_payload,
    load_a2_native_expert,
    load_h4_a4_source,
    load_p3_rgb_anchor,
    weights_payload as h6_weights_payload,
)


class _Variant:
    name = "H7-CER-Damaged-EventContext"


H7_VARIANTS = {"cer_damaged": _Variant()}


def load_sources(model, args, provenance: MutableMapping[str, object]) -> None:
    if not args.sar_pretrained_checkpoint:
        raise RuntimeError("H7-CER requires the locked H6-A2 checkpoint")
    if not args.joint_pretrained_checkpoint:
        raise RuntimeError("H7-CER requires the locked P3 checkpoint")
    provenance["a2_native_expert"] = load_a2_native_expert(
        model, args.sar_pretrained_checkpoint
    )
    provenance["p3_anchor"] = load_p3_rgb_anchor(
        model, args.joint_pretrained_checkpoint
    )


def configure_router_only(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("safe_residual_router"))
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_frozen_experts_in_eval(model) -> None:
    """Keep every frozen expert deterministic while the small router trains."""

    model.eval()
    model.safe_residual_router.train()


def counterfactual_router_loss(
    outputs: Mapping[str, torch.Tensor],
    damage: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    route_logit = outputs["cer_route_logit"][:, 0]
    eligibility = outputs["cer_eligible_mask"][:, 0] > 0
    p3_prediction = outputs["cer_p3_prediction"][:, 0]
    native_prediction = outputs["cer_native_prediction"][:, 0]
    valid = damage != ignore_index
    positive = eligibility & valid & (native_prediction == damage) & (
        p3_prediction != damage
    )
    negative = eligibility & valid & (p3_prediction == damage) & (
        native_prediction != damage
    )

    # Each image contributes equally.  Positive and negative counterfactual
    # cases are balanced inside an image so large events cannot dominate only
    # because they contain more pixels.
    sample_losses = []
    for index in range(route_logit.shape[0]):
        parts = []
        if positive[index].any():
            parts.append(
                F.binary_cross_entropy_with_logits(
                    route_logit[index][positive[index]],
                    torch.ones_like(route_logit[index][positive[index]]),
                )
            )
        if negative[index].any():
            parts.append(
                F.binary_cross_entropy_with_logits(
                    route_logit[index][negative[index]],
                    torch.zeros_like(route_logit[index][negative[index]]),
                )
            )
        if parts:
            sample_losses.append(torch.stack(parts).mean())
    loss = (
        torch.stack(sample_losses).mean()
        if sample_losses
        else route_logit.sum() * 0.0
    )
    eligible_count = eligibility.sum().clamp_min(1)
    return {
        "extra_supervised_loss": loss,
        "cer_positive_rate": positive.sum().float() / eligible_count,
        "cer_negative_rate": negative.sum().float() / eligible_count,
        "cer_eligible_rate": eligibility.float().mean(),
        "cer_route_probability": torch.sigmoid(route_logit)[eligibility].mean()
        if eligibility.any()
        else route_logit.sum() * 0.0,
    }


def weights_payload(model, step, metrics, provenance, args, selector):
    payload = h6_weights_payload(
        model, step, metrics, provenance, args, selector
    )
    payload.update(
        {
            "protocol": "H7_CER_best_weights_v1",
            "experiment": "H7_CER_RGBSAR_DamageFormer",
            "variant": "cer_damaged",
            "routing_policy": "damaged_only_counterfactual_event_context",
        }
    )
    return payload


def checkpoint_payload(
    model, optimizer, scaler, step, best_metrics, selection_records, provenance, args
):
    payload = h6_checkpoint_payload(
        model,
        optimizer,
        scaler,
        step,
        best_metrics,
        selection_records,
        provenance,
        args,
    )
    payload.update(
        {
            "protocol": "H7_CER_resume_v1",
            "experiment": "H7_CER_RGBSAR_DamageFormer",
            "variant": "cer_damaged",
            "routing_policy": "damaged_only_counterfactual_event_context",
        }
    )
    return payload


def load_h7_resume(path, model, optimizer, scaler, variant):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H7_CER_resume_v1":
        raise RuntimeError("unexpected H7-CER resume protocol")
    if checkpoint.get("variant") != "cer_damaged" or variant != "cer_damaged":
        raise RuntimeError("H7-CER resume variant mismatch")
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


def main() -> None:
    h4_train.VALID_H4_VARIANTS = ("cer_damaged",)
    h4_train.H4_VARIANTS = H7_VARIANTS
    h4_train.build_h4_damageformer = lambda variant, pretrained_rgb=True: (
        build_h7_cer_damageformer(pretrained_rgb=pretrained_rgb)
    )
    h4_train.load_h3_source = load_h4_a4_source
    h4_train.load_optional_pretraining = load_sources
    h4_train.configure_freeze_policy = configure_router_only
    h4_train.keep_frozen_modules_in_eval = keep_frozen_experts_in_eval
    h4_train.NEW_MODULE_TOKENS = H6_NEW_MODULE_TOKENS + ("route_predictor",)
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.EXTRA_SUPERVISED_LOSS = counterfactual_router_loss
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h7_resume
    print(
        "H7-CER contract: frozen P3 + frozen H6-A2; train only a damaged-proposal "
        "counterfactual router; no event ID enters the model; test is disabled.",
        flush=True,
    )
    h4_train.main()


if __name__ == "__main__":
    main()
