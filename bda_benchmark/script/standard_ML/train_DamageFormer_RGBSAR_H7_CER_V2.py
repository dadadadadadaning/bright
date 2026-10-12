# -*- coding: utf-8 -*-
"""Train H7-CER-V2-A with one event/pair-balanced selector CE objective."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H7_CER_V2 import (  # noqa: E402
    KEEP_P3,
    REJECT_BOTH,
    USE_A2,
    V2A_ALLOWED_PAIR_IDS,
    build_h7_cer_v2_damageformer,
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
    name = "H7-CER-V2-A-Rejectable-Pair-Conditional"


H7_V2_VARIANTS = {"cer_v2a": _Variant()}


def event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def load_sources(model, args, provenance: MutableMapping[str, object]) -> None:
    if not args.sar_pretrained_checkpoint:
        raise RuntimeError("H7-CER-V2 requires the locked H6-A2 checkpoint")
    if not args.joint_pretrained_checkpoint:
        raise RuntimeError("H7-CER-V2 requires the locked P3 checkpoint")
    provenance["a2_native_expert"] = load_a2_native_expert(
        model, args.sar_pretrained_checkpoint
    )
    provenance["p3_anchor"] = load_p3_rgb_anchor(
        model, args.joint_pretrained_checkpoint
    )


def configure_selector_only(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("safe_residual_router"))
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def keep_frozen_experts_in_eval(model) -> None:
    model.eval()
    model.safe_residual_router.train()


def event_balanced_sampler(dataset, generator) -> WeightedRandomSampler:
    """Draw events uniformly without exposing their names to the model."""
    names = [event_name(name) for name in dataset.data_list]
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    weights = torch.tensor(
        [1.0 / counts[name] for name in names], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(names),
        replacement=True,
        generator=generator,
    )


def selector_targets(
    p3_prediction: torch.Tensor,
    a2_prediction: torch.Tensor,
    damage: torch.Tensor,
    candidate: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create KEEP/USE/REJECT labels only for audited V2-A disagreements."""
    valid = candidate & (damage != ignore_index)
    target = torch.full_like(damage, ignore_index)
    p3_correct = p3_prediction == damage
    a2_correct = a2_prediction == damage
    target[valid & p3_correct & ~a2_correct] = KEEP_P3
    target[valid & a2_correct & ~p3_correct] = USE_A2
    target[valid & ~p3_correct & ~a2_correct] = REJECT_BOTH
    supervised = valid & (target != ignore_index)
    return target, supervised


def _balanced_sample(
    mask: torch.Tensor,
    target: torch.Tensor,
    maximum_per_pair_image: int = 256,
) -> torch.Tensor:
    """Cap dense events while retaining all three selector decisions."""
    selected = torch.zeros_like(mask)
    per_decision = max(1, maximum_per_pair_image // 3)
    for decision in (KEEP_P3, USE_A2, REJECT_BOTH):
        indices = (mask & (target == decision)).flatten().nonzero().flatten()
        if indices.numel() == 0:
            continue
        if indices.numel() > per_decision:
            indices = indices[
                torch.randperm(indices.numel(), device=indices.device)[:per_decision]
            ]
        selected.view(-1)[indices] = True
    return selected


def event_pair_balanced_selector_loss(
    outputs: Mapping[str, torch.Tensor],
    damage: torch.Tensor,
    ignore_index: int,
    sample_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """One CE, averaged decision -> pair -> event (no auxiliary objectives)."""
    logits = outputs["cer_v2_selector_logits"]
    pair_id = outputs["cer_v2_pair_id"][:, 0].long()
    candidate = outputs["cer_v2_candidate_mask"][:, 0] > 0
    p3_prediction = outputs["cer_v2_p3_prediction"][:, 0]
    a2_prediction = outputs["cer_v2_a2_prediction"][:, 0]
    target, supervised = selector_targets(
        p3_prediction, a2_prediction, damage, candidate, ignore_index
    )

    grouped: Dict[str, list[torch.Tensor]] = {}
    sampled_total = logits.new_zeros(())
    decision_counts = [logits.new_zeros(()) for _ in range(3)]
    pair_counts: Dict[int, torch.Tensor] = {
        pair: logits.new_zeros(()) for pair in V2A_ALLOWED_PAIR_IDS
    }
    for index, raw_name in enumerate(sample_names):
        per_pair_losses = []
        for pair in V2A_ALLOWED_PAIR_IDS:
            pair_mask = supervised[index] & (pair_id[index] == pair)
            sampled = _balanced_sample(pair_mask, target[index])
            if not sampled.any():
                continue
            decision_losses = []
            for decision in (KEEP_P3, USE_A2, REJECT_BOTH):
                decision_mask = sampled & (target[index] == decision)
                count = decision_mask.sum()
                decision_counts[decision] = decision_counts[decision] + count
                if decision_mask.any():
                    pixel_logits = logits[index].permute(1, 2, 0)[decision_mask]
                    pixel_target = target[index][decision_mask]
                    decision_losses.append(F.cross_entropy(pixel_logits, pixel_target))
            if decision_losses:
                per_pair_losses.append(torch.stack(decision_losses).mean())
                pair_counts[pair] = pair_counts[pair] + sampled.sum()
                sampled_total = sampled_total + sampled.sum()
        if per_pair_losses:
            grouped.setdefault(event_name(str(raw_name)), []).append(
                torch.stack(per_pair_losses).mean()
            )

    event_losses = [torch.stack(values).mean() for values in grouped.values()]
    loss = torch.stack(event_losses).mean() if event_losses else logits.sum() * 0.0
    normalizer = sampled_total.clamp_min(1.0)
    probabilities = logits.float().softmax(1)
    return {
        "extra_supervised_loss": loss,
        "cer_v2_selector_ce": loss,
        "cer_v2_event_count": logits.new_tensor(float(len(event_losses))),
        "cer_v2_sample_count": sampled_total,
        "cer_v2_keep_fraction": decision_counts[KEEP_P3] / normalizer,
        "cer_v2_use_fraction": decision_counts[USE_A2] / normalizer,
        "cer_v2_reject_fraction": decision_counts[REJECT_BOTH] / normalizer,
        "cer_v2_pair_damaged_background_fraction": (
            pair_counts[V2A_ALLOWED_PAIR_IDS[0]] / normalizer
        ),
        "cer_v2_pair_destroyed_background_fraction": (
            pair_counts[V2A_ALLOWED_PAIR_IDS[1]] / normalizer
        ),
        "cer_v2_probability_keep": probabilities[:, KEEP_P3].mean(),
        "cer_v2_probability_use": probabilities[:, USE_A2].mean(),
        "cer_v2_probability_reject": probabilities[:, REJECT_BOTH].mean(),
    }


def weights_payload(model, step, metrics, provenance, args, selector):
    payload = h6_weights_payload(model, step, metrics, provenance, args, selector)
    payload.update(
        {
            "protocol": "H7_CER_V2_best_weights_v1",
            "experiment": "H7_CER_V2_RGBSAR_DamageFormer",
            "variant": "cer_v2a",
            "routing_policy": "rejectable_pair_conditional_event_balanced_v2a",
            "allowed_pair_ids": list(V2A_ALLOWED_PAIR_IDS),
            "optimized_losses": ["event_pair_balanced_selector_cross_entropy"],
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
            "protocol": "H7_CER_V2_resume_v1",
            "experiment": "H7_CER_V2_RGBSAR_DamageFormer",
            "variant": "cer_v2a",
            "routing_policy": "rejectable_pair_conditional_event_balanced_v2a",
            "allowed_pair_ids": list(V2A_ALLOWED_PAIR_IDS),
            "optimized_losses": ["event_pair_balanced_selector_cross_entropy"],
        }
    )
    return payload


def load_h7_v2_resume(path, model, optimizer, scaler, variant):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H7_CER_V2_resume_v1":
        raise RuntimeError("unexpected H7-CER-V2 resume protocol")
    if checkpoint.get("variant") != variant or variant != "cer_v2a":
        raise RuntimeError("H7-CER-V2 resume variant mismatch")
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
    h4_train.VALID_H4_VARIANTS = ("cer_v2a",)
    h4_train.H4_VARIANTS = H7_V2_VARIANTS
    h4_train.build_h4_damageformer = lambda variant, pretrained_rgb=True: (
        build_h7_cer_v2_damageformer(pretrained_rgb=pretrained_rgb)
    )
    h4_train.load_h3_source = load_h4_a4_source
    h4_train.load_optional_pretraining = load_sources
    h4_train.configure_freeze_policy = configure_selector_only
    h4_train.keep_frozen_modules_in_eval = keep_frozen_experts_in_eval
    h4_train.NEW_MODULE_TOKENS = H6_NEW_MODULE_TOKENS + (
        "safe_residual_router",
        "pair_embedding",
        "selector",
    )
    h4_train.REQUIRE_H4_SAR_PRETRAINING = False
    h4_train.EXTRA_SUPERVISED_LOSS = event_pair_balanced_selector_loss
    h4_train.EXTRA_SUPERVISED_USES_NAMES = True
    h4_train.TRAIN_SAMPLER_FACTORY = event_balanced_sampler
    h4_train.checkpoint_payload = checkpoint_payload
    h4_train.weights_payload = weights_payload
    h4_train.load_h4_resume = load_h7_v2_resume
    print(
        "H7-CER-V2-A contract: frozen P3/A2; only Dmg->Bg and Des->Bg; "
        "train a KEEP/USE/REJECT selector with ONE event/pair-balanced CE; "
        "event names are used only for loss averaging and never enter the model; "
        "test is disabled.",
        flush=True,
    )
    h4_train.main()


if __name__ == "__main__":
    main()
