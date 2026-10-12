# -*- coding: utf-8 -*-
"""Train controlled H4 upgrades from a validated H3 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H4 import (  # noqa: E402
    H4_VARIANTS,
    VALID_H4_VARIANTS,
    build_h4_damageformer,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (  # noqa: E402
    fast_hist,
    lovasz_softmax,
    metrics_from_hist,
    seed_worker,
)


NEW_MODULE_TOKENS = (
    "flow_alignment",
    "gate_predictor",
    "gated_transition",
    "hierarchy_",
)

# Descendant experiments may register one principled, supervised auxiliary
# objective without copying this validated optimizer/data/evaluation loop.
# The default remains None, so every existing H4/H5/H6 experiment preserves
# its original numerical behaviour.
EXTRA_SUPERVISED_LOSS = None
# Descendants that need event-balanced supervision may request the sample names
# returned by the dataset.  The default keeps every existing H4/H5/H6 callback
# signature and numerical path unchanged.
EXTRA_SUPERVISED_USES_NAMES = False
# Optional sampler hook for descendants that require event-balanced training.
# Existing experiments keep shuffle=True and therefore preserve behaviour.
TRAIN_SAMPLER_FACTORY = None

# Controlled descendant trainers reuse this validated driver with different
# variant names and their own checkpoint contracts.  H4 keeps the historical
# default; descendants may explicitly disable only this parser-level check.
REQUIRE_H4_SAR_PRETRAINING = True

# Zero-residual descendants may explicitly admit their strictly validated
# step-0 state to checkpoint selection.  The default is False so every
# historical H4/H5/H6 experiment retains its original selection behaviour.
SELECT_INITIAL_CHECKPOINT = False
# Descendant experiments may track additional diagnostic selectors without
# changing the default H4 checkpoint policy.
SECONDARY_BEST_SELECTORS = ("event", "balance")
# When CE/Lovasz are both disabled, the dummy 0 * output_dam.sum() becomes
# NaN if the routed logits overflow.  Descendants may point this at a tensor
# known to stay finite (H12 uses output_loc).  Default keeps H4/H5/H6 identical.
ZERO_LOSS_REFERENCE_KEY = "output_dam"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_container(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a model state dictionary")
    return state  # type: ignore[return-value]


def load_h3_source(
    model: nn.Module,
    path: str,
    require_exact: bool,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "h3":
        raise RuntimeError(
            f"H4 source must be H3, received variant={checkpoint.get('variant')!r}"
        )
    missing, unexpected = model.load_state_dict(
        _state_container(checkpoint), strict=False
    )
    allowed_missing = [
        key
        for key in missing
        if any(token in key for token in NEW_MODULE_TOKENS)
    ]
    illegal_missing = sorted(set(missing) - set(allowed_missing))
    if unexpected or illegal_missing or (require_exact and missing):
        raise RuntimeError(
            "H3-to-H4 migration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_variant": checkpoint.get("variant"),
        "missing_new_keys": sorted(allowed_missing),
        "unexpected_keys": sorted(unexpected),
        "source_best_metrics": dict(
            checkpoint.get("best_metrics", checkpoint.get("metrics", {}))
        ),
    }


def load_sar_pretraining(model: nn.Module, path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H4_SAR_MAE_encoder_v1":
        raise RuntimeError(
            f"unexpected SAR pretraining protocol: {checkpoint.get('protocol')}"
        )
    state = checkpoint.get("sar_encoder")
    if not isinstance(state, Mapping):
        raise RuntimeError("SAR pretraining checkpoint has no sar_encoder state")
    model.sar_encoder.load_state_dict(state, strict=True)
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": checkpoint.get("protocol"),
        "step": int(checkpoint.get("step", 0)),
        "source_h3": dict(checkpoint.get("source_h3", {})),
    }


def load_optional_pretraining(
    model: nn.Module,
    args,
    provenance: MutableMapping[str, object],
) -> None:
    """Load variant-specific pretraining without coupling the main driver.

    H4 keeps its historical SAR-MAE behavior.  Later controlled branches may
    replace this hook while continuing to reuse the validated data, loss,
    evaluation, and checkpoint pipeline.
    """

    if args.variant not in ("a3", "a4"):
        return
    if not args.sar_pretrained_checkpoint:
        raise RuntimeError(f"{args.variant.upper()} requires SAR pretraining")
    provenance["sar_pretraining"] = load_sar_pretraining(
        model,
        args.sar_pretrained_checkpoint,
    )
    pretrained_source = provenance["sar_pretraining"].get("source_h3", {})
    if (
        not isinstance(pretrained_source, Mapping)
        or pretrained_source.get("sha256")
        != provenance["h3_source"].get("sha256")
    ):
        raise RuntimeError(
            "SAR pretraining and supervised H4 use different H3 sources"
        )


def load_damage_priors(path: str) -> Dict[str, Sequence[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != "H4_train_only_damage_priors_v1":
        raise RuntimeError("unexpected damage-prior protocol")
    hierarchical = payload.get("hierarchical_counts", {})
    required = ("building", "affected", "destroyed")
    if not all(key in hierarchical for key in required):
        raise RuntimeError("damage-prior file lacks hierarchical counts")
    result: Dict[str, Sequence[float]] = {}
    for key in required:
        values = [float(value) for value in hierarchical[key]]
        if len(values) != 2 or min(values) <= 0:
            raise RuntimeError(f"invalid {key} prior counts: {values}")
        result[key] = values
    return result


def configure_freeze_policy(model: nn.Module) -> None:
    """Freeze RGB/localization and the shallow half of the SAR hierarchy."""

    for name, parameter in model.named_parameters():
        permanently_frozen = (
            name.startswith("rgb_encoder")
            or name.startswith("localization_decoder")
            or name.startswith("localization_head")
            or name.startswith("sar_encoder.stem")
            or name.startswith("sar_encoder.stages.0")
            or name.startswith("sar_encoder.stages.1")
            or name.startswith("sar_encoder.downsamples.0")
            or name.startswith("sar_encoder.downsamples.1")
        )
        parameter.requires_grad_(not permanently_frozen)


def keep_frozen_modules_in_eval(model: nn.Module) -> None:
    model.rgb_encoder.eval()
    model.localization_decoder.eval()
    model.localization_head.eval()
    model.sar_encoder.stem.eval()
    model.sar_encoder.stages[0].eval()
    model.sar_encoder.stages[1].eval()
    model.sar_encoder.downsamples[0].eval()
    model.sar_encoder.downsamples[1].eval()


def suppress_warmup_gradients(model: nn.Module, step: int, warmup_steps: int) -> None:
    if step >= warmup_steps:
        return
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not any(token in name for token in NEW_MODULE_TOKENS):
            parameter.grad = None


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    new_parameters = []
    sar_parameters = []
    core_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in NEW_MODULE_TOKENS):
            new_parameters.append(parameter)
        elif name.startswith("sar_encoder"):
            sar_parameters.append(parameter)
        else:
            core_parameters.append(parameter)
    groups = []
    if new_parameters:
        groups.append(
            {"params": new_parameters, "lr": args.new_module_learning_rate}
        )
    if sar_parameters:
        groups.append({"params": sar_parameters, "lr": args.sar_learning_rate})
    if core_parameters:
        groups.append({"params": core_parameters, "lr": args.core_learning_rate})
    if not groups:
        raise RuntimeError("freeze policy left no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    ce_weight: float,
    lovasz_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.float()
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    lovasz = lovasz_softmax(
        F.softmax(logits, dim=1),
        labels,
        classes="present",
        ignore=ignore_index,
    )
    return ce_weight * ce + lovasz_weight * lovasz, ce, lovasz


def balanced_binary_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    counts: Sequence[float],
    ignore_index: int,
    prior_strength: float,
) -> torch.Tensor:
    if not (targets != ignore_index).any():
        return logits.float().sum() * 0.0
    counts_tensor = torch.as_tensor(
        counts, dtype=torch.float32, device=logits.device
    )
    adjustment = float(prior_strength) * counts_tensor.clamp_min(1.0).log()
    return F.cross_entropy(
        logits.float() + adjustment.view(1, 2, 1, 1),
        targets,
        ignore_index=ignore_index,
    )


def hierarchical_damage_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    priors: Mapping[str, Sequence[float]],
    ignore_index: int,
    prior_strength: float,
) -> Dict[str, torch.Tensor]:
    reference = outputs["output_dam"]
    zero = reference.float().sum() * 0.0
    required = (
        "hierarchy_building_logits",
        "hierarchy_affected_logits",
        "hierarchy_destroyed_logits",
    )
    if not all(key in outputs for key in required):
        return {
            "hierarchy_loss": zero,
            "hierarchy_building_loss": zero,
            "hierarchy_affected_loss": zero,
            "hierarchy_destroyed_loss": zero,
        }

    valid = labels != ignore_index
    building_target = torch.full_like(labels, ignore_index)
    building_target[valid & (labels == 0)] = 0
    building_target[valid & (labels >= 1) & (labels <= 3)] = 1

    affected_target = torch.full_like(labels, ignore_index)
    affected_target[valid & (labels == 1)] = 0
    affected_target[valid & ((labels == 2) | (labels == 3))] = 1

    destroyed_target = torch.full_like(labels, ignore_index)
    destroyed_target[valid & (labels == 2)] = 0
    destroyed_target[valid & (labels == 3)] = 1

    building_loss = balanced_binary_ce(
        outputs["hierarchy_building_logits"],
        building_target,
        priors["building"],
        ignore_index,
        prior_strength,
    )
    affected_loss = balanced_binary_ce(
        outputs["hierarchy_affected_logits"],
        affected_target,
        priors["affected"],
        ignore_index,
        prior_strength,
    )
    destroyed_loss = balanced_binary_ce(
        outputs["hierarchy_destroyed_logits"],
        destroyed_target,
        priors["destroyed"],
        ignore_index,
        prior_strength,
    )
    return {
        "hierarchy_loss": 0.25 * building_loss
        + 0.50 * affected_loss
        + 0.50 * destroyed_loss,
        "hierarchy_building_loss": building_loss,
        "hierarchy_affected_loss": affected_loss,
        "hierarchy_destroyed_loss": destroyed_loss,
    }


def flow_regularization(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    """Regularize flow only on unchanged background/intact pixels."""

    zero = outputs["output_dam"].float().sum() * 0.0
    magnitude_terms = []
    smoothness_terms = []
    for stage in range(1, 5):
        key = f"stage{stage}_flow"
        if key not in outputs:
            continue
        flow = outputs[key].float()
        target = F.interpolate(
            labels.unsqueeze(1).float(),
            size=flow.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        unchanged = ((target == 0) | (target == 1)) & (target != ignore_index)
        # The flow predictor is deliberately initialized to exact zero.
        # sqrt(x**2 + y**2) has an undefined derivative at that point and can
        # turn the first flow update into NaN.  A zero-centred Charbonnier norm
        # preserves a zero loss/gradient at zero while remaining smooth.
        squared_magnitude = flow.square().sum(1)
        epsilon = 1e-6
        magnitude = torch.sqrt(squared_magnitude + epsilon) - math.sqrt(
            epsilon
        )
        if unchanged.any():
            magnitude_terms.append(magnitude[unchanged].mean())

        dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean(1)
        dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean(1)
        valid_x = unchanged[:, :, 1:] & unchanged[:, :, :-1]
        valid_y = unchanged[:, 1:, :] & unchanged[:, :-1, :]
        if valid_x.any():
            smoothness_terms.append(dx[valid_x].mean())
        if valid_y.any():
            smoothness_terms.append(dy[valid_y].mean())

    magnitude_loss = (
        torch.stack(magnitude_terms).mean() if magnitude_terms else zero
    )
    smoothness_loss = (
        torch.stack(smoothness_terms).mean() if smoothness_terms else zero
    )
    return {
        "flow_regularization_loss": magnitude_loss + 0.5 * smoothness_loss,
        "flow_magnitude_loss": magnitude_loss,
        "flow_smoothness_loss": smoothness_loss,
    }


def common_unique_regularization(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    zero = outputs["output_dam"].float().sum() * 0.0
    unchanged_terms = []
    orthogonal_terms = []
    for stage in range(1, 5):
        keys = (
            f"stage{stage}_rgb_common",
            f"stage{stage}_sar_common",
            f"stage{stage}_rgb_unique",
            f"stage{stage}_sar_unique",
        )
        if not all(key in outputs for key in keys):
            continue
        rgb_common, sar_common, rgb_unique, sar_unique = (
            outputs[key].float() for key in keys
        )
        target = F.interpolate(
            labels.unsqueeze(1).float(),
            size=rgb_common.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        cosine_distance = 1.0 - F.cosine_similarity(
            rgb_common, sar_common, dim=1
        )
        class_terms = []
        for class_index in (0, 1):
            mask = (target == class_index) & (target != ignore_index)
            if mask.any():
                class_terms.append(cosine_distance[mask].mean())
        if class_terms:
            unchanged_terms.append(torch.stack(class_terms).mean())
        for common, unique in (
            (rgb_common, rgb_unique),
            (sar_common, sar_unique),
        ):
            common = F.normalize(common.flatten(2), dim=1)
            unique = F.normalize(unique.flatten(2), dim=1)
            orthogonal_terms.append((common * unique).sum(1).square().mean())
    unchanged_loss = (
        torch.stack(unchanged_terms).mean() if unchanged_terms else zero
    )
    orthogonal_loss = (
        torch.stack(orthogonal_terms).mean() if orthogonal_terms else zero
    )
    return {
        "representation_loss": unchanged_loss + 0.05 * orthogonal_loss,
        "common_unchanged_loss": unchanged_loss,
        "common_unique_orthogonality_loss": orthogonal_loss,
    }


def _accumulate(
    totals: MutableMapping[str, float],
    counts: MutableMapping[str, int],
    key: str,
    value: float,
) -> None:
    totals[key] = totals.get(key, 0.0) + float(value)
    counts[key] = counts.get(key, 0) + 1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hist: Dict[str, torch.Tensor] = {}
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    processed = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rgb, sar, location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar, return_dict=True)
        loc_prediction = outputs["output_loc"].argmax(1)
        # H5-style building-query children may expose a strict one-label-per-
        # instance decoder.  Prefer it for validation/checkpoint selection
        # while retaining output_dam for the differentiable training path.
        evaluation_damage_logits = outputs.get(
            "instance_query_output_dam",
            outputs["output_dam"],
        )
        damage_prediction = evaluation_damage_logits.argmax(1)
        loc_hist += fast_hist(loc_prediction, location, 2, ignore_index)
        damage_hist += fast_hist(damage_prediction, damage, 4, ignore_index)

        for key, value in outputs.items():
            if key.endswith("_reliability"):
                _accumulate(totals, counts, key, value.float().mean().item())
            elif key.endswith("_sar_gate"):
                gate = value.float()
                _accumulate(totals, counts, key, gate.mean().item())
                target = F.interpolate(
                    damage.unsqueeze(1).float(),
                    size=gate.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
                for class_index, class_name in enumerate(
                    ("background", "intact", "damaged", "destroyed")
                ):
                    mask = target == class_index
                    if mask.any():
                        gate_channel = (
                            gate[:, class_index]
                            if gate.shape[1] == 4
                            else gate[:, 0]
                        )
                        _accumulate(
                            totals,
                            counts,
                            f"{key}_{class_name}",
                            gate_channel[mask].mean().item(),
                        )
            elif key.endswith("_flow"):
                magnitude = value.float().square().sum(1).sqrt()
                _accumulate(
                    totals, counts, f"{key}_mean", magnitude.mean().item()
                )
                _accumulate(
                    totals,
                    counts,
                    f"{key}_p90",
                    torch.quantile(magnitude.flatten(), 0.9).item(),
                )
            elif key.endswith("_transition"):
                _accumulate(totals, counts, key, value.float().mean().item())
                if key == "safe_class_transition" and value.numel() == 4:
                    flat = value.float().reshape(-1)
                    for class_index, class_name in enumerate(
                        ("background", "intact", "damaged", "destroyed")
                    ):
                        _accumulate(
                            totals,
                            counts,
                            f"{key}_{class_name}",
                            flat[class_index].item(),
                        )
            elif key.endswith("_sensor_gap_prototype"):
                _accumulate(
                    totals,
                    counts,
                    f"{key}_abs_mean",
                    value.float().abs().mean().item(),
                )
            elif key.endswith(
                (
                    "_sensor_gap_raw_abs",
                    "_sensor_gap_corrected_abs",
                    "_unchanged_confidence",
                    "_unchanged_mass",
                    "_low_frequency_abs",
                    "_frequency_gate",
                )
            ):
                _accumulate(totals, counts, key, value.float().mean().item())
            elif key.endswith("_bcsc_building_prototype"):
                _accumulate(
                    totals,
                    counts,
                    f"{key}_abs_mean",
                    value.float().abs().mean().item(),
                )
            elif key.endswith(
                (
                    "_bcsc_prototype_shift_abs",
                    "_bcsc_prototype_confidence",
                    "_bcsc_building_mass",
                    "_bcsc_prototype_blend",
                    "_bcsc_reliability_gate",
                )
            ):
                diagnostic = value.float()
                _accumulate(totals, counts, key, diagnostic.mean().item())
                if key.endswith("_bcsc_reliability_gate"):
                    target = F.interpolate(
                        damage.unsqueeze(1).float(),
                        size=diagnostic.shape[-2:],
                        mode="nearest",
                    ).squeeze(1).long()
                    gate = diagnostic[:, 0]
                    for class_index, class_name in enumerate(
                        ("background", "intact", "damaged", "destroyed")
                    ):
                        mask = target == class_index
                        if mask.any():
                            _accumulate(
                                totals,
                                counts,
                                f"{key}_{class_name}",
                                gate[mask].mean().item(),
                            )
            elif key.endswith(
                (
                    "_query_count",
                    "_instance_count",
                    "_delta_abs",
                    "_offset_mean",
                    "_residual_abs",
                )
            ):
                # H5-style structural diagnostics are scalar or per-sample
                # tensors.  H4 itself does not emit these keys, so this keeps
                # the validated H4 metric path unchanged while allowing a
                # structurally compatible child model to report its activity.
                _accumulate(totals, counts, key, value.float().mean().item())

        for sample_index in range(rgb.shape[0]):
            event = re.sub(r"_\d+$", "", str(names[sample_index]))
            sample_hist = fast_hist(
                damage_prediction[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1],
                4,
                ignore_index,
            )
            event_hist[event] = event_hist.get(
                event, torch.zeros_like(sample_hist)
            ) + sample_hist
        processed += int(rgb.shape[0])

    metrics = metrics_from_hist(loc_hist, damage_hist)
    # Hierarchical occurrence diagnostics derived from the same official
    # 4-class confusion matrix.  These are metrics only; they do not change
    # the training objective or checkpoint selector.
    affected_total = damage_hist[2:4, :].sum().clamp_min(1.0)
    affected_tp = damage_hist[2:4, 2:4].sum()
    affected_fp = damage_hist[0:2, 2:4].sum()
    affected_fn = damage_hist[2:4, 0:2].sum()
    metrics["affected_recall"] = float((affected_tp / affected_total).item())
    metrics["affected_iou"] = float(
        (affected_tp / (affected_tp + affected_fp + affected_fn).clamp_min(1.0)).item()
    )
    metrics["affected_f1"] = float(
        (2.0 * affected_tp / (2.0 * affected_tp + affected_fp + affected_fn).clamp_min(1.0)).item()
    )
    damaged_total = damage_hist[2, :].sum().clamp_min(1.0)
    destroyed_total = damage_hist[3, :].sum().clamp_min(1.0)
    metrics["damaged_gate_miss"] = float(
        (damage_hist[2, 0:2].sum() / damaged_total).item()
    )
    metrics["destroyed_gate_miss"] = float(
        (damage_hist[3, 0:2].sum() / destroyed_total).item()
    )
    location_tp = loc_hist[1, 1]
    metrics["loc_precision"] = float(
        (location_tp / loc_hist[:, 1].sum().clamp_min(1.0)).item()
    )
    metrics["loc_recall"] = float(
        (location_tp / loc_hist[1, :].sum().clamp_min(1.0)).item()
    )
    intact_total = damage_hist[1].sum().clamp_min(1.0)
    metrics["intact_to_damaged"] = float(
        (damage_hist[1, 2] / intact_total).item()
    )
    metrics["intact_to_destroyed"] = float(
        (damage_hist[1, 3] / intact_total).item()
    )
    event_values = []
    for histogram in event_hist.values():
        intersection = torch.diag(histogram)
        union = histogram.sum(1) + histogram.sum(0) - intersection
        event_values.append(
            float((intersection / union.clamp_min(1.0)).mean().item())
        )
    metrics["processed_images"] = processed
    metrics["event_count"] = len(event_values)
    metrics["event_mean_official_style_4class_miou"] = (
        float(np.mean(event_values)) if event_values else 0.0
    )
    metrics["event_worst_official_style_4class_miou"] = (
        float(np.min(event_values)) if event_values else 0.0
    )
    metrics["event_std_official_style_4class_miou"] = (
        float(np.std(event_values)) if event_values else 0.0
    )
    metrics["damage_balance_score"] = float(
        0.5 * (metrics["damaged_iou"] + metrics["destroyed_iou"])
        - 0.25
        * (metrics["damaged_to_intact"] + metrics["destroyed_to_intact"])
    )
    for key, total in totals.items():
        metrics[key] = total / max(counts.get(key, 1), 1)
    model.train()
    keep_frozen_modules_in_eval(model)
    return metrics


def metrics_line(prefix: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{prefix} | valid_mIoU={100*metrics['valid_miou']:.4f} | "
        f"mIoU4={100*metrics['miou4']:.4f} | "
        f"Loc F1={100*metrics['loc_f1']:.2f} | "
        f"IoU(int/dmg/des)={100*metrics['intact_iou']:.2f}/"
        f"{100*metrics['damaged_iou']:.2f}/{100*metrics['destroyed_iou']:.2f} | "
        f"Dmg->Int={100*metrics['damaged_to_intact']:.2f}% | "
        f"Des->Int={100*metrics['destroyed_to_intact']:.2f}% | "
        f"Event mean/worst={100*metrics['event_mean_official_style_4class_miou']:.2f}/"
        f"{100*metrics['event_worst_official_style_4class_miou']:.2f}"
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
        "protocol": "H4_RGBSAR_resume_v1",
        "experiment": "DamageFormer_RGBSAR_H4",
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
        "protocol": "H4_RGBSAR_best_weights_v1",
        "experiment": "DamageFormer_RGBSAR_H4",
        "variant": args.variant,
        "selector": selector,
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "args": vars(args),
    }


def load_h4_resume(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    variant: str,
) -> tuple[int, Dict[str, float], Dict[str, Dict[str, object]], Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "H4_RGBSAR_resume_v1":
        raise RuntimeError("unexpected H4 resume protocol")
    if checkpoint.get("variant") != variant:
        raise RuntimeError("H4 resume variant mismatch")
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


def _update_selection_records(
    records: MutableMapping[str, Dict[str, object]],
    metrics: Mapping[str, float],
    step: int,
) -> Sequence[str]:
    selectors = {
        "valid": float(metrics["valid_miou"]),
        "event": float(metrics["event_mean_official_style_4class_miou"]),
        "balance": float(metrics["damage_balance_score"]),
    }
    improved = []
    for name, score in selectors.items():
        previous = float(records.get(name, {}).get("score", -math.inf))
        if score > previous:
            records[name] = {
                "score": score,
                "step": int(step),
                "metrics": dict(metrics),
            }
            improved.append(name)
    return improved


def main() -> None:
    parser = argparse.ArgumentParser("Train H4 heterogeneous RGB/SAR DamageFormer")
    parser.add_argument("--variant", choices=VALID_H4_VARIANTS, default="a1")
    parser.add_argument("--source_h3_checkpoint", required=True)
    parser.add_argument("--sar_pretrained_checkpoint", default="")
    parser.add_argument("--joint_pretrained_checkpoint", default="")
    parser.add_argument("--damage_priors_json", default="")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--test_dataset_path", default="")
    parser.add_argument("--test_data_list_path", default="")
    parser.add_argument(
        "--test_every_validation",
        action="store_true",
        help=(
            "Monitor the test split after every validation pass. Test metrics "
            "are logged only and never participate in checkpoint selection."
        ),
    )
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument(
        "--dataset_variant",
        choices=("h3", "h4"),
        default="h3",
        help="h4 additionally carries an un-clipped audit channel for H6-B P3 safety.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument(
        "--focus_probability",
        type=float,
        default=0.0,
        help=(
            "Probability that a training crop is centred on a labelled class. "
            "The default 0 preserves every historical H4/H6 experiment."
        ),
    )
    parser.add_argument(
        "--focus_class_weights",
        type=float,
        nargs=4,
        default=(0.0, 1.0, 2.0, 2.0),
        metavar=("BACKGROUND", "INTACT", "DAMAGED", "DESTROYED"),
        help="Relative class weights used only when a focus crop is requested.",
    )
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=3000)
    parser.add_argument("--new_module_warmup_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--test_max_batches", type=int, default=0)
    parser.add_argument("--new_module_learning_rate", type=float, default=1e-4)
    parser.add_argument("--sar_learning_rate", type=float, default=1e-5)
    parser.add_argument("--core_learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--damage_ce_weight", type=float, default=1.0)
    parser.add_argument("--damage_lovasz_weight", type=float, default=0.75)
    parser.add_argument("--sar_aux_weight", type=float, default=0.10)
    parser.add_argument("--representation_weight", type=float, default=0.05)
    parser.add_argument("--flow_regularization_weight", type=float, default=0.01)
    parser.add_argument("--hierarchy_weight", type=float, default=0.50)
    parser.add_argument("--extra_supervised_weight", type=float, default=0.0)
    parser.add_argument("--prior_strength", type=float, default=0.50)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_pretrained_rgb", action="store_true")
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--save_secondary_best", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.focus_probability <= 1.0:
        parser.error("--focus_probability must lie in [0, 1]")
    if min(args.focus_class_weights) < 0 or not any(args.focus_class_weights):
        parser.error("--focus_class_weights must be non-negative and not all zero")

    if args.test_every_validation and (
        not args.test_dataset_path or not args.test_data_list_path
    ):
        parser.error(
            "--test_every_validation requires --test_dataset_path and "
            "--test_data_list_path"
        )

    if (
        REQUIRE_H4_SAR_PRETRAINING
        and args.variant in ("a3", "a4")
        and not args.sar_pretrained_checkpoint
    ):
        parser.error("A3/A4 require --sar_pretrained_checkpoint")
    if args.variant == "a4" and not args.damage_priors_json:
        parser.error("A4 requires --damage_priors_json")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_h4_damageformer(
        args.variant, pretrained_rgb=not args.no_pretrained_rgb
    ).to(device)
    provenance: Dict[str, object] = {}
    if not args.resume:
        provenance["h3_source"] = load_h3_source(
            model,
            args.source_h3_checkpoint,
            require_exact=args.variant == "a0",
        )
        load_optional_pretraining(model, args, provenance)
    configure_freeze_policy(model)
    optimizer = make_optimizer(model, args)
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and args.amp == "fp16"
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and args.amp == "fp16"
        )

    step = 0
    best_metrics: Dict[str, float] = {}
    selection_records: Dict[str, Dict[str, object]] = {}
    if args.resume:
        step, best_metrics, selection_records, provenance = load_h4_resume(
            args.resume, model, optimizer, scaler, args.variant
        )

    priors: Mapping[str, Sequence[float]] = {
        "building": (1.0, 1.0),
        "affected": (1.0, 1.0),
        "destroyed": (1.0, 1.0),
    }
    if args.damage_priors_json:
        priors = load_damage_priors(args.damage_priors_json)
        provenance["damage_priors_json"] = str(
            Path(args.damage_priors_json).resolve()
        )

    total_crops = max(
        args.train_batch_size,
        args.optimizer_steps
        * args.train_batch_size
        * args.accumulation_steps,
    )
    train_dataset = HeterogeneousRGBSARDataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        max_iters=total_crops,
        split="train",
        variant=args.dataset_variant,
        sar_stats_path=args.sar_stats_path,
        focus_probability=args.focus_probability,
        focus_class_weights=args.focus_class_weights,
    )
    val_dataset = HeterogeneousRGBSARDataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        split="val",
        variant=args.dataset_variant,
        sar_stats_path=args.sar_stats_path,
    )
    test_dataset = None
    if args.test_every_validation:
        test_dataset = HeterogeneousRGBSARDataset(
            args.test_dataset_path,
            args.test_data_list_path,
            args.crop_size,
            split="test",
            variant=args.dataset_variant,
            sar_stats_path=args.sar_stats_path,
        )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_sampler = (
        TRAIN_SAMPLER_FACTORY(train_dataset, generator)
        if callable(TRAIN_SAMPLER_FACTORY)
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("=" * 80, flush=True)
    print(f"Experiment       : {H4_VARIANTS[args.variant].name}", flush=True)
    print(f"Variant          : {args.variant}", flush=True)
    print("Teacher          : NONE", flush=True)
    print("RGB/Loc          : FROZEN / FROZEN", flush=True)
    print("SAR trainable    : stages 3-4 only after warmup", flush=True)
    print(
        f"LR new/SAR/core  : {args.new_module_learning_rate}/"
        f"{args.sar_learning_rate}/{args.core_learning_rate}",
        flush=True,
    )
    print(
        f"Steps/EBS/crops  : {args.optimizer_steps}/"
        f"{args.train_batch_size*args.accumulation_steps}/{total_crops}",
        flush=True,
    )
    print(f"Trainable/total  : {trainable}/{total}", flush=True)
    print(
        "Checkpoint policy: best valid weights + latest resume"
        + (" + secondary best weights" if args.save_secondary_best else ""),
        flush=True,
    )
    print(
        "Test monitoring   : "
        + (
            "EVERY validation; LOG ONLY; NEVER a checkpoint selector"
            if args.test_every_validation
            else "disabled"
        ),
        flush=True,
    )
    print(f"Output           : {output_dir}", flush=True)
    print("=" * 80, flush=True)

    initial_metrics = evaluate(
        model, val_loader, device, args.ignore_index, args.val_max_batches
    )
    print(metrics_line("Initial validation", initial_metrics), flush=True)
    (output_dir / "initial_validation.json").write_text(
        json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8"
    )
    test_monitor_history = []
    test_history_path = output_dir / "test_monitor_history.json"
    if args.resume and test_history_path.exists():
        loaded_history = json.loads(test_history_path.read_text(encoding="utf-8"))
        if isinstance(loaded_history, list):
            test_monitor_history = loaded_history
    if test_loader is not None:
        initial_test_metrics = evaluate(
            model,
            test_loader,
            device,
            args.ignore_index,
            args.test_max_batches,
        )
        print(
            metrics_line("Test monitor (initial; not selector)", initial_test_metrics),
            flush=True,
        )
        test_monitor_history.append(
            {
                "step": int(step),
                "phase": "initial",
                "selector": False,
                "validation_metrics": dict(initial_metrics),
                "metrics": dict(initial_test_metrics),
            }
        )
        test_history_path.write_text(
            json.dumps(test_monitor_history, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "latest_test_monitor.json").write_text(
            json.dumps(test_monitor_history[-1], indent=2) + "\n",
            encoding="utf-8",
        )
    if SELECT_INITIAL_CHECKPOINT and not args.validation_only:
        improved = _update_selection_records(
            selection_records, initial_metrics, step
        )
        if "valid" in improved:
            best_metrics = dict(initial_metrics)
            atomic_save(
                weights_payload(
                    model,
                    step,
                    initial_metrics,
                    provenance,
                    args,
                    selector="valid",
                ),
                output_dir / "best_val_miou.pth",
            )
            print(
                "Initial safe state admitted to selection | "
                f"step={step} | valid_mIoU="
                f"{100*initial_metrics['valid_miou']:.4f}",
                flush=True,
            )
        if args.save_secondary_best:
            for selector in SECONDARY_BEST_SELECTORS:
                if selector in improved:
                    atomic_save(
                        weights_payload(
                            model,
                            step,
                            initial_metrics,
                            provenance,
                            args,
                            selector=selector,
                        ),
                        output_dir / f"best_{selector}.pth",
                    )
    if args.validation_only or args.optimizer_steps == 0:
        return

    model.train()
    keep_frozen_modules_in_eval(model)
    optimizer.zero_grad(set_to_none=True)
    running: Dict[str, float] = {}
    micro_step = 0
    iterator = iter(train_loader)
    window_start = time.time()
    while step < args.optimizer_steps:
        try:
            rgb, sar, location, damage, sample_names = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            rgb, sar, location, damage, sample_names = next(iterator)
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(rgb, sar, return_dict=True)
            if args.damage_ce_weight == 0.0 and args.damage_lovasz_weight == 0.0:
                # Selector-only descendants must not silently optimise a
                # segmentation objective through their routed output.
                zero_reference = outputs.get(ZERO_LOSS_REFERENCE_KEY, outputs["output_dam"])
                damage_loss = zero_reference.float().sum() * 0.0
                damage_ce = damage_loss.detach()
                damage_lovasz = damage_loss.detach()
            else:
                damage_loss, damage_ce, damage_lovasz = segmentation_loss(
                    outputs["output_dam"],
                    damage,
                    args.ignore_index,
                    args.damage_ce_weight,
                    args.damage_lovasz_weight,
                )
            loc_monitor, loc_ce, loc_lovasz = segmentation_loss(
                outputs["output_loc"],
                location,
                args.ignore_index,
                1.0,
                0.5,
            )
            sar_aux_loss = F.cross_entropy(
                outputs["sar_aux_logits"].float(),
                damage,
                ignore_index=args.ignore_index,
            )
            representation = common_unique_regularization(
                outputs, damage, args.ignore_index
            )
            flow_losses = flow_regularization(
                outputs, damage, args.ignore_index
            )
            hierarchy = hierarchical_damage_loss(
                outputs,
                damage,
                priors,
                args.ignore_index,
                args.prior_strength,
            )
            extra_supervised = {}
            if callable(EXTRA_SUPERVISED_LOSS):
                if EXTRA_SUPERVISED_USES_NAMES:
                    extra_supervised = EXTRA_SUPERVISED_LOSS(
                        outputs, damage, args.ignore_index, sample_names
                    )
                else:
                    extra_supervised = EXTRA_SUPERVISED_LOSS(
                        outputs, damage, args.ignore_index
                    )
                if "extra_supervised_loss" not in extra_supervised:
                    raise RuntimeError(
                        "EXTRA_SUPERVISED_LOSS must return extra_supervised_loss"
                    )
            extra_supervised_loss = extra_supervised.get(
                "extra_supervised_loss", damage_loss.new_zeros(())
            )
            total_loss = (
                damage_loss
                + args.sar_aux_weight * sar_aux_loss
                + args.representation_weight * representation["representation_loss"]
                + args.flow_regularization_weight
                * flow_losses["flow_regularization_loss"]
                + args.hierarchy_weight * hierarchy["hierarchy_loss"]
                + args.extra_supervised_weight * extra_supervised_loss
            )
            scaled_loss = total_loss / args.accumulation_steps
        values = {
            "total_loss": total_loss,
            "loc_loss_monitor_only": loc_monitor,
            "loc_ce_loss_monitor_only": loc_ce,
            "loc_lovasz_loss_monitor_only": loc_lovasz,
            "damage_loss": damage_loss,
            "damage_ce_loss": damage_ce,
            "damage_lovasz_loss": damage_lovasz,
            "sar_aux_loss": sar_aux_loss,
            **representation,
            **flow_losses,
            **hierarchy,
            **extra_supervised,
        }
        if not torch.isfinite(total_loss):
            finite_diagnostics = {
                key: float(value.detach().float().item())
                for key, value in values.items()
            }
            for stage_index in range(1, 5):
                flow = outputs.get(f"stage{stage_index}_flow")
                if flow is not None:
                    finite_diagnostics[
                        f"stage{stage_index}_flow_finite"
                    ] = bool(torch.isfinite(flow).all().item())
                    finite_diagnostics[
                        f"stage{stage_index}_flow_abs_max"
                    ] = float(
                        torch.nan_to_num(flow.detach().float())
                        .abs()
                        .max()
                        .item()
                    )
            print(
                "Non-finite diagnostics: "
                + json.dumps(finite_diagnostics, allow_nan=True),
                flush=True,
            )
            raise FloatingPointError(
                f"non-finite H4 loss before optimizer step {step + 1}"
            )
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        for key, value in values.items():
            running[key] = running.get(key, 0.0) + float(value.detach().item())
        if micro_step % args.accumulation_steps:
            continue

        suppress_warmup_gradients(model, step, args.new_module_warmup_steps)
        scaler.unscale_(optimizer)
        parameters_with_grad = [
            p for p in model.parameters() if p.grad is not None
        ]
        nn.utils.clip_grad_norm_(
            parameters_with_grad,
            5.0,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        nonfinite_parameters = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not torch.isfinite(parameter.detach()).all()
        ]
        if nonfinite_parameters:
            raise FloatingPointError(
                "non-finite trainable parameters after optimizer step "
                f"{step + 1}: {nonfinite_parameters}"
            )
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            denominator = args.log_every * args.accumulation_steps
            elapsed = max(time.time() - window_start, 1e-6)
            payload = {
                "step": step,
                "variant": args.variant,
                "phase": (
                    "new_modules_only"
                    if step <= args.new_module_warmup_steps
                    else "joint_damage_adaptation"
                ),
                **{
                    key: value / denominator
                    for key, value in running.items()
                },
                "optimizer_steps_per_second": args.log_every / elapsed,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            print(json.dumps(payload, allow_nan=False), flush=True)
            running = {}
            window_start = time.time()

        should_evaluate = (
            step % args.eval_every == 0 or step == args.optimizer_steps
        )
        if should_evaluate:
            metrics = evaluate(
                model, val_loader, device, args.ignore_index, args.val_max_batches
            )
            print(metrics_line("Validation", metrics), flush=True)
            improved = _update_selection_records(
                selection_records, metrics, step
            )
            if "valid" in improved:
                best_metrics = dict(metrics)
                atomic_save(
                    weights_payload(
                        model,
                        step,
                        metrics,
                        provenance,
                        args,
                        selector="valid",
                    ),
                    output_dir / "best_val_miou.pth",
                )
                print(
                    f"New best valid_mIoU={100*metrics['valid_miou']:.4f}",
                    flush=True,
                )
            if args.save_secondary_best:
                for selector in SECONDARY_BEST_SELECTORS:
                    if selector in improved:
                        atomic_save(
                            weights_payload(
                                model,
                                step,
                                metrics,
                                provenance,
                                args,
                                selector=selector,
                            ),
                            output_dir / f"best_{selector}.pth",
                        )
            (output_dir / "latest_validation.json").write_text(
                json.dumps({"step": step, "metrics": metrics}, indent=2) + "\n",
                encoding="utf-8",
            )
            if test_loader is not None:
                test_metrics = evaluate(
                    model,
                    test_loader,
                    device,
                    args.ignore_index,
                    args.test_max_batches,
                )
                print(
                    metrics_line(
                        "Test monitor (not selector)", test_metrics
                    ),
                    flush=True,
                )
                test_monitor_history.append(
                    {
                        "step": int(step),
                        "phase": "post_validation",
                        "selector": False,
                        "validation_metrics": dict(metrics),
                        "metrics": dict(test_metrics),
                    }
                )
                test_history_path.write_text(
                    json.dumps(test_monitor_history, indent=2) + "\n",
                    encoding="utf-8",
                )
                (output_dir / "latest_test_monitor.json").write_text(
                    json.dumps(test_monitor_history[-1], indent=2) + "\n",
                    encoding="utf-8",
                )

        if step % args.save_every == 0 or step == args.optimizer_steps:
            atomic_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_metrics,
                    selection_records,
                    provenance,
                    args,
                ),
                output_dir / "latest_resume.pth",
            )

    summary = {
        "protocol": "H4_RGBSAR_training_summary_v1",
        "variant": args.variant,
        "completed_step": step,
        "best_valid_miou": best_metrics.get("valid_miou"),
        "best_metrics": best_metrics,
        "selection_records": selection_records,
        "provenance": provenance,
        "test_accessed": bool(args.test_every_validation),
        "test_monitor_policy": (
            "after_every_validation_log_only_never_selector"
            if args.test_every_validation
            else "not_accessed"
        ),
        "test_monitor_history": test_monitor_history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Training complete. Best valid_mIoU="
        f"{100*best_metrics.get('valid_miou', 0.0):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
