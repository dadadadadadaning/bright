# -*- coding: utf-8 -*-
"""Shared, side-effect-free runtime for the isolated AL0--AL3 protocol."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DamageFormer_RGBSAR_Heterogeneous import (
    NativeSARPyramid,
    SwinRGBPyramid,
)
from model.DamageFormer_S7_SIRM_M2B import connected_components_4
from script.standard_ML.im4cd_matching_runtime import binary_auroc
from script.standard_ML.rm_region_retrieval_runtime import event_from_name


AL_PROTOCOL = "intact_anchored_alignment_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(checkpoint: object) -> MutableMapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    candidate = checkpoint.get("model", checkpoint)
    if not isinstance(candidate, Mapping):
        raise TypeError("checkpoint has no model state mapping")
    state: MutableMapping[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[7:]
        state[name] = value
    return state


def _load_prefix_exact(module: nn.Module, state: Mapping[str, torch.Tensor], prefix: str) -> int:
    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    target = module.state_dict()
    missing_source = sorted(set(target) - set(selected))
    wrong_shape = sorted(
        key for key in target if key in selected and target[key].shape != selected[key].shape
    )
    unexpected = sorted(set(selected) - set(target))
    if missing_source or wrong_shape or unexpected:
        raise RuntimeError(
            f"unsafe H6 encoder load for {prefix}: missing={missing_source[:5]}, "
            f"wrong_shape={wrong_shape[:5]}, unexpected={unexpected[:5]}"
        )
    module.load_state_dict(selected, strict=True)
    return len(selected)


class FrozenH6Stage23(nn.Module):
    """Read only the locked H6-A2 RGB/SAR Stage2 and Stage3 encoders."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if str(checkpoint.get("variant", "")).lower() != "a2":
            raise RuntimeError("AL requires the locked H6-A2 checkpoint")
        state = checkpoint_state(checkpoint)
        self.rgb_encoder = SwinRGBPyramid(pretrained=False)
        self.sar_encoder = NativeSARPyramid()
        self.loaded = {
            "rgb": _load_prefix_exact(self.rgb_encoder, state, "rgb_encoder."),
            "sar": _load_prefix_exact(self.sar_encoder, state, "sar_encoder."),
        }
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        # A caller cannot accidentally enable stochastic/frozen encoder state.
        return super().train(False)

    @torch.no_grad()
    def forward(
        self, rgb: torch.Tensor, native_sar: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        rgb_values = self.rgb_encoder(rgb)
        sar_values = self.sar_encoder(native_sar[:, :1])
        if len(rgb_values) != 4 or len(sar_values) != 4:
            raise RuntimeError("H6 encoders must return four stages")
        return (
            {"stage2": rgb_values[1], "stage3": rgb_values[2]},
            {"stage2": sar_values[1], "stage3": sar_values[2]},
        )


def alignment_checkpoint_payload(
    model: nn.Module,
    *,
    step: int,
    h6_source: FrozenH6Stage23,
    metrics: Mapping[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "protocol": AL_PROTOCOL,
        "stage": "al1",
        "step": int(step),
        "model": model.state_dict(),
        "metrics": dict(metrics),
        "h6_checkpoint": h6_source.checkpoint_path,
        "h6_checkpoint_sha256": h6_source.checkpoint_sha256,
        "test_accessed": False,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def load_alignment_checkpoint(
    model: nn.Module,
    path: str | Path,
    h6_source: FrozenH6Stage23,
    *,
    allow_occurrence_missing: bool = False,
) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != AL_PROTOCOL or checkpoint.get("stage") != "al1":
        raise RuntimeError("checkpoint is not an AL1 alignment checkpoint")
    if checkpoint.get("h6_checkpoint_sha256") != h6_source.checkpoint_sha256:
        raise RuntimeError("AL1 checkpoint was trained with another H6 source")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected AL1 tensors: {unexpected}")
    if allow_occurrence_missing:
        illegal = [key for key in missing if not key.startswith("occurrence_head.")]
        if illegal:
            raise RuntimeError(f"unsafe AL1 load, missing={illegal}")
    elif missing:
        raise RuntimeError(f"incomplete AL1 checkpoint, missing={missing}")
    return checkpoint


def _region_records(
    score: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
    *,
    min_area: int = 32,
    min_purity: float = 0.75,
) -> List[Dict[str, object]]:
    score = F.interpolate(score.float(), target.shape[-2:], mode="bilinear", align_corners=False)
    records: List[Dict[str, object]] = []
    for batch_index in range(target.shape[0]):
        label = target[batch_index].cpu()
        building = ((label >= 1) & (label <= 3)).numpy()
        for component in connected_components_4(building):
            if int(component.sum()) < int(min_area):
                continue
            mask = torch.from_numpy(component.astype(np.bool_, copy=False))
            values = label[mask]
            valid = (values >= 1) & (values <= 3)
            if int(valid.sum()) == 0:
                continue
            counts = torch.bincount(values[valid].long(), minlength=4)[1:4]
            purity = float(counts.max() / counts.sum().clamp_min(1))
            if purity < float(min_purity):
                continue
            records.append(
                {
                    "score": float(score[batch_index, 0].detach().cpu()[mask].mean()),
                    "label": int(counts.argmax()) + 1,
                    "event": event_from_name(str(names[batch_index])),
                    "area": int(component.sum()),
                }
            )
    return records


def _auc_summary(records: Sequence[Mapping[str, object]], positive_labels: Sequence[int]) -> Dict[str, object]:
    positive_set = set(int(value) for value in positive_labels)
    selected = [row for row in records if int(row["label"]) == 1 or int(row["label"]) in positive_set]
    labels = torch.tensor(
        [int(int(row["label"]) in positive_set) for row in selected], dtype=torch.long
    )
    scores = torch.tensor([float(row["score"]) for row in selected], dtype=torch.float32)
    global_auc = binary_auroc(labels, scores) if labels.unique().numel() == 2 else float("nan")
    events: Dict[str, float] = {}
    for event in sorted({str(row["event"]) for row in selected}):
        indices = [index for index, row in enumerate(selected) if str(row["event"]) == event]
        event_labels = labels[indices]
        if event_labels.unique().numel() != 2:
            continue
        events[event] = binary_auroc(event_labels, scores[indices])
    return {
        "global_auroc": float(global_auc),
        "event_macro_auroc": (
            float(sum(events.values()) / len(events)) if events else float("nan")
        ),
        "comparable_events": len(events),
        "direction_consistent_events": sum(value >= 0.5 for value in events.values()),
        "event_auroc": events,
        "regions": len(selected),
        "positive_regions": int(labels.sum()),
    }


def evidence_audit(
    aligned_records: Sequence[Mapping[str, object]],
    unaligned_records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    damaged = _auc_summary(aligned_records, (2,))
    damaged_base = _auc_summary(unaligned_records, (2,))
    affected = _auc_summary(aligned_records, (2, 3))
    affected_base = _auc_summary(unaligned_records, (2, 3))
    gain = float(damaged["global_auroc"]) - float(damaged_base["global_auroc"])
    checks = {
        "global_damaged_auroc_at_least_0_65": damaged["global_auroc"] >= 0.65,
        "event_damaged_auroc_at_least_0_60": damaged["event_macro_auroc"] >= 0.60,
        "ten_events_direction_consistent": damaged["direction_consistent_events"] >= 10,
        "aligned_plus_0_05_vs_unaligned": gain >= 0.05,
    }
    return {
        "damaged_vs_intact": damaged,
        "unaligned_damaged_vs_intact": damaged_base,
        "affected_vs_intact": affected,
        "unaligned_affected_vs_intact": affected_base,
        "aligned_gain_global_auroc": gain,
        "checks": checks,
        "passed": all(checks.values()),
    }


def scene_relative_residual(
    residual: torch.Tensor,
    building_probability: torch.Tensor,
    *,
    probability_threshold: float = 0.50,
    minimum_support: int = 16,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Robustly standardise each scene without using target labels.

    The reference population is selected only by a frozen P3 building
    probability.  When a crop contains too few thresholded pixels, the
    highest-probability pixels form a deterministic fallback support.  This
    keeps AL0-N defined for sparse scenes without consulting validation GT.
    """

    if residual.ndim != 4 or residual.shape[1] != 1:
        raise ValueError("residual must have shape [B,1,H,W]")
    if building_probability.ndim != 4 or building_probability.shape[1] != 1:
        raise ValueError("building_probability must have shape [B,1,H,W]")
    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("probability_threshold must lie in [0,1]")
    if int(minimum_support) < 1:
        raise ValueError("minimum_support must be positive")

    probability = F.interpolate(
        building_probability.float(), residual.shape[-2:],
        mode="bilinear", align_corners=False,
    )
    normalised = []
    centres = []
    scales = []
    support_counts = []
    for index in range(residual.shape[0]):
        value = residual[index, 0].float()
        confidence = probability[index, 0]
        support = confidence >= float(probability_threshold)
        required = min(int(minimum_support), int(value.numel()))
        if int(support.sum()) < required:
            chosen = confidence.flatten().topk(required, largest=True, sorted=False).indices
            support = torch.zeros_like(confidence, dtype=torch.bool).flatten()
            support[chosen] = True
            support = support.reshape_as(confidence)
        reference = value[support]
        centre = reference.median()
        mad = (reference - centre).abs().median()
        robust_scale = 1.4826 * mad
        # A constant/quantised support has zero MAD.  Std is a deterministic
        # fallback; eps is only the final numerical guard.
        scale = torch.where(
            robust_scale > float(eps),
            robust_scale,
            reference.std(unbiased=False).clamp_min(float(eps)),
        )
        normalised.append(((value - centre) / scale).unsqueeze(0))
        centres.append(centre)
        scales.append(scale)
        support_counts.append(support.sum().to(dtype=torch.float32))
    return {
        "normalised": torch.stack(normalised, dim=0),
        "centre": torch.stack(centres),
        "scale": torch.stack(scales),
        "support_count": torch.stack(support_counts),
    }


def occurrence_evidence_audit(
    candidate_records: Sequence[Mapping[str, object]],
    baseline_records: Sequence[Mapping[str, object]],
    *,
    minimum_event_gain: float = 0.02,
) -> Dict[str, object]:
    """AL0-N gate aligned with the downstream Intact/Affected task."""

    affected = _auc_summary(candidate_records, (2, 3))
    affected_base = _auc_summary(baseline_records, (2, 3))
    damaged = _auc_summary(candidate_records, (2,))
    damaged_base = _auc_summary(baseline_records, (2,))
    event_gain = (
        float(affected["event_macro_auroc"])
        - float(affected_base["event_macro_auroc"])
    )
    checks = {
        "global_affected_auroc_at_least_0_65": affected["global_auroc"] >= 0.65,
        "event_affected_auroc_at_least_0_60": affected["event_macro_auroc"] >= 0.60,
        "ten_events_direction_consistent": affected["direction_consistent_events"] >= 10,
        "event_affected_plus_0_02_vs_raw": event_gain >= float(minimum_event_gain),
    }
    return {
        "affected_vs_intact": affected,
        "raw_affected_vs_intact": affected_base,
        "damaged_vs_intact_diagnostic": damaged,
        "raw_damaged_vs_intact_diagnostic": damaged_base,
        "affected_event_gain": event_gain,
        "minimum_event_gain": float(minimum_event_gain),
        "checks": checks,
        "passed": all(checks.values()),
    }


def binary_occurrence_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
) -> Dict[str, object]:
    logits = F.interpolate(logits.float(), target.shape[-2:], mode="bilinear", align_corners=False)
    prediction = logits.argmax(1).cpu()
    target = target.cpu()
    valid = (target >= 1) & (target <= 3)
    truth = (target >= 2) & (target <= 3)

    def one(mask: torch.Tensor) -> Dict[str, float]:
        y = truth[mask].long()
        p = prediction[mask].long()
        confusion = torch.zeros((2, 2), dtype=torch.long)
        for actual in range(2):
            for guessed in range(2):
                confusion[actual, guessed] = ((y == actual) & (p == guessed)).sum()
        f1s = []
        recalls = []
        for cls in range(2):
            tp = float(confusion[cls, cls])
            fp = float(confusion[:, cls].sum() - confusion[cls, cls])
            fn = float(confusion[cls, :].sum() - confusion[cls, cls])
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1.0))
            recalls.append(tp / max(tp + fn, 1.0))
        return {
            "macro_f1": sum(f1s) / 2.0,
            "affected_f1": f1s[1],
            "affected_recall": recalls[1],
        }

    overall = one(valid)
    event_rows = {}
    for index, name in enumerate(names):
        event = event_from_name(str(name))
        event_rows.setdefault(event, []).append(index)
    per_event = {
        event: one(valid[indices]) for event, indices in event_rows.items()
    }
    overall["event_macro_f1"] = sum(
        row["macro_f1"] for row in per_event.values()
    ) / max(len(per_event), 1)
    overall["events"] = per_event
    return overall


__all__ = [
    "AL_PROTOCOL",
    "FrozenH6Stage23",
    "alignment_checkpoint_payload",
    "binary_occurrence_metrics",
    "evidence_audit",
    "occurrence_evidence_audit",
    "scene_relative_residual",
    "file_sha256",
    "load_alignment_checkpoint",
    "_region_records",
]
