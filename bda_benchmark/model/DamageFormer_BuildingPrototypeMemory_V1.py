from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    event_name, get_variant, multimodal_descriptor,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


class DamageFormerBuildingPrototypeMemoryV1(nn.Module):
    """Frozen A0 plus a small, confidence-gated train-only prototype reader."""

    def __init__(self, variant: str, hidden_dim: int = 32, topk: int = 5, max_logit_shift: float = 1.0):
        super().__init__()
        self.variant = variant
        self.spec = get_variant(variant)
        self.topk = int(topk)
        self.max_logit_shift = float(max_logit_shift)
        self.parent = DamageFormer_S8_MICB_DSMIL(
            roi_alpha=1.0, detach_building_prior=True, token_dim=128,
            mil_topk=8, max_tokens_per_bag=256, mil_residual_scale=0.0,
        )
        self.head = nn.Sequential(
            nn.Conv2d(10, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, 1, 1),
        )
        self.gain = nn.Parameter(torch.zeros(()))
        self.projection = nn.Linear(64, 64, bias=False)
        nn.init.eye_(self.projection.weight)
        self.register_buffer("memory_features", torch.empty(0, 64), persistent=False)
        self.register_buffer("memory_labels", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("memory_event_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pca_mean", torch.empty(0), persistent=False)
        self.register_buffer("pca_components", torch.empty(0, 64), persistent=False)
        self.event_to_id: dict[str, int] = {}
        self.memory_report: dict[str, object] = {}

    def load_parent(self, checkpoint: str) -> dict[str, object]:
        return load_parent(self.parent, checkpoint)

    def load_memory(self, path: str) -> dict[str, object]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        device = next(self.parameters()).device
        features = F.normalize(payload["features"].float(), dim=1).to(device)
        pca_mean = payload["projection_mean"].float().to(device)
        pca_components = payload["projection_components"].float().to(device)
        if features.shape[1] != 64 or pca_components.shape[1] != 64:
            raise RuntimeError("V1 requires a 64-dimensional train-only projection")
        labels = payload["labels"].long().to(device)
        if self.spec["shuffle_labels"]:
            labels = labels.remainder(3) + 1
        events = [str(value) for value in payload["events"]]
        vocabulary = sorted(set(events))
        self.event_to_id = {name: index for index, name in enumerate(vocabulary)}
        event_ids = torch.tensor(
            [self.event_to_id[name] for name in events], dtype=torch.long, device=device
        )
        self.memory_features = features
        self.memory_labels = labels
        self.memory_event_ids = event_ids
        self.pca_mean = pca_mean
        self.pca_components = pca_components
        self.memory_report = dict(payload.get("metadata", {}))
        self.memory_report.update({"path": str(Path(path).resolve()), "entries": int(labels.numel())})
        return self.memory_report

    def freeze_parent(self) -> None:
        self.parent.eval()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def _scores(self, rgb: torch.Tensor, sar: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        descriptor = multimodal_descriptor(rgb, sar)
        b, _, h, w = descriptor.shape
        if not self.spec["use_memory"]:
            return descriptor.new_zeros((b, 3, h, w))
        if self.memory_features.numel() == 0:
            raise RuntimeError("memory-enabled variant has no loaded memory")
        raw = descriptor.flatten(2).transpose(1, 2)
        pca_query = (raw - self.pca_mean) @ self.pca_components
        query = F.normalize(self.projection(pca_query), dim=2)
        projected_bank = F.normalize(self.projection(self.memory_features), dim=1)
        output = query.new_full((b, h * w, 3), -1.0)
        for batch_index, name in enumerate(names):
            for class_index, label in enumerate((1, 2, 3)):
                keep = self.memory_labels == label
                if self.training and self.spec["exclude_event"]:
                    identifier = self.event_to_id.get(event_name(str(name)), -1)
                    if identifier >= 0:
                        keep = keep & (self.memory_event_ids != identifier)
                bank = projected_bank[keep]
                if bank.numel() == 0:
                    continue
                k = min(self.topk, int(bank.shape[0]))
                pieces = []
                for start in range(0, query.shape[1], 1024):
                    similarity = query[batch_index, start : start + 1024] @ bank.t()
                    pieces.append(similarity.topk(k, dim=1).values.mean(dim=1))
                output[batch_index, :, class_index] = torch.cat(pieces)
        return output.transpose(1, 2).reshape(b, 3, h, w).to(descriptor.dtype)

    def forward(self, pre: torch.Tensor, post: torch.Tensor, names: Optional[Sequence[str]] = None, return_aux: bool = False):
        if names is None:
            names = [""] * int(pre.shape[0])
        with torch.no_grad():
            rgb, sar = self.parent._encode_and_fuse(pre, post)
            loc_feature = self.parent.clf_1(rgb)
            p_building = torch.softmax(loc_feature, dim=1)[:, 1:2]
            fused = self.parent.fusion_layer_3(torch.cat((rgb, sar), dim=1))
            parent_logits = self.parent.clf_2(fused)
        probabilities = torch.softmax(parent_logits.float(), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-7).log()).sum(dim=1, keepdim=True) / 1.38629436112
        top2 = probabilities.topk(2, dim=1).values
        parent_margin = top2[:, 0:1] - top2[:, 1:2]
        scores = self._scores(rgb, sar, names).float()
        score_top2 = scores.topk(2, dim=1).values
        memory_confidence = torch.sigmoid((score_top2[:, 0:1] - score_top2[:, 1:2] - 0.02) / 0.05)
        if not self.spec["use_memory"]:
            memory_confidence = torch.ones_like(memory_confidence)
        head_input = torch.cat((probabilities, scores, entropy, parent_margin, p_building.float()), dim=1)
        tendency = scores[:, 1:3].amax(dim=1, keepdim=True) - scores[:, 0:1]
        raw = self.head(head_input) + tendency
        uncertainty_gate = torch.sigmoid((0.25 - parent_margin) / 0.05)
        gate = p_building.float() * uncertainty_gate * memory_confidence
        delta = torch.tanh(self.gain) * self.max_logit_shift * torch.tanh(raw.float()) * gate
        if not self.spec["train_head"]:
            delta = delta * 0.0
        adjusted = parent_logits.float().clone()
        adjusted[:, 2:4] += delta
        output_loc = F.interpolate(loc_feature, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        output_damage = F.interpolate(adjusted, size=post.shape[-2:], mode="bilinear", align_corners=False)
        if not return_aux:
            return output_loc, output_damage
        return output_loc, output_damage, {
            "parent_logits_feature": parent_logits,
            "adjusted_logits_feature": adjusted,
            "p_building": p_building,
            "memory_scores": scores,
            "memory_confidence": memory_confidence,
            "uncertainty": entropy,
            "uncertainty_gate": uncertainty_gate,
            "memory_tendency": tendency,
            "delta": delta,
            "gain": torch.tanh(self.gain),
            "fused_roi": fused,
        }
