# -*- coding: utf-8 -*-
"""Train-only event x class Memory queue for CMR hard-pair mining.

The queue is training infrastructure, not an inference module.  Entries are
detached teacher features and are partitioned by disaster event and class so a
query can never silently draw a same-event positive.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("EMA momentum must be in [0,1)")
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    if teacher_parameters.keys() != student_parameters.keys():
        raise ValueError("teacher/student parameter contracts differ")
    for name, target in teacher_parameters.items():
        target.mul_(float(momentum)).add_(student_parameters[name], alpha=1.0 - float(momentum))
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, target in teacher_buffers.items():
        if name in student_buffers:
            target.copy_(student_buffers[name])


class CMREventClassQueue(nn.Module):
    """Fixed-capacity FIFO queue with one independent cell per event/class."""

    def __init__(
        self,
        feature_dim: int,
        num_events: int,
        *,
        num_classes: int = 3,
        capacity_per_cell: int = 200,
    ) -> None:
        super().__init__()
        if min(feature_dim, num_events, num_classes, capacity_per_cell) <= 0:
            raise ValueError("queue dimensions and capacity must be positive")
        shape = (num_classes, num_events, capacity_per_cell)
        self.feature_dim = int(feature_dim)
        self.num_events = int(num_events)
        self.num_classes = int(num_classes)
        self.capacity_per_cell = int(capacity_per_cell)
        self.register_buffer("features", torch.zeros(*shape, feature_dim))
        self.register_buffer("confidence", torch.zeros(*shape))
        self.register_buffer("valid", torch.zeros(*shape, dtype=torch.bool))
        self.register_buffer("write_pointer", torch.zeros(num_classes, num_events, dtype=torch.long))
        self.register_buffer("total_seen", torch.zeros(num_classes, num_events, dtype=torch.long))

    @torch.no_grad()
    def enqueue(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        events: torch.Tensor,
        *,
        confidence: torch.Tensor | None = None,
        quality_mask: torch.Tensor | None = None,
    ) -> int:
        feature = F.normalize(features.detach().float(), dim=1, eps=1e-8).to(self.features.device)
        label = labels.detach().long().flatten().to(self.features.device)
        event = events.detach().long().flatten().to(self.features.device)
        if feature.ndim != 2 or feature.shape[1] != self.feature_dim:
            raise ValueError("queue features have the wrong shape")
        if not (feature.shape[0] == label.numel() == event.numel()):
            raise ValueError("queue rows/labels/events do not align")
        if bool(((label < 0) | (label >= self.num_classes)).any()):
            raise ValueError("queue label outside declared classes")
        if bool(((event < 0) | (event >= self.num_events)).any()):
            raise ValueError("queue event outside declared events")
        score = (
            torch.ones(label.numel(), device=self.features.device)
            if confidence is None
            else confidence.detach().float().flatten().to(self.features.device)
        )
        keep = (
            torch.ones(label.numel(), dtype=torch.bool, device=self.features.device)
            if quality_mask is None
            else quality_mask.detach().bool().flatten().to(self.features.device)
        )
        if score.numel() != label.numel() or keep.numel() != label.numel():
            raise ValueError("confidence/quality mask rows do not align")

        written = 0
        for row in torch.nonzero(keep, as_tuple=False).flatten().tolist():
            cls, evt = int(label[row]), int(event[row])
            slot = int(self.write_pointer[cls, evt])
            self.features[cls, evt, slot].copy_(feature[row])
            self.confidence[cls, evt, slot] = score[row]
            self.valid[cls, evt, slot] = True
            self.write_pointer[cls, evt] = (slot + 1) % self.capacity_per_cell
            self.total_seen[cls, evt] += 1
            written += 1
        return written

    def class_entries(self, class_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls = int(class_index)
        if not 0 <= cls < self.num_classes:
            raise ValueError("class index outside queue")
        valid = self.valid[cls]
        event_grid = torch.arange(self.num_events, device=valid.device)[:, None].expand_as(valid)
        return self.features[cls][valid], event_grid[valid], self.confidence[cls][valid]

    @torch.no_grad()
    def retrieve_cross_event(
        self,
        query_features: torch.Tensor,
        query_events: torch.Tensor,
        *,
        class_index: int,
        topk: int,
        maximum_per_event: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """Retrieve class-specific supports while excluding the query event."""

        query = F.normalize(query_features.detach().float(), dim=1, eps=1e-8).to(self.features.device)
        query_event = query_events.detach().long().flatten().to(self.features.device)
        support, support_event, support_confidence = self.class_entries(class_index)
        if support.numel() == 0:
            raise RuntimeError(f"queue class {class_index} is empty")
        if query.shape[0] != query_event.numel():
            raise ValueError("query features/events do not align")
        count = int(topk)
        output_score = torch.full((query.shape[0], count), float("-inf"), device=query.device)
        output_event = torch.full((query.shape[0], count), -1, dtype=torch.long, device=query.device)
        output_index = torch.full((query.shape[0], count), -1, dtype=torch.long, device=query.device)
        similarities = query @ support.t()
        similarities.masked_fill_(query_event[:, None] == support_event[None, :], float("-inf"))
        order = similarities.argsort(dim=1, descending=True, stable=True)
        for row in range(query.shape[0]):
            per_event: Dict[int, int] = {}
            selected = []
            for index in order[row].tolist():
                if not torch.isfinite(similarities[row, index]):
                    continue
                event = int(support_event[index])
                if per_event.get(event, 0) >= int(maximum_per_event):
                    continue
                selected.append(index)
                per_event[event] = per_event.get(event, 0) + 1
                if len(selected) == count:
                    break
            if selected:
                index_tensor = torch.tensor(selected, device=query.device)
                take = len(selected)
                output_score[row, :take] = similarities[row, index_tensor]
                output_event[row, :take] = support_event[index_tensor]
                output_index[row, :take] = index_tensor
        return {
            "indices": output_index.cpu(),
            "scores": output_score.cpu(),
            "events": output_event.cpu(),
            "valid": (output_index >= 0).cpu(),
            "support_confidence": support_confidence.cpu(),
        }

