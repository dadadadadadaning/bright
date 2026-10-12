# -*- coding: utf-8 -*-
"""Deterministic event-capped schedules for CMR-H0B and later H1 training."""
from __future__ import annotations

from collections import Counter
from typing import Dict

import torch


def event_anchor_pools(anchor_events: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Map each event code to row positions in the H0 pair tensors."""

    events = anchor_events.long().flatten().cpu()
    if events.numel() == 0:
        raise ValueError("anchor_events cannot be empty")
    return {
        int(event): torch.nonzero(events == int(event), as_tuple=False).flatten()
        for event in sorted(torch.unique(events).tolist())
    }


def _fixed_event_permutation(
    pool: torch.Tensor,
    *,
    seed: int,
    event: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        int(seed) + 1009 * (int(event) + 1)
    )
    return pool[torch.randperm(pool.numel(), generator=generator)]


def build_event_capped_epoch(
    anchor_events: torch.Tensor,
    *,
    cap_per_event: int = 64,
    seed: int = 3407,
    epoch: int = 0,
) -> torch.Tensor:
    """Create one no-replacement epoch with deterministic large-pool rotation.

    Small events contribute every available anchor exactly once.  Large events
    contribute ``cap_per_event`` rows.  Their fixed permutation is traversed in
    circular blocks across epochs, so discarded rows rotate into later epochs.
    The returned values are row positions in H0's anchor/pair arrays.
    """

    if int(cap_per_event) <= 0 or int(epoch) < 0:
        raise ValueError("cap_per_event must be positive and epoch nonnegative")
    selected = []
    for event, pool in event_anchor_pools(anchor_events).items():
        permutation = _fixed_event_permutation(pool, seed=seed, event=event)
        count = int(permutation.numel())
        take = min(count, int(cap_per_event))
        if count <= int(cap_per_event):
            rows = permutation
        else:
            start = (int(epoch) * int(cap_per_event)) % count
            end = start + take
            if end <= count:
                rows = permutation[start:end]
            else:
                rows = torch.cat((permutation[start:], permutation[: end - count]))
        if torch.unique(rows).numel() != rows.numel():
            raise AssertionError("event schedule repeated an anchor within an epoch")
        selected.append(rows)
    rows = torch.cat(selected)
    shuffle_generator = torch.Generator().manual_seed(
        int(seed) + 104729 * (int(epoch) + 1)
    )
    rows = rows[torch.randperm(rows.numel(), generator=shuffle_generator)]
    if torch.unique(rows).numel() != rows.numel():
        raise AssertionError("epoch schedule contains duplicate anchors")
    return rows


def event_occurrence_statistics(
    labels: torch.Tensor,
    base_predictions: torch.Tensor,
    events: torch.Tensor,
    event_map: Dict[str, int],
    anchor_indices: torch.Tensor,
    *,
    cap_per_event: int,
) -> Dict[str, object]:
    """Report per-event Damaged denominators, P3 D->I errors and capped shares."""

    labels = labels.long().flatten().cpu()
    base_predictions = base_predictions.long().flatten().cpu()
    events = events.long().flatten().cpu()
    anchors = anchor_indices.long().flatten().cpu()
    if not (labels.shape == base_predictions.shape == events.shape):
        raise ValueError("labels, predictions and events must align")
    if not bool((labels[anchors] == 1).all()) or not bool((base_predictions[anchors] == 0).all()):
        raise RuntimeError("H0B anchors must be GT Damaged and frozen-P3 Intact")
    inverse = {int(code): str(name) for name, code in event_map.items()}
    raw_counts = Counter(events[anchors].tolist())
    effective_counts = {
        event: min(count, int(cap_per_event)) for event, count in raw_counts.items()
    }
    raw_total = max(sum(raw_counts.values()), 1)
    effective_total = max(sum(effective_counts.values()), 1)
    per_event = {}
    for event in sorted(int(item) for item in torch.unique(events).tolist()):
        event_mask = events == event
        damaged_total = int((event_mask & (labels == 1)).sum())
        missed = int(raw_counts.get(event, 0))
        effective = int(effective_counts.get(event, 0))
        per_event[inverse.get(event, str(event))] = {
            "event_code": event,
            "damaged_regions": damaged_total,
            "damaged_to_intact": missed,
            "damaged_to_intact_rate": missed / max(damaged_total, 1),
            "raw_anchor_share": missed / raw_total,
            "effective_epoch_anchors": effective,
            "effective_epoch_share": effective / effective_total,
        }
    return {
        "per_event": per_event,
        "raw_anchor_count": int(sum(raw_counts.values())),
        "effective_epoch_anchor_count": int(sum(effective_counts.values())),
        "raw_maximum_event_share": max(raw_counts.values()) / raw_total,
        "effective_maximum_event_share": max(effective_counts.values()) / effective_total,
        "effective_training_events": sum(count > 0 for count in effective_counts.values()),
        "cap_per_event": int(cap_per_event),
    }


def audit_event_capped_schedules(
    anchor_events: torch.Tensor,
    *,
    cap_per_event: int,
    seed: int,
    audit_epochs: int,
) -> Dict[str, object]:
    """Simulate schedules and verify no within-epoch repetition or event drift."""

    if int(audit_epochs) <= 0:
        raise ValueError("audit_epochs must be positive")
    events = anchor_events.long().flatten().cpu()
    epoch_summaries = []
    union = set()
    all_unique = True
    maximum_share = 0.0
    expected_size = None
    preview = None
    for epoch in range(int(audit_epochs)):
        rows = build_event_capped_epoch(
            events,
            cap_per_event=cap_per_event,
            seed=seed,
            epoch=epoch,
        )
        if preview is None:
            preview = rows
        unique = torch.unique(rows).numel() == rows.numel()
        all_unique = all_unique and bool(unique)
        counts = Counter(events[rows].tolist())
        size = int(rows.numel())
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise AssertionError("event-capped epoch size changed across rotations")
        share = max(counts.values()) / max(size, 1)
        maximum_share = max(maximum_share, share)
        union.update(rows.tolist())
        epoch_summaries.append({
            "epoch": epoch,
            "anchors": size,
            "events": len(counts),
            "maximum_event_share": share,
            "no_replacement_within_epoch": bool(unique),
        })
    return {
        "audit_epochs": int(audit_epochs),
        "anchors_per_epoch": int(expected_size or 0),
        "events_per_epoch": int(len(torch.unique(events))),
        "maximum_event_share_across_epochs": float(maximum_share),
        "no_replacement_within_every_epoch": bool(all_unique),
        "unique_anchors_seen_across_audit_epochs": int(len(union)),
        "unique_anchor_coverage_across_audit_epochs": len(union) / max(events.numel(), 1),
        "epoch_summaries": epoch_summaries,
        "preview_epoch_rows": preview,
    }


def h0b_gate(
    core_h0_checks: Dict[str, bool],
    occurrence: Dict[str, object],
    schedule: Dict[str, object],
) -> Dict[str, bool]:
    """H0B gate: preserve H0 pair quality and repair only training distribution."""

    return {
        "all_core_h0_pair_checks_pass": all(bool(value) for value in core_h0_checks.values()),
        "effective_training_events_at_least_10": int(occurrence["effective_training_events"]) >= 10,
        "effective_maximum_event_share_at_most_0_15": float(occurrence["effective_maximum_event_share"]) <= 0.15,
        "same_epoch_anchor_repetition_is_zero": bool(schedule["no_replacement_within_every_epoch"]),
        "epoch_size_is_positive": int(schedule["anchors_per_epoch"]) > 0,
    }
