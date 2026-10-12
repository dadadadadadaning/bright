from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def sample_event(sample_name: str) -> str:
    name = str(sample_name).strip()
    if not name or "_" not in name:
        raise ValueError(f"cannot infer event from sample name: {sample_name!r}")
    return name.rsplit("_", 1)[0]


def assert_event_disjoint(
    train_names: Iterable[str], validation_names: Iterable[str]
) -> tuple[set[str], set[str]]:
    train_events = {sample_event(value) for value in train_names}
    validation_events = {sample_event(value) for value in validation_names}
    overlap = train_events & validation_events
    if overlap:
        raise ValueError(f"event leakage detected: {sorted(overlap)}")
    return train_events, validation_events


def balanced_event_folds(
    sample_names: Sequence[str], fold_count: int = 4
) -> list[list[str]]:
    """Greedy event grouping; an event can appear in exactly one fold."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    by_event: dict[str, list[str]] = defaultdict(list)
    for name in sample_names:
        by_event[sample_event(name)].append(str(name))
    if len(by_event) < fold_count:
        raise ValueError("fewer events than requested folds")
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    sizes = [0] * fold_count
    for _, rows in sorted(by_event.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(fold_count), key=lambda index: (sizes[index], index))
        folds[target].extend(sorted(rows))
        sizes[target] += len(rows)
    return folds


@dataclass(frozen=True)
class CrossEventGate:
    macro_miou_gain_pp_min: float = 0.5
    damaged_f1_gain_pp_min: float = 1.0
    nonnegative_events_min: int = 10
    worst_event_gain_pp_min: float = -0.5
    bootstrap_ci_low_min: float = 0.0


def cross_event_decision(
    event_deltas: Mapping[str, Mapping[str, float]],
    *,
    bootstrap_ci_low: float,
    gate: CrossEventGate = CrossEventGate(),
) -> dict[str, object]:
    if not event_deltas:
        raise ValueError("event_deltas cannot be empty")
    required = {"miou_pp", "damaged_f1_pp"}
    for event, values in event_deltas.items():
        if not required.issubset(values):
            raise KeyError(f"event {event} lacks {sorted(required - set(values))}")
    miou = [float(values["miou_pp"]) for values in event_deltas.values()]
    damaged = [float(values["damaged_f1_pp"]) for values in event_deltas.values()]
    macro_miou = sum(miou) / len(miou)
    macro_damaged = sum(damaged) / len(damaged)
    nonnegative = sum(value >= 0.0 for value in miou)
    worst = min(miou)
    checks = {
        "macro_miou_gain": macro_miou >= gate.macro_miou_gain_pp_min,
        "macro_damaged_f1_gain": macro_damaged >= gate.damaged_f1_gain_pp_min,
        "nonnegative_event_count": nonnegative >= gate.nonnegative_events_min,
        "worst_event_guard": worst >= gate.worst_event_gain_pp_min,
        "bootstrap_ci_positive": float(bootstrap_ci_low) >= gate.bootstrap_ci_low_min,
    }
    return {
        "macro_miou_gain_pp": macro_miou,
        "macro_damaged_f1_gain_pp": macro_damaged,
        "nonnegative_events": nonnegative,
        "event_count": len(miou),
        "worst_event_gain_pp": worst,
        "bootstrap_ci_low_pp": float(bootstrap_ci_low),
        "checks": checks,
        "decision": (
            "CROSS_EVENT_IMPROVEMENT_SUPPORTED"
            if all(checks.values())
            else "CROSS_EVENT_EVIDENCE_INSUFFICIENT"
        ),
    }


__all__ = [
    "CrossEventGate",
    "assert_event_disjoint",
    "balanced_event_folds",
    "cross_event_decision",
    "sample_event",
]
