# -*- coding: utf-8 -*-
"""Train-only event balancing utilities for BRIGHT experiments.

The event is derived from the official sample identifier by removing only the
final numeric tile suffix.  No validation/test labels or event performance are
used to construct the weights.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
import re
from typing import Iterable, Sequence


_TILE_SUFFIX = re.compile(r"_\d+$")


def event_from_sample_name(name: str) -> str:
    """Return the disaster event encoded in a BRIGHT sample identifier."""

    event = _TILE_SUFFIX.sub("", str(name).strip())
    if not event or event == str(name).strip():
        raise ValueError(f"sample name has no final numeric tile suffix: {name!r}")
    return event


def event_balanced_weights(sample_names: Sequence[str]) -> tuple[list[float], dict[str, int]]:
    """Give every event equal expected sampling mass.

    Each image from event ``e`` receives weight ``1 / count(e)``.  Therefore
    the sum of weights is exactly one for every event, regardless of how many
    tiles that event contributes.
    """

    if not sample_names:
        raise ValueError("cannot balance an empty sample list")
    events = [event_from_sample_name(name) for name in sample_names]
    counts = Counter(events)
    weights = [1.0 / float(counts[event]) for event in events]
    return weights, dict(sorted(counts.items()))


def expected_event_mass(sample_names: Iterable[str]) -> dict[str, float]:
    """Return normalized event probabilities implied by the balancing rule."""

    names = list(sample_names)
    weights, _ = event_balanced_weights(names)
    mass: Counter[str] = Counter()
    for name, weight in zip(names, weights):
        mass[event_from_sample_name(name)] += weight
    total = sum(mass.values())
    return {event: value / total for event, value in sorted(mass.items())}


class EventBalancedBatchSampler:
    """Yield diverse micro-batches with equal long-run event exposure."""

    def __init__(
        self,
        sample_names: Sequence[str],
        batch_size: int,
        num_batches: int,
        seed: int = 3407,
    ) -> None:
        if batch_size <= 0 or num_batches <= 0:
            raise ValueError("batch_size and num_batches must be positive")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(sample_names):
            grouped[event_from_sample_name(name)].append(index)
        if not grouped:
            raise ValueError("cannot sample an empty data list")
        if batch_size > len(grouped):
            raise ValueError(
                "event-balanced micro-batch cannot exceed the number of events"
            )
        self.event_to_indices = dict(grouped)
        self.events = tuple(sorted(grouped))
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools: dict[str, deque[int]] = {}

        def refill(event: str) -> None:
            indices = list(self.event_to_indices[event])
            rng.shuffle(indices)
            pools[event] = deque(indices)

        for event in self.events:
            refill(event)
        event_queue: deque[str] = deque()

        def refill_events() -> None:
            events = list(self.events)
            rng.shuffle(events)
            event_queue.extend(events)

        refill_events()
        for _ in range(self.num_batches):
            selected = []
            deferred = []
            while len(selected) < self.batch_size:
                if not event_queue:
                    refill_events()
                event = event_queue.popleft()
                if event not in selected:
                    selected.append(event)
                else:
                    deferred.append(event)
            # Preserve duplicate slots for later batches without allowing a
            # queue containing only duplicates to spin forever.
            event_queue.extendleft(reversed(deferred))
            batch = []
            for event in selected:
                if not pools[event]:
                    refill(event)
                batch.append(pools[event].popleft())
            yield batch


__all__ = [
    "event_balanced_weights",
    "EventBalancedBatchSampler",
    "event_from_sample_name",
    "expected_event_mass",
]
