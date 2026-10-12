from __future__ import annotations

import torch

from bda_benchmark.model.PseudoInstanceHierarchicalMemory import stable_string_hash


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def feature_matrix(cache: dict, groups=DEFAULT_GROUPS) -> torch.Tensor:
    missing = [name for name in groups if name not in cache["features"]]
    if missing:
        raise RuntimeError(f"missing feature groups: {missing}")
    value = torch.cat([cache["features"][name].float() for name in groups], 1)
    return torch.nan_to_num(value, nan=0.0, posinf=65504.0, neginf=-65504.0)


def provenance(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    image_row = cache["image_index"].long()
    image_names = [cache["image_names"][int(i)] for i in image_row]
    event_names = [cache["image_events"][int(i)] for i in image_row]
    return stable_string_hash(event_names), stable_string_hash(image_names)


def parent_ids(cache: dict) -> torch.Tensor:
    image = cache["image_index"].long()
    region = cache["region_label"].long()
    return (image << 32) | region


def parent_areas(parent: torch.Tensor, child_area: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(child_area, dtype=torch.float)
    for identity in torch.unique(parent):
        mask = parent == identity
        output[mask] = child_area[mask].float().sum()
    return output


def balanced_memory_indices(labels: torch.Tensor, events: torch.Tensor,
                            capacity_per_class: int, seed: int) -> torch.Tensor:
    """Deterministically share each class budget across available events."""
    selected = []
    generator = torch.Generator().manual_seed(int(seed))
    for label in range(3):
        class_ids = torch.nonzero(labels == label, as_tuple=False).flatten()
        if not len(class_ids):
            continue
        event_values = torch.unique(events[class_ids])
        quota = max(1, capacity_per_class // max(len(event_values), 1))
        leftovers = []
        for event in event_values:
            ids = class_ids[events[class_ids] == event]
            order = ids[torch.randperm(len(ids), generator=generator)]
            selected.append(order[:quota])
            leftovers.append(order[quota:])
        current = sum(len(row) for row in selected if len(row) and bool((labels[row] == label).all()))
        remaining = max(0, capacity_per_class - current)
        pool = torch.cat([row for row in leftovers if len(row)], 0) if any(len(row) for row in leftovers) else class_ids[:0]
        if remaining and len(pool):
            selected.append(pool[torch.randperm(len(pool), generator=generator)[:remaining]])
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)


def classification_metrics(target: torch.Tensor, probability: torch.Tensor) -> dict:
    prediction = probability.argmax(1)
    recalls, f1s = [], []
    for label in range(3):
        tp = ((prediction == label) & (target == label)).sum().float()
        fp = ((prediction == label) & (target != label)).sum().float()
        fn = ((prediction != label) & (target == label)).sum().float()
        recalls.append(float(tp / (tp + fn).clamp_min(1)))
        f1s.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    return {
        "accuracy": float((prediction == target).float().mean()),
        "macro_f1": sum(f1s) / 3, "intact_recall": recalls[0],
        "damaged_recall": recalls[1], "destroyed_recall": recalls[2],
    }
