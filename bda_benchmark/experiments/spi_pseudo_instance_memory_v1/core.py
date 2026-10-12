from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


PROTOCOL = "SPI_PSEUDO_INSTANCE_UTILITY_MEMORY_M0_v1"
MEMORY_PROTOCOL = "SPI_PSEUDO_INSTANCE_UTILITY_MEMORY_BANK_v1"

REPRESENTATIONS: Mapping[str, tuple[str, ...]] = {
    "geometry": ("shape",),
    "semantic": ("semantic",),
    "s2_behavior": ("s2",),
    "base": ("shape", "semantic", "s2"),
    "base_direct": ("shape", "semantic", "s2", "direct"),
    "base_relation": ("shape", "semantic", "s2", "relation"),
    "base_matching": ("shape", "semantic", "s2", "matching"),
    "base_grid": ("shape", "semantic", "s2", "grid"),
    "base_slic": ("shape", "semantic", "s2", "slic"),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utility_labels(cache: dict, margin: float) -> torch.Tensor:
    ratio = cache["utility"]["utility_ratio"].float()
    labels = torch.full((ratio.numel(),), -1, dtype=torch.long)
    labels[ratio >= float(margin)] = 1
    labels[ratio <= -float(margin)] = 0
    return labels


def building_events(cache: dict) -> tuple[str, ...]:
    image_indices = cache["image_index"].long().tolist()
    events = tuple(str(cache["image_events"][index]) for index in image_indices)
    if len(events) != int(cache["utility"]["net"].numel()):
        raise RuntimeError("building/event rows do not align")
    return events


def building_ids(cache: dict) -> tuple[str, ...]:
    output = []
    for image_index, region_label in zip(
        cache["image_index"].long().tolist(), cache["region_label"].long().tolist()
    ):
        output.append(f"{cache['image_names'][image_index]}::loccc-{region_label}")
    if len(set(output)) != len(output):
        raise RuntimeError("pseudo-instance identifiers are not unique")
    return tuple(output)


def validate_cache(cache: dict, expected_split: str) -> None:
    if cache.get("protocol") != "SPI_BUILDING_UTILITY_CACHE_v1":
        raise RuntimeError("PI-M0 requires a BU-E0 cache")
    if cache.get("split") != expected_split:
        raise RuntimeError(f"expected {expected_split}, observed {cache.get('split')}")
    if any(
        bool(cache.get(name, True))
        for name in ("validation_accessed", "official_instances_accessed", "test_accessed")
    ):
        raise RuntimeError("BU cache accessed a forbidden split")
    rows = int(cache["utility"]["net"].numel())
    if rows <= 0:
        raise RuntimeError("BU cache has no changed pseudo-instances")
    for group, values in cache["features"].items():
        if values.ndim != 2 or values.shape[0] != rows:
            raise RuntimeError(f"feature group {group} does not align")
        if not bool(torch.isfinite(values.float()).all()):
            raise FloatingPointError(f"feature group {group} contains non-finite values")
    building_events(cache)
    building_ids(cache)


def compatible_representations(train: dict, query: dict) -> dict[str, tuple[str, ...]]:
    available = set(train["features"]) & set(query["features"])
    return {
        name: groups
        for name, groups in REPRESENTATIONS.items()
        if all(group in available for group in groups)
    }


@dataclass(frozen=True)
class RobustNormalizer:
    center: torch.Tensor
    scale: torch.Tensor
    active: torch.Tensor
    clip: float = 5.0

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        values = values.detach().float().cpu()
        if values.ndim != 2 or values.shape[1] != self.center.numel():
            raise ValueError("normalizer dimension mismatch")
        transformed = (values - self.center) / self.scale
        transformed = transformed[:, self.active].clamp(-float(self.clip), float(self.clip))
        if transformed.shape[1] == 0:
            raise RuntimeError("all representation dimensions are constant")
        return F.normalize(transformed, dim=1, eps=1e-8)

    def payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "scale": self.scale,
            "active": self.active,
            "clip": float(self.clip),
        }


def concatenate_features(cache: dict, groups: Sequence[str]) -> torch.Tensor:
    missing = [group for group in groups if group not in cache["features"]]
    if missing:
        raise KeyError(f"cache lacks feature groups: {missing}")
    return torch.cat([cache["features"][group].float() for group in groups], dim=1)


def fit_robust_normalizer(values: torch.Tensor, clip: float = 5.0) -> RobustNormalizer:
    values = values.detach().float().cpu()
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("normalizer needs at least two rows")
    center = values.median(dim=0).values
    q25 = torch.quantile(values, 0.25, dim=0)
    q75 = torch.quantile(values, 0.75, dim=0)
    robust_scale = (q75 - q25) / 1.349
    standard_scale = values.std(dim=0, unbiased=False)
    scale = torch.where(robust_scale > 1e-6, robust_scale, standard_scale)
    active = scale > 1e-6
    scale = scale.clamp_min(1e-6)
    return RobustNormalizer(center=center, scale=scale, active=active, clip=float(clip))


def event_label_balanced_indices(
    labels: torch.Tensor,
    events: Sequence[str],
    capacity_per_event_label: int,
    seed: int,
) -> torch.Tensor:
    labels = labels.long().cpu()
    events = tuple(map(str, events))
    if labels.numel() != len(events):
        raise ValueError("label/event rows do not align")
    if capacity_per_event_label <= 0:
        raise ValueError("capacity must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for event in sorted(set(events)):
        for label in (0, 1):
            indices = torch.tensor(
                [i for i, item in enumerate(events) if item == event and int(labels[i]) == label],
                dtype=torch.long,
            )
            if not indices.numel():
                continue
            order = torch.randperm(indices.numel(), generator=generator)
            selected.append(indices[order[: int(capacity_per_event_label)]])
    if not selected:
        raise RuntimeError("event-label balancing selected no memory rows")
    output = torch.cat(selected)
    for label in (0, 1):
        if int((labels[output] == label).sum()) == 0:
            raise RuntimeError(f"memory utility class {label} is empty")
    return output


@dataclass(frozen=True)
class UtilityMemory:
    keys: torch.Tensor
    labels: torch.Tensor
    utilities: torch.Tensor
    events: tuple[str, ...]
    instance_ids: tuple[str, ...]
    source_indices: torch.Tensor

    def validate(self) -> None:
        rows = self.keys.shape[0]
        if self.keys.ndim != 2 or rows == 0:
            raise ValueError("memory keys must be non-empty [N,D]")
        if self.labels.shape != (rows,) or self.utilities.shape != (rows,):
            raise ValueError("memory targets do not align")
        if len(self.events) != rows or len(self.instance_ids) != rows:
            raise ValueError("memory metadata do not align")
        if not set(self.labels.tolist()).issubset({0, 1}):
            raise ValueError("memory labels must be binary building utility")
        if not bool(torch.isfinite(self.keys).all()) or not bool(torch.isfinite(self.utilities).all()):
            raise FloatingPointError("memory contains non-finite values")


def build_memory(
    keys: torch.Tensor,
    labels: torch.Tensor,
    utilities: torch.Tensor,
    events: Sequence[str],
    instance_ids: Sequence[str],
    selected: torch.Tensor,
) -> UtilityMemory:
    selected = selected.long().cpu()
    memory = UtilityMemory(
        keys=keys[selected].float().cpu(),
        labels=labels[selected].long().cpu(),
        utilities=utilities[selected].float().cpu(),
        events=tuple(str(events[i]) for i in selected.tolist()),
        instance_ids=tuple(str(instance_ids[i]) for i in selected.tolist()),
        source_indices=selected,
    )
    memory.validate()
    return memory


def _top_event_diverse(
    similarities: torch.Tensor,
    memory_events: Sequence[str],
    query_events: Sequence[str],
    *,
    max_k: int,
    max_per_event: int,
    exclude_same_event: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top neighbors after an explicit per-event cap."""

    batch, memory_size = similarities.shape
    candidates_values: list[torch.Tensor] = []
    candidates_indices: list[torch.Tensor] = []
    memory_events = tuple(map(str, memory_events))
    for event in sorted(set(memory_events)):
        local = torch.tensor(
            [i for i, value in enumerate(memory_events) if value == event], dtype=torch.long
        )
        take = min(int(max_per_event), local.numel())
        values, positions = similarities[:, local].topk(take, dim=1)
        indices = local[positions]
        if exclude_same_event:
            blocked = torch.tensor([value == event for value in query_events], dtype=torch.bool)
            values[blocked] = -torch.inf
        candidates_values.append(values)
        candidates_indices.append(indices)
    values = torch.cat(candidates_values, dim=1)
    indices = torch.cat(candidates_indices, dim=1)
    take = min(int(max_k), values.shape[1])
    best_values, order = values.topk(take, dim=1)
    best_indices = indices.gather(1, order)
    if take < max_k:
        padding = max_k - take
        best_values = F.pad(best_values, (0, padding), value=-torch.inf)
        best_indices = F.pad(best_indices, (0, padding), value=0)
    return best_values, best_indices


def retrieve_topk_by_utility_label(
    query_keys: torch.Tensor,
    query_events: Sequence[str],
    query_ids: Sequence[str],
    memory: UtilityMemory,
    *,
    max_k: int,
    max_neighbors_per_event: int,
    exclude_same_event: bool,
    chunk_size: int = 512,
) -> dict[str, torch.Tensor]:
    memory.validate()
    query_keys = query_keys.detach().float().cpu()
    query_events = tuple(map(str, query_events))
    query_ids = tuple(map(str, query_ids))
    if query_keys.shape[0] != len(query_events) or len(query_events) != len(query_ids):
        raise ValueError("query metadata do not align")
    if query_keys.shape[1] != memory.keys.shape[1]:
        raise ValueError("query/memory key dimensions differ")
    if max_k <= 0 or max_neighbors_per_event <= 0:
        raise ValueError("neighbor budgets must be positive")
    all_similarity, all_utility, all_indices = [], [], []
    for start in range(0, query_keys.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query_keys.shape[0])
        block = query_keys[start:stop]
        block_events = query_events[start:stop]
        block_ids = query_ids[start:stop]
        label_similarity, label_utility, label_indices = [], [], []
        for label in (0, 1):
            class_rows = torch.nonzero(memory.labels == label, as_tuple=False).flatten()
            similarities = block @ memory.keys[class_rows].T
            # Always exclude an identical pseudo-instance, including leakage diagnostics.
            for row, query_id in enumerate(block_ids):
                identical = torch.tensor(
                    [memory.instance_ids[int(i)] == query_id for i in class_rows],
                    dtype=torch.bool,
                )
                similarities[row, identical] = -torch.inf
            values, local_indices = _top_event_diverse(
                similarities,
                tuple(memory.events[int(i)] for i in class_rows),
                block_events,
                max_k=max_k,
                max_per_event=max_neighbors_per_event,
                exclude_same_event=exclude_same_event,
            )
            global_indices = class_rows[local_indices]
            utilities = memory.utilities[global_indices]
            utilities = torch.where(torch.isfinite(values), utilities, torch.zeros_like(utilities))
            label_similarity.append(values)
            label_utility.append(utilities)
            label_indices.append(global_indices)
        all_similarity.append(torch.stack(label_similarity, dim=1))
        all_utility.append(torch.stack(label_utility, dim=1))
        all_indices.append(torch.stack(label_indices, dim=1))
    return {
        "similarities": torch.cat(all_similarity),  # [N,2,K]
        "utilities": torch.cat(all_utility),
        "indices": torch.cat(all_indices),
    }


def scores_from_neighbors(
    retrieval: Mapping[str, torch.Tensor], *, k: int, temperature: float
) -> dict[str, torch.Tensor]:
    similarities = retrieval["similarities"][:, :, : int(k)].float()
    utilities = retrieval["utilities"][:, :, : int(k)].float()
    finite = torch.isfinite(similarities)
    available = finite.sum(dim=(1, 2)) == 2 * int(k)
    safe = torch.where(finite, similarities, torch.full_like(similarities, -1e4))
    affinity_margin = safe[:, 1].mean(dim=1) - safe[:, 0].mean(dim=1)
    flat_similarity = safe.flatten(1)
    weights = torch.softmax(flat_similarity / float(temperature), dim=1)
    utility_vote = (weights * utilities.flatten(1)).sum(dim=1)
    signed = torch.tensor([-1.0, 1.0]).view(1, 2, 1).expand_as(safe).flatten(1)
    label_vote = (weights * signed).sum(dim=1)
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / math.log(max(2 * int(k), 2))
    for tensor in (affinity_margin, utility_vote, label_vote):
        tensor[~available] = 0.0
    return {
        "affinity_margin": affinity_margin,
        "utility_vote": utility_vote,
        "label_vote": label_vote,
        "neighbor_entropy": entropy,
        "available": available,
    }


def binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.long().flatten().cpu()
    scores = scores.float().flatten().cpu()
    positive = labels == 1
    negative = labels == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    cursor = 0
    for count in counts.tolist():
        end = cursor + int(count)
        ranks[cursor:end] = ranks[cursor:end].mean()
        cursor = end
    restored = torch.empty_like(ranks)
    restored[order] = ranks
    return float(
        (restored[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.long().flatten().cpu()
    scores = scores.float().flatten().cpu()
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    truth = (labels[order] == 1).float()
    precision = truth.cumsum(0) / torch.arange(1, truth.numel() + 1)
    return float((precision * truth).sum() / positives)


def evaluate_scores(
    labels: torch.Tensor, scores: torch.Tensor, events: Sequence[str]
) -> dict[str, object]:
    labels = labels.long().cpu()
    scores = scores.float().cpu()
    events = tuple(map(str, events))
    if labels.numel() != scores.numel() or labels.numel() != len(events):
        raise ValueError("evaluation rows do not align")
    rows = {}
    for event in sorted(set(events)):
        mask = torch.tensor([item == event for item in events], dtype=torch.bool)
        if int((labels[mask] == 0).sum()) == 0 or int((labels[mask] == 1).sum()) == 0:
            continue
        rows[event] = {
            "auroc": binary_auroc(labels[mask], scores[mask]),
            "count": int(mask.sum()),
            "positive": int((labels[mask] == 1).sum()),
            "negative": int((labels[mask] == 0).sum()),
        }
    event_aucs = [float(row["auroc"]) for row in rows.values()]
    return {
        "auroc": binary_auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "count": int(labels.numel()),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "event_macro_auroc": sum(event_aucs) / max(len(event_aucs), 1),
        "event_direction_correct": sum(value >= 0.5 for value in event_aucs),
        "event_total": len(event_aucs),
        "events": rows,
    }


def permutation_null(
    labels: torch.Tensor, scores: torch.Tensor, iterations: int, seed: int
) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observed = binary_auroc(labels, scores)
    null = []
    for _ in range(int(iterations)):
        order = torch.randperm(labels.numel(), generator=generator)
        null.append(binary_auroc(labels[order], scores))
    values = torch.tensor(null)
    return {
        "iterations": int(iterations),
        "observed": observed,
        "null_mean": float(values.mean()),
        "null_q95": float(torch.quantile(values, 0.95)),
        "one_sided_p": float((1 + (values >= observed).sum()) / (int(iterations) + 1)),
    }

