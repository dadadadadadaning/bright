from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


STRUCTURE_GROUPS = ("spatial", "rgb_edge")
DAMAGE_GROUPS = (
    "sar_texture", "stage2_direct", "stage2_matching", "structure_relation",
)
LOCAL_GROUPS = STRUCTURE_GROUPS + DAMAGE_GROUPS
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


@dataclass
class PreparedCards:
    local_key: torch.Tensor
    building_key: torch.Tensor
    cluster_to_building: torch.Tensor
    parent_id: torch.Tensor
    event: torch.Tensor
    image: torch.Tensor
    direction: torch.Tensor
    utility_label: torch.Tensor
    utility_quality: torch.Tensor
    area: torch.Tensor
    event_names: list[str]


def stable_string_hash(values: list[str]) -> torch.Tensor:
    result = []
    for value in values:
        number = 1469598103934665603
        for byte in value.encode("utf-8"):
            number ^= byte
            number = (number * 1099511628211) & ((1 << 63) - 1)
        result.append(number)
    return torch.tensor(result, dtype=torch.long)


def parent_ids(cache: dict) -> torch.Tensor:
    return (cache["image_index"].long() << 32) | cache["region_label"].long()


def direction_name(code: int) -> str:
    code = int(code)
    if not 0 <= code < 16:
        raise ValueError("direction code must be in [0,15]")
    return f"{CLASS_NAMES[code // 4]}_to_{CLASS_NAMES[code % 4]}"


def feature_matrix(cache: dict, groups: tuple[str, ...]) -> torch.Tensor:
    missing = [name for name in groups if name not in cache["features"]]
    if missing:
        raise RuntimeError(f"missing feature groups: {missing}")
    value = torch.cat([cache["features"][name].float() for name in groups], 1)
    return torch.nan_to_num(value, nan=0.0, posinf=65504.0, neginf=-65504.0)


def standardize(
    value: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = value.double()
    if mean is None:
        mean = value.mean(0)
        std = value.std(0, unbiased=False).clamp_min(1e-4)
    output = ((value - mean) / std).clamp(-8, 8).float()
    return output, mean, std


def aggregate_building_cards(
    cluster_feature: torch.Tensor,
    parent: torch.Tensor,
    area: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate changed-region descriptors without discarding their spread."""
    unique, inverse = torch.unique(parent.long(), sorted=True, return_inverse=True)
    cards = []
    for index in range(len(unique)):
        mask = inverse == index
        child = cluster_feature[mask].float()
        weight = area[mask].float().clamp_min(1)
        weight = weight / weight.sum()
        mean = (child * weight[:, None]).sum(0)
        variance = ((child - mean).square() * weight[:, None]).sum(0)
        metadata = child.new_tensor(
            [float(mask.sum()), float(area[mask].sum())]
        ).log1p()
        cards.append(torch.cat((mean, variance.clamp_min(0).sqrt(), metadata)))
    return torch.stack(cards), unique, inverse


def prepare_cards(
    cache: dict,
    statistics: dict[str, torch.Tensor] | None = None,
) -> tuple[PreparedCards, dict[str, torch.Tensor]]:
    if cache.get("meta_confirm_accessed", False) or cache.get("val450_accessed", False) or cache.get("test_accessed", False):
        raise RuntimeError("forbidden cache supplied to retrieval development")
    parent_all = parent_ids(cache)
    changed = cache["utility"]["changed"].float()
    net = cache["utility"]["net"].float()
    keep = (changed > 0) & (net != 0)
    if not bool(keep.any()):
        raise RuntimeError("cache contains no non-tied utility clusters")

    # Tied-utility clusters cannot teach a keep/revert retrieval target.  Fit
    # and aggregate the R0 cards on the same non-tied population that is
    # actually searchable, so no empty parent can enter coarse routing.
    structure_raw = feature_matrix(cache, STRUCTURE_GROUPS)[keep]
    damage_raw = feature_matrix(cache, DAMAGE_GROUPS)[keep]
    local_raw = feature_matrix(cache, LOCAL_GROUPS)[keep]
    statistics = {} if statistics is None else dict(statistics)
    structure, sm, ss = standardize(
        structure_raw, statistics.get("structure_mean"), statistics.get("structure_std")
    )
    damage, dm, ds = standardize(
        damage_raw, statistics.get("damage_mean"), statistics.get("damage_std")
    )
    local, lm, ls = standardize(
        local_raw, statistics.get("local_mean"), statistics.get("local_std")
    )
    statistics.update({
        "structure_mean": sm, "structure_std": ss,
        "damage_mean": dm, "damage_std": ds,
        "local_mean": lm, "local_std": ls,
    })

    area_all = cache["cluster_area"].float()[keep]
    parent_kept = parent_all[keep]
    structure_cards, unique_parent, inverse = aggregate_building_cards(
        structure, parent_kept, area_all
    )
    damage_cards, damage_parent, _ = aggregate_building_cards(damage, parent_kept, area_all)
    if not torch.equal(unique_parent, damage_parent):
        raise AssertionError("structure and damage parent ordering differs")
    building_raw = torch.cat((structure_cards, damage_cards), 1)
    building, bm, bs = standardize(
        building_raw, statistics.get("building_mean"), statistics.get("building_std")
    )
    statistics.update({"building_mean": bm, "building_std": bs})

    image_rows = cache["image_index"].long()
    image_names = [cache["image_names"][int(i)] for i in image_rows]
    event_names_all = [cache["image_events"][int(i)] for i in image_rows]
    event_all = stable_string_hash(event_names_all)
    image_all = stable_string_hash(image_names)
    return PreparedCards(
        local_key=F.normalize(local, dim=1, eps=1e-8),
        building_key=F.normalize(building, dim=1, eps=1e-8),
        cluster_to_building=inverse, parent_id=parent_kept,
        event=event_all[keep], image=image_all[keep],
        direction=cache["direction_code"].long()[keep],
        utility_label=(net[keep] > 0).long(),
        utility_quality=(net[keep].abs() / changed[keep].clamp_min(1)).clamp(0, 1),
        area=area_all,
        event_names=[name for name, flag in zip(event_names_all, keep.tolist()) if flag],
    ), statistics


def event_diverse_topk(
    indices: torch.Tensor,
    scores: torch.Tensor,
    events: torch.Tensor,
    *,
    topk: int,
    maximum_per_event: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen, chosen_scores, counts = [], [], {}
    for index, score in zip(indices.tolist(), scores.tolist()):
        event = int(events[index])
        if counts.get(event, 0) >= int(maximum_per_event):
            continue
        chosen.append(index); chosen_scores.append(score)
        counts[event] = counts.get(event, 0) + 1
        if len(chosen) == int(topk):
            break
    while len(chosen) < int(topk):
        chosen.append(-1); chosen_scores.append(float("-inf"))
    return torch.tensor(chosen, dtype=torch.long), torch.tensor(chosen_scores)


def _retrieval_summary(
    query: PreparedCards,
    memory: PreparedCards,
    indices: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = indices >= 0
    safe = indices.clamp_min(0)
    labels = memory.utility_label[safe]
    quality = memory.utility_quality[safe]
    logits = scores / 0.10
    weights = torch.softmax(logits.masked_fill(~valid, -1e4), 1) * quality
    weights = weights * valid
    probability = (weights * labels.float()).sum(1) / weights.sum(1).clamp_min(1e-8)
    probability[~valid.any(1)] = 0.5
    purity = ((labels == query.utility_label[:, None]) & valid).sum(1).float() / valid.sum(1).clamp_min(1)
    event_count = torch.tensor([
        len(torch.unique(memory.event[safe[row][valid[row]]])) if bool(valid[row].any()) else 0
        for row in range(len(query.local_key))
    ])
    return {
        "indices": indices, "scores": scores, "valid": valid,
        "probability": probability, "neighbour_purity": purity,
        "neighbour_event_count": event_count,
    }


@torch.no_grad()
def flat_retrieve(
    query: PreparedCards,
    memory: PreparedCards,
    *,
    topk: int = 8,
    exact_direction: bool = False,
    maximum_per_event: int = 8,
) -> dict[str, torch.Tensor]:
    """Historical one-vector cluster retrieval used as the controlled B0."""
    output_index, output_score = [], []
    for row in range(len(query.local_key)):
        candidate = memory.event != query.event[row]
        candidate &= memory.image != query.image[row]
        if exact_direction:
            candidate &= memory.direction == query.direction[row]
        ids = torch.nonzero(candidate, as_tuple=False).flatten()
        if ids.numel():
            score = memory.local_key[ids] @ query.local_key[row]
            order = score.argsort(descending=True, stable=True)
            selected, selected_score = event_diverse_topk(
                ids[order], score[order], memory.event, topk=topk,
                maximum_per_event=maximum_per_event,
            )
        else:
            selected = torch.full((topk,), -1, dtype=torch.long)
            selected_score = torch.full((topk,), float("-inf"))
        output_index.append(selected); output_score.append(selected_score)
    return _retrieval_summary(
        query, memory, torch.stack(output_index), torch.stack(output_score)
    )


@torch.no_grad()
def hierarchical_retrieve(
    query: PreparedCards,
    memory: PreparedCards,
    *,
    coarse_buildings: int = 32,
    topk: int = 8,
    exact_direction: bool = True,
    maximum_per_event: int = 2,
) -> dict[str, torch.Tensor]:
    """Building coarse routing followed by child-cluster local re-ranking."""
    memory_building_event = torch.empty(len(memory.building_key), dtype=torch.long)
    for building in range(len(memory.building_key)):
        members = memory.cluster_to_building == building
        if members.any():
            memory_building_event[building] = memory.event[members][0]
        else:
            memory_building_event[building] = -1

    output_index, output_score = [], []
    for row in range(len(query.local_key)):
        query_building = int(query.cluster_to_building[row])
        coarse = query.building_key[query_building] @ memory.building_key.T
        allowed_building = memory_building_event != query.event[row]
        coarse = coarse.masked_fill(~allowed_building, float("-inf"))
        count = min(int(coarse_buildings), int(allowed_building.sum()))
        routed = coarse.topk(count).indices if count else torch.empty(0, dtype=torch.long)
        candidate = torch.isin(memory.cluster_to_building, routed)
        candidate &= memory.event != query.event[row]
        candidate &= memory.image != query.image[row]
        if exact_direction:
            candidate &= memory.direction == query.direction[row]
        ids = torch.nonzero(candidate, as_tuple=False).flatten()
        if ids.numel():
            score = memory.local_key[ids] @ query.local_key[row]
            order = score.argsort(descending=True, stable=True)
            ids, score = ids[order], score[order]
            selected, selected_score = event_diverse_topk(
                ids, score, memory.event, topk=topk,
                maximum_per_event=maximum_per_event,
            )
        else:
            selected = torch.full((topk,), -1, dtype=torch.long)
            selected_score = torch.full((topk,), float("-inf"))
        output_index.append(selected); output_score.append(selected_score)
    indices = torch.stack(output_index)
    scores = torch.stack(output_score)
    return _retrieval_summary(query, memory, indices, scores)


def binary_auroc(target: torch.Tensor, score: torch.Tensor) -> float:
    target = target.long().flatten(); score = score.double().flatten()
    positive = target == 1; negative = target == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = score.argsort(stable=True)
    sorted_score = score[order]
    sorted_rank = torch.empty(len(score), dtype=torch.double)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        sorted_rank[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    rank = torch.empty_like(sorted_rank); rank[order] = sorted_rank
    return float((rank[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) /
                 (positive.sum() * negative.sum()))


__all__ = [
    "DAMAGE_GROUPS", "LOCAL_GROUPS", "PreparedCards", "STRUCTURE_GROUPS",
    "aggregate_building_cards", "binary_auroc", "direction_name",
    "event_diverse_topk", "flat_retrieve",
    "hierarchical_retrieve", "prepare_cards", "standardize",
]
