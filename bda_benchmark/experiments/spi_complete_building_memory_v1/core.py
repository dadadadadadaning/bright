from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ROLE_NAMES = ("whole", "core", "ring", "high_change", "stable")


def stable_string_hash(values: list[str]) -> torch.Tensor:
    output = []
    for value in values:
        number = 1469598103934665603
        for byte in value.encode("utf-8"):
            number ^= byte
            number = (number * 1099511628211) & ((1 << 63) - 1)
        output.append(number)
    return torch.tensor(output, dtype=torch.long)


def geometry_vector(mask: torch.Tensor) -> torch.Tensor:
    """Label-free geometry for one complete P3 connected component."""
    mask = mask.bool().cpu()
    y, x = torch.nonzero(mask, as_tuple=True)
    if y.numel() == 0:
        raise ValueError("empty building mask")
    h, w = mask.shape
    bh = int(y.max() - y.min() + 1); bw = int(x.max() - x.min() + 1)
    area = float(mask.sum())
    eroded = ~F.max_pool2d((~mask).float()[None, None], 3, 1, 1).bool()[0, 0]
    perimeter = float((mask & ~eroded).sum())
    return torch.tensor([
        torch.log1p(torch.tensor(area)).item(),
        bh / max(h, 1), bw / max(w, 1),
        bw / max(bh, 1), area / max(bh * bw, 1),
        perimeter / max(area, 1.0),
        float(y.float().mean()) / max(h - 1, 1),
        float(x.float().mean()) / max(w - 1, 1),
    ], dtype=torch.float32)


def _pick_tokens(tokens: torch.Tensor, score: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = tokens.shape[1]
    output = torch.zeros(count, dimension, dtype=torch.float16)
    valid = torch.zeros(count, dtype=torch.bool)
    if tokens.shape[0] == 0:
        return output, valid
    take = min(int(count), tokens.shape[0])
    chosen = score.argsort(descending=True, stable=True)[:take]
    output[:take] = F.normalize(tokens[chosen].float(), dim=1, eps=1e-8).cpu().half()
    valid[:take] = True
    return output, valid


def building_role_tokens(
    token_map: torch.Tensor,
    change_energy: torch.Tensor,
    component_mask: torch.Tensor,
    *,
    tokens_per_role: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Representative tokens for the whole/core/ring/change/stable roles.

    Roles are constructed without labels.  They describe a complete LocCC,
    rather than only its S2-changed children.
    """
    if token_map.ndim != 3 or change_energy.ndim != 2:
        raise ValueError("expected token_map [C,h,w] and energy [h,w]")
    weight = F.interpolate(
        component_mask.float()[None, None].to(token_map.device),
        size=token_map.shape[-2:], mode="area",
    )[0, 0]
    whole = weight > 0.05
    if not bool(whole.any()):
        raise RuntimeError("building vanished at token resolution")
    outside = (~whole).float()[None, None]
    core = ~F.max_pool2d(outside, 3, 1, 1).bool()[0, 0]
    core &= whole
    ring = whole & ~core
    values = change_energy[whole].float()
    low = torch.quantile(values, .25); high = torch.quantile(values, .75)
    roles = (whole, core, ring, whole & (change_energy >= high), whole & (change_energy <= low))
    flat_tokens = token_map.permute(1, 2, 0)
    packed, masks, fractions = [], [], []
    for role_index, role in enumerate(roles):
        if not bool(role.any()):
            role = whole
        score = change_energy[role].float()
        if role_index == 4:
            score = -score
        item, valid = _pick_tokens(flat_tokens[role], score, tokens_per_role)
        packed.append(item); masks.append(valid)
        fractions.append(float(role.sum()) / max(float(whole.sum()), 1.0))
    return torch.stack(packed), torch.stack(masks), torch.tensor(fractions)


def masked_mean_std(feature: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
    """Pool one modality over the complete building without using labels."""
    if feature.ndim != 3:
        raise ValueError("feature must be [C,h,w]")
    weight = F.interpolate(
        component_mask.float()[None, None].to(feature.device),
        size=feature.shape[-2:], mode="area",
    )[0, 0].clamp(0, 1)
    denominator = weight.sum().clamp_min(1e-6)
    mean = (feature.float() * weight[None]).sum((1, 2)) / denominator
    variance = ((feature.float() - mean[:, None, None]).square() * weight[None]).sum((1, 2)) / denominator
    return torch.cat((mean, variance.clamp_min(0).sqrt())).detach().cpu()


def standardize(
    value: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=65504.0, neginf=-65504.0).double()
    if mean is None:
        mean = value.mean(0)
        std = value.std(0, unbiased=False).clamp_min(1e-4)
    return ((value - mean) / std).clamp(-8, 8).float(), mean, std


def composite_building_key(
    cache: dict,
    statistics: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combine global, critical, role and geometry views of a complete card."""
    statistics = {} if statistics is None else dict(statistics)
    parts = []
    for name in ("mean", "hard_critical", "global_exist"):
        value = cache["key_variants"][name].float()
        parts.append(F.normalize(value, dim=1, eps=1e-8))
    rgb = F.normalize(cache["modality_keys"]["rgb"].float(), dim=1, eps=1e-8)
    sar = F.normalize(cache["modality_keys"]["sar"].float(), dim=1, eps=1e-8)
    parts.extend((rgb, sar, F.normalize(torch.abs(rgb - sar), dim=1, eps=1e-8)))
    role = cache["role_tokens"].float()
    role_mask = cache["role_token_mask"].float()[..., None]
    role = (role * role_mask).sum(2) / role_mask.sum(2).clamp_min(1)
    parts.append(F.normalize(role.flatten(1), dim=1, eps=1e-8))
    auxiliary = torch.cat((cache["geometry"].float(), cache["role_fractions"].float()), 1)
    auxiliary, mean, std = standardize(
        auxiliary, statistics.get("aux_mean"), statistics.get("aux_std")
    )
    statistics.update({"aux_mean": mean, "aux_std": std})
    parts.append(F.normalize(auxiliary, dim=1, eps=1e-8))
    return F.normalize(torch.cat(parts, 1), dim=1, eps=1e-8), statistics


def balanced_memory_indices(
    labels: torch.Tensor,
    events: list[str],
    *,
    maximum_per_event_class: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for event in sorted(set(events)):
        event_mask = torch.tensor([item == event for item in events])
        for label in sorted(torch.unique(labels).tolist()):
            rows = torch.nonzero(event_mask & (labels == int(label)), as_tuple=False).flatten()
            if rows.numel() > maximum_per_event_class:
                rows = rows[torch.randperm(rows.numel(), generator=generator)[:maximum_per_event_class]]
            selected.append(rows)
    return torch.cat(selected).sort().values if selected else torch.empty(0, dtype=torch.long)


def binary_auroc(target: torch.Tensor, score: torch.Tensor) -> float:
    target = target.long().flatten(); score = score.double().flatten()
    positive = target == 1; negative = target == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = score.argsort(stable=True); sorted_score = score[order]
    rank_sorted = torch.empty(len(score), dtype=torch.double)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        rank_sorted[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    rank = torch.empty_like(rank_sorted); rank[order] = rank_sorted
    return float((rank[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) /
                 (positive.sum() * negative.sum()))


def macro_f1(target: torch.Tensor, prediction: torch.Tensor, classes: int = 4) -> float:
    values = []
    for label in range(classes):
        tp = ((target == label) & (prediction == label)).sum().float()
        fp = ((target != label) & (prediction == label)).sum().float()
        fn = ((target == label) & (prediction != label)).sum().float()
        if tp + fp + fn > 0:
            values.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    return sum(values) / max(len(values), 1)


@dataclass
class RetrievalOutput:
    indices: torch.Tensor
    scores: torch.Tensor
    coverage: torch.Tensor


@torch.no_grad()
def cross_event_topk(
    query_key: torch.Tensor,
    memory_key: torch.Tensor,
    query_events: list[str],
    memory_events: list[str],
    *, topk: int, candidate_multiplier: int = 8,
    maximum_per_event: int = 2, device: str = "cpu",
) -> RetrievalOutput:
    """Chunked exact cosine search with mandatory held-event exclusion."""
    qevent = stable_string_hash(query_events); mevent = stable_string_hash(memory_events)
    output_i, output_s = [], []
    memory_device = memory_key.to(device)
    for start in range(0, len(query_key), 128):
        stop = min(start + 128, len(query_key))
        score = query_key[start:stop].to(device) @ memory_device.T
        forbidden = qevent[start:stop, None].to(device) == mevent[None].to(device)
        score.masked_fill_(forbidden, float("-inf"))
        count = min(max(topk * candidate_multiplier, topk), score.shape[1])
        values, indices = score.topk(count, 1)
        for local_row in range(stop - start):
            chosen_i, chosen_s, event_count = [], [], {}
            for index, value in zip(indices[local_row].tolist(), values[local_row].tolist()):
                if value == float("-inf"):
                    continue
                event = int(mevent[index])
                if event_count.get(event, 0) >= int(maximum_per_event):
                    continue
                chosen_i.append(index); chosen_s.append(value)
                event_count[event] = event_count.get(event, 0) + 1
                if len(chosen_i) == topk:
                    break
            while len(chosen_i) < topk:
                chosen_i.append(0); chosen_s.append(float("-inf"))
            output_i.append(torch.tensor(chosen_i)[None]); output_s.append(torch.tensor(chosen_s)[None])
    indices = torch.cat(output_i); scores = torch.cat(output_s)
    return RetrievalOutput(indices, scores, torch.isfinite(scores).any(1))
