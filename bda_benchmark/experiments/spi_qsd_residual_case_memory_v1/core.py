from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_independent_building_memory_v1.core import group_matrix


STRUCTURE_GROUPS = ("rgb_structure", "geometry")
DAMAGE_GROUPS = ("sar_physics", "relative")
QUALITY_GROUPS = ("quality",)


def p3_margin(probability: torch.Tensor) -> torch.Tensor:
    top = probability.float().topk(2, dim=1).values
    return top[:, 0] - top[:, 1]


def correction_direction(cache: dict) -> torch.Tensor:
    """Source is observable P3 class; target is train-only GT majority class."""
    return cache["p3_predictions"].long() * 4 + cache["labels"].long()


def row_events(cache: dict) -> list[str]:
    return [cache["image_events"][int(i)] for i in cache["image_index"].long().tolist()]


def prepare_groups(train: dict, query: dict):
    train_structure, structure_stats = group_matrix(train, STRUCTURE_GROUPS)
    query_structure, _ = group_matrix(query, STRUCTURE_GROUPS, structure_stats)
    train_damage, damage_stats = group_matrix(train, DAMAGE_GROUPS)
    query_damage, _ = group_matrix(query, DAMAGE_GROUPS, damage_stats)
    train_quality, quality_stats = group_matrix(train, QUALITY_GROUPS)
    query_quality, _ = group_matrix(query, QUALITY_GROUPS, quality_stats)
    return {
        "train_structure": train_structure, "query_structure": query_structure,
        "train_damage": train_damage, "query_damage": query_damage,
        "train_quality": train_quality, "query_quality": query_quality,
        "statistics": {"structure": structure_stats, "damage": damage_stats, "quality": quality_stats},
    }


class FactorizedCaseMetric(nn.Module):
    """A Memory indexer, not a new damage classifier."""

    def __init__(self, structure_dim: int, damage_dim: int, hidden: int = 64):
        super().__init__()
        self.structure = nn.Sequential(
            nn.Linear(structure_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.damage = nn.Sequential(
            nn.Linear(damage_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))

    def encode(self, structure: torch.Tensor, damage: torch.Tensor):
        return (
            F.normalize(self.structure(structure.float()), dim=1, eps=1e-8),
            F.normalize(self.damage(damage.float()), dim=1, eps=1e-8),
        )

    def pair_logits(self, left_structure, left_damage, right_structure, right_damage):
        ls, ld = self.encode(left_structure, left_damage)
        rs, rd = self.encode(right_structure, right_damage)
        similarity = .35 * (ls * rs).sum(1) + .65 * (ld * rd).sum(1)
        return self.logit_scale.exp().clamp(1, 50) * (similarity - .35)


def build_pair_pools(cache: dict, minimum_purity: float = .60):
    events = row_events(cache)
    source = cache["p3_predictions"].long()
    direction = correction_direction(cache)
    valid = cache["label_purity"].float() >= minimum_purity
    by_direction, by_source = {}, {}
    for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        by_direction.setdefault(int(direction[index]), []).append(index)
        by_source.setdefault(int(source[index]), []).append(index)
    return events, source, direction, valid, by_direction, by_source


def sample_metric_pairs(cache: dict, count: int, generator: torch.Generator,
                        pools=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced cross-event pairs: same P3 source, same/different correction direction."""
    if pools is None:
        pools = build_pair_pools(cache)
    events, source, direction, valid, by_direction, by_source = pools
    anchors = torch.nonzero(valid, as_tuple=False).flatten()
    left, right, labels = [], [], []
    attempts = 0
    while len(left) < count and attempts < count * 100:
        attempts += 1
        anchor = int(anchors[torch.randint(len(anchors), (1,), generator=generator)])
        positive = len(left) % 2 == 0
        candidates = by_direction.get(int(direction[anchor]), []) if positive else by_source.get(int(source[anchor]), [])
        if not candidates:
            continue
        candidate = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
        if candidate == anchor or events[candidate] == events[anchor]:
            continue
        same_direction = int(direction[candidate]) == int(direction[anchor])
        if same_direction != positive:
            continue
        left.append(anchor); right.append(candidate); labels.append(float(positive))
    if len(left) < count:
        raise RuntimeError(f"could only sample {len(left)}/{count} valid cross-event metric pairs")
    return torch.tensor(left), torch.tensor(right), torch.tensor(labels)


def bounded_observable_memory_rows(cache: dict, maximum_per_event_source: int, seed: int) -> torch.Tensor:
    """Selection uses event and P3 source only; never query GT."""
    events = row_events(cache)
    source = cache["p3_predictions"].long()
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for event in sorted(set(events)):
        event_mask = torch.tensor([name == event for name in events])
        for p3_class in range(4):
            found = torch.nonzero(event_mask & (source == p3_class), as_tuple=False).flatten()
            if len(found) > maximum_per_event_source:
                found = found[torch.randperm(len(found), generator=generator)[:maximum_per_event_source]]
            rows.append(found)
    return torch.cat(rows).sort().values


@dataclass
class Retrieval:
    indices: torch.Tensor
    scores: torch.Tensor
    quality_distance: torch.Tensor
    coverage: torch.Tensor


def _select_event_diverse(indices, scores, quality, memory_events, topk, maximum_per_event):
    chosen_i, chosen_s, chosen_q, event_count = [], [], [], {}
    for idx, score, qdist in zip(indices.tolist(), scores.tolist(), quality.tolist()):
        if not math.isfinite(score):
            continue
        event = memory_events[idx]
        if event_count.get(event, 0) >= maximum_per_event:
            continue
        chosen_i.append(idx); chosen_s.append(score); chosen_q.append(qdist)
        event_count[event] = event_count.get(event, 0) + 1
        if len(chosen_i) == topk:
            break
    while len(chosen_i) < topk:
        chosen_i.append(-1); chosen_s.append(float("-inf")); chosen_q.append(float("inf"))
    return chosen_i, chosen_s, chosen_q


@torch.no_grad()
def retrieve(
    query_structure: torch.Tensor, memory_structure: torch.Tensor,
    query_damage: torch.Tensor, memory_damage: torch.Tensor,
    query_quality: torch.Tensor, memory_quality: torch.Tensor,
    query_events: list[str], memory_events: list[str],
    query_source: torch.Tensor, memory_source: torch.Tensor,
    query_area: torch.Tensor, memory_area: torch.Tensor,
    query_margin: torch.Tensor, memory_margin: torch.Tensor,
    *, mode: str, coarse_k: int = 64, topk: int = 8,
    maximum_per_event: int = 2, quality_limit: float = 3.0,
    area_ratio_limit: float = 2.0, chunk_size: int = 128,
    seed: int = 7341, similarity_device: str = "cpu",
) -> Retrieval:
    modes = {"random", "flat", "structure", "damage", "hierarchical", "quality", "route", "learned"}
    if mode not in modes:
        raise ValueError(f"unknown retrieval mode: {mode}")
    requested = torch.device(similarity_device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    qs = F.normalize(query_structure.float(), dim=1, eps=1e-8)
    ms = F.normalize(memory_structure.float(), dim=1, eps=1e-8).to(device)
    qd = F.normalize(query_damage.float(), dim=1, eps=1e-8)
    md = F.normalize(memory_damage.float(), dim=1, eps=1e-8).to(device)
    flat_q = F.normalize(torch.cat((qs, qd), 1), dim=1, eps=1e-8)
    flat_m = F.normalize(torch.cat((ms, md), 1), dim=1, eps=1e-8)
    memory_quality_device = memory_quality.float().to(device)
    generator = torch.Generator().manual_seed(seed)
    output_i, output_s, output_q = [], [], []
    for start in range(0, len(qs), chunk_size):
        stop = min(start + chunk_size, len(qs))
        query_structure_chunk = qs[start:stop].to(device)
        query_damage_chunk = qd[start:stop].to(device)
        structure = query_structure_chunk @ ms.T
        damage = query_damage_chunk @ md.T
        flat = flat_q[start:stop].to(device) @ flat_m.T
        if mode in {"quality", "route", "learned"}:
            quality_distance = (
                torch.cdist(query_quality[start:stop].float().to(device), memory_quality_device)
                / math.sqrt(max(memory_quality.shape[1], 1))
            )
        else:
            quality_distance = torch.zeros_like(structure)
        event_ok = torch.tensor([
            [qe != me for me in memory_events] for qe in query_events[start:stop]
        ], dtype=torch.bool, device=device)
        allowed = event_ok
        if mode in {"quality", "route", "learned"}:
            allowed = allowed & (quality_distance <= quality_limit)
        if mode in {"route", "learned"}:
            source_ok = query_source[start:stop, None].to(device) == memory_source[None].to(device)
            ratio = query_area[start:stop, None].float().to(device) / memory_area[None].float().to(device).clamp_min(1)
            area_ok = (ratio >= 1 / area_ratio_limit) & (ratio <= area_ratio_limit)
            margin_ok = (query_margin[start:stop, None].to(device) - memory_margin[None].to(device)).abs() <= .35
            allowed = allowed & source_ok & area_ok & margin_ok
        if mode == "random":
            primary = torch.rand(structure.shape, generator=generator, device="cpu").to(device)
        elif mode == "flat":
            primary = flat
        elif mode == "structure":
            primary = structure
        elif mode == "damage":
            primary = damage
        else:
            primary = structure
        primary = primary.masked_fill(~allowed, -torch.inf)
        candidate_count = min(max(coarse_k, topk * maximum_per_event * 4), memory_structure.shape[0])
        candidate_score, candidate_index = primary.topk(candidate_count, dim=1)
        if mode in {"hierarchical", "quality", "route", "learned"}:
            damage_local = damage.gather(1, candidate_index)
            structure_local = structure.gather(1, candidate_index)
            quality_local = quality_distance.gather(1, candidate_index)
            rerank = .25 * structure_local + .65 * damage_local + .10 * torch.exp(-quality_local)
            rerank = rerank.masked_fill(~torch.isfinite(candidate_score), -torch.inf)
            # Use the functional form with an explicit keyword for compatibility
            # with PyTorch builds whose Tensor.argsort accepts keyword-only args.
            order = torch.argsort(rerank, dim=1, descending=True, stable=True)
            candidate_index = candidate_index.gather(1, order)
            candidate_score = rerank.gather(1, order)
        candidate_quality = quality_distance.gather(1, candidate_index)
        candidate_index = candidate_index.cpu()
        candidate_score = candidate_score.cpu()
        candidate_quality = candidate_quality.cpu()
        for row in range(stop - start):
            ii, ss, qq = _select_event_diverse(
                candidate_index[row], candidate_score[row], candidate_quality[row],
                memory_events, topk, maximum_per_event,
            )
            output_i.append(ii); output_s.append(ss); output_q.append(qq)
    indices = torch.tensor(output_i, dtype=torch.long)
    scores = torch.tensor(output_s)
    quality = torch.tensor(output_q)
    return Retrieval(indices, scores, quality, (indices >= 0).any(1))


def neighbour_diagnostics(retrieval: Retrieval, query_direction: torch.Tensor,
                          memory_direction: torch.Tensor, memory_events: list[str]) -> dict:
    safe = retrieval.indices.clamp_min(0)
    valid = retrieval.indices >= 0
    direction = memory_direction[safe]
    purity = ((direction == query_direction[:, None]) & valid).sum(1).float() / valid.sum(1).clamp_min(1)
    hard = (query_direction // 4) != (query_direction % 4)
    diversity = torch.tensor([
        len({memory_events[int(i)] for i in row if int(i) >= 0}) for row in retrieval.indices
    ], dtype=torch.float32)
    return {
        "coverage": float(retrieval.coverage.float().mean()),
        "direction_purity_all": float(purity[retrieval.coverage].mean()) if retrieval.coverage.any() else 0.0,
        "direction_purity_p3_errors": float(purity[hard & retrieval.coverage].mean()) if (hard & retrieval.coverage).any() else 0.0,
        "p3_error_queries": int(hard.sum()),
        "mean_neighbour_events": float(diversity[retrieval.coverage].mean()) if retrieval.coverage.any() else 0.0,
    }


def retrieval_weights(retrieval: Retrieval, temperature: float = .10):
    valid = retrieval.indices >= 0
    score = retrieval.scores.masked_fill(~valid, -1e4)
    weight = torch.softmax(score / temperature, 1) * valid
    weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
    ess = 1 / weight.square().sum(1).clamp_min(1e-8)
    return weight, ess


def neighbour_consensus(retrieval: Retrieval, memory_direction: torch.Tensor):
    safe = retrieval.indices.clamp_min(0); valid = retrieval.indices >= 0
    direction = memory_direction[safe]
    consensus = torch.zeros(len(direction)); majority = torch.zeros(len(direction), dtype=torch.long)
    for row in range(len(direction)):
        values = direction[row][valid[row]]
        if len(values):
            counts = torch.bincount(values, minlength=16)
            majority[row] = counts.argmax(); consensus[row] = counts.max().float() / len(values)
    return consensus, majority


def memory_probability(base: torch.Tensor, retrieval: Retrieval, memory: dict,
                       value: str, alpha: float, gate: torch.Tensor | None = None,
                       permutation: torch.Tensor | None = None) -> torch.Tensor:
    # The zero control must be bitwise identical to P3.  This makes it possible
    # to distinguish a Memory effect from numerical reconstruction noise.
    if value == "zero":
        return base.float().clone()
    weight, _ = retrieval_weights(retrieval)
    indices = retrieval.indices.clamp_min(0)
    if permutation is not None:
        indices = permutation[indices]
    if value == "hard_label":
        values = F.one_hot(memory["labels"].long(), 4).float()
        proposal = (values[indices] * weight[..., None]).sum(1)
        output = (1 - alpha) * base.float() + alpha * proposal
    elif value == "soft_label":
        proposal = (memory["class_fractions"].float()[indices] * weight[..., None]).sum(1)
        output = (1 - alpha) * base.float() + alpha * proposal
    elif value in {"residual", "shuffled_residual"}:
        residual = memory["class_fractions"].float() - memory["p3_probabilities"].float()
        proposal = (residual[indices] * weight[..., None]).sum(1)
        output = base.float() + alpha * proposal
    else:
        raise ValueError(value)
    output = output.clamp_min(1e-6); output /= output.sum(1, keepdim=True)
    if gate is not None:
        output = torch.where(gate[:, None], output, base.float())
    return output


def reliability_gates(retrieval: Retrieval, memory_direction: torch.Tensor,
                      memory_events: list[str], *, minimum_similarity: float,
                      minimum_consensus: float, minimum_events: int, minimum_ess: float,
                      maximum_quality_distance: float):
    valid = retrieval.indices >= 0
    weights, ess = retrieval_weights(retrieval)
    consensus, _ = neighbour_consensus(retrieval, memory_direction)
    mean_similarity = (retrieval.scores.masked_fill(~valid, 0) * weights).sum(1)
    diversity = torch.tensor([
        len({memory_events[int(i)] for i in row if int(i) >= 0}) for row in retrieval.indices
    ])
    mean_quality = (retrieval.quality_distance.masked_fill(~valid, 0) * weights).sum(1)
    g0 = retrieval.coverage
    g1 = g0 & (mean_similarity >= minimum_similarity)
    g2 = g1 & (consensus >= minimum_consensus)
    g3 = g2 & (diversity >= minimum_events)
    g4 = g3 & (ess >= minimum_ess) & (mean_quality <= maximum_quality_distance)
    diagnostics = {
        "mean_similarity": float(mean_similarity[g0].mean()) if g0.any() else 0.0,
        "mean_consensus": float(consensus[g0].mean()) if g0.any() else 0.0,
        "mean_ess": float(ess[g0].mean()) if g0.any() else 0.0,
        "mean_quality_distance": float(mean_quality[g0].mean()) if g0.any() else 0.0,
    }
    return {"g0_all": g0, "g1_similarity": g1, "g2_consensus": g2,
            "g3_event_diversity": g3, "g4_full_reliability": g4}, diagnostics
