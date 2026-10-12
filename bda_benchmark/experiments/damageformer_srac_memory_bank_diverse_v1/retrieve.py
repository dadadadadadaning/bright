"""Frozen J3 retrieval interventions. Query truth is never an input here."""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch


TOPK = 8
CANDIDATES = 64
PER_IMAGE = 1
PER_EVENT = 2


def diversify_ids(
    ranked: torch.Tensor,
    bank: dict[str, Any],
    topk: int = TOPK,
    per_image: int = PER_IMAGE,
    per_event: int = PER_EVENT,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Take rank-ordered supports; relax image cap, then event cap if needed.

    The strict pass prefers distinct source images and at most two supports per
    source event. The fallback fills from the same ranked candidate list, so a
    query never loses available support solely because diversity is impossible.
    The return value has the unchanged Reader's fixed ``topk`` width.
    """
    if ranked.ndim != 2 or ranked.dtype not in (torch.int32, torch.int64):
        raise ValueError("ranked candidates must be [query, rank] integer IDs")
    if topk < 1 or per_image < 1 or per_event < 1:
        raise ValueError("topk and diversity caps must be positive")
    images = bank["images"].detach().cpu().long().tolist()
    events = bank["events"].detach().cpu().long().tolist()
    count = len(images)
    if len(events) != count:
        raise ValueError("bank image/event lengths differ")
    rows = ranked.detach().cpu().long().tolist()
    output = torch.full((len(rows), topk), -1, dtype=torch.long)
    stats = dict(queries=len(rows), candidate_slots=0, valid_slots=0,
                 relaxed_image_slots=0, relaxed_event_slots=0,
                 queries_relaxed_image=0, queries_relaxed_event=0,
                 queries_with_shortlist_shortage=0)
    for row_number, row in enumerate(rows):
        candidates = []
        seen_candidates = set()
        for item in row:
            if item == -1:
                continue
            if item < -1 or item >= count:
                raise ValueError("shortlist contains an invalid bank row")
            if item in seen_candidates:
                raise ValueError("shortlist contains duplicate bank rows")
            seen_candidates.add(item)
            candidates.append(item)
        stats["candidate_slots"] += len(candidates)
        chosen = []
        selected = set()
        by_image, by_event = Counter(), Counter()

        def fill(stage: int) -> int:
            added = 0
            for item in candidates:
                if len(chosen) >= topk:
                    break
                if item in selected:
                    continue
                image, event = images[item], events[item]
                if stage == 0 and by_image[image] >= per_image:
                    continue
                if stage <= 1 and by_event[event] >= per_event:
                    continue
                chosen.append(item)
                selected.add(item)
                by_image[image] += 1
                by_event[event] += 1
                added += 1
            return added

        fill(0)
        if len(chosen) < min(topk, len(candidates)):
            added = fill(1)
            stats["relaxed_image_slots"] += added
            stats["queries_relaxed_image"] += int(added > 0)
        if len(chosen) < min(topk, len(candidates)):
            added = fill(2)
            stats["relaxed_event_slots"] += added
            stats["queries_relaxed_event"] += int(added > 0)
        if len(chosen) != min(topk, len(candidates)):
            raise RuntimeError("Diversity fallback did not fill available slots")
        # Fallback stages decide membership, not Reader slot order. Put the
        # chosen supports back in original similarity rank order so the arm
        # changes only which rows are present, never their relative ranking.
        chosen = [item for item in candidates if item in selected]
        stats["queries_with_shortlist_shortage"] += int(len(candidates) < topk)
        stats["valid_slots"] += len(chosen)
        if chosen:
            output[row_number, :len(chosen)] = torch.tensor(chosen)
    return output.to(ranked.device), stats


@torch.no_grad()
def select_diverse(
    query: torch.Tensor,
    bank: dict[str, Any],
    event_id: int,
    image_id: int,
    topk: int,
    projector,
    candidates: int = CANDIDATES,
) -> tuple[torch.Tensor, dict[str, int]]:
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import select_ids
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import validate_selected

    if topk != TOPK:
        raise ValueError("Frozen J3 was trained with Top-8; this ablation fixes topk=8")
    if candidates < topk:
        raise ValueError("Candidate shortlist must contain at least topk ranks")
    shortlist = select_ids(query, bank, event_id, image_id, candidates, projector=projector)
    validate_selected(shortlist, bank, event_id, image_id)
    ids, stats = diversify_ids(shortlist, bank, topk)
    validate_selected(ids, bank, event_id, image_id)
    return ids, stats


def support_diagnostics(ids: torch.Tensor, bank: dict[str, Any]) -> dict[str, Any]:
    """Support-only counts, deliberately independent of Val GT or predictions."""
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("support IDs must be a two-dimensional integer tensor")
    rows = ids.detach().cpu().long()
    valid = rows >= 0
    if bool(((rows < -1) | (rows >= len(bank["tokens"]))).any()):
        raise ValueError("Invalid support ID")
    flat = rows[valid]
    image = bank["images"].detach().cpu().long()
    event = bank["events"].detach().cpu().long()
    labels = bank["distribution"].detach().cpu().float()
    if labels.shape != (len(image), 4):
        raise ValueError("Invalid bank label shape")
    class_slots = [0] * 4
    purity_sum = entropy_sum = 0.0
    low_purity = 0
    if len(flat):
        selected = labels[flat]
        purity = selected.max(-1).values
        entropy = -(selected * selected.clamp_min(1e-12).log()).sum(-1)
        purity_sum = float(purity.double().sum())
        entropy_sum = float(entropy.double().sum())
        low_purity = int((purity < .8).sum())
        class_slots = torch.bincount(selected.argmax(-1), minlength=4).tolist()
    unique_image_sum = unique_event_sum = 0
    for row in rows.tolist():
        chosen = [item for item in row if item >= 0]
        unique_image_sum += len({int(image[item]) for item in chosen})
        unique_event_sum += len({int(event[item]) for item in chosen})
    return dict(queries=len(rows), supported_queries=int(valid.any(-1).sum()),
                valid_slots=int(valid.sum()),
                unique_image_sum=unique_image_sum,
                unique_event_sum=unique_event_sum,
                support_purity_sum=purity_sum,
                support_entropy_sum=entropy_sum,
                support_low_purity_slots=low_purity,
                support_dominant_class_slots=[int(x) for x in class_slots])


def add_support_diagnostics(total: dict[str, Any], row: dict[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, list):
            old = total.setdefault(key, [0] * len(value))
            if len(old) != len(value):
                raise ValueError("Diagnostic vector lengths differ")
            for index, item in enumerate(value):
                old[index] += item
        else:
            total[key] = total.get(key, 0) + value


def final_support_diagnostics(counts: dict[str, Any]) -> dict[str, Any]:
    slots = counts.get("valid_slots", 0)
    queries = counts.get("queries", 0)
    result = dict(counts)
    result.update(mean_valid_slots_per_query=slots / queries if queries else None,
                  mean_unique_images_per_query=(counts.get("unique_image_sum", 0) / queries
                                                if queries else None),
                  mean_unique_events_per_query=(counts.get("unique_event_sum", 0) / queries
                                                if queries else None),
                  mean_support_purity=(counts.get("support_purity_sum", 0) / slots
                                       if slots else None),
                  mean_support_entropy=(counts.get("support_entropy_sum", 0) / slots
                                        if slots else None))
    return result


def posthoc_query_label_compatibility(
    ids: torch.Tensor,
    bank: dict[str, Any],
    query_distribution: torch.Tensor,
    query_weight: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    """Compare frozen Top-8 soft labels with Val Q labels only after prediction.

    This is an observational diagnostic for Q regions, not a building-level
    accuracy metric. It never changes retrieval, ranking or Reader output.
    """
    ids = ids.detach().cpu().long()
    query = query_distribution.detach().cpu().float()
    weight = query_weight.detach().cpu().float()
    labels = bank["distribution"].detach().cpu().float()
    if (ids.ndim != 2 or query.shape != (len(ids), 4)
            or weight.shape != (len(ids),)
            or bool(((ids < -1) | (ids >= len(labels))).any())):
        raise ValueError("Post-hoc query/support shapes or IDs disagree")
    valid = ids >= 0
    count = valid.sum(-1)
    support = labels[ids.clamp_min(0)]
    mean = (support * valid[..., None]).sum(1) / count.clamp_min(1)[:, None]
    fg = query[:, 1:].sum(-1)
    fg_purity = query[:, 1:].max(-1).values / fg.clamp_min(1e-12)
    partitions = {
        "all": weight > 0,
        "foreground_pure": (weight > 0) & (fg >= .8) & (fg_purity >= .8),
        "foreground_mixed": (weight > 0) & (fg >= .8) & (fg_purity < .8),
        "background_diluted": (weight > 0) & (fg >= .2) & (fg < .8),
        "mostly_background": (weight > 0) & (fg < .2),
    }
    result = {}
    for name, mask in partitions.items():
        supported = mask & (count > 0)
        compat = (query * mean).sum(-1)
        pred = mean.argmax(-1)
        true = query.argmax(-1)
        intact = supported & (true == 1)
        damaged = supported & (true == 2)
        result[name] = dict(
            queries=int(mask.sum()), supported_queries=int(supported.sum()),
            compatibility_sum=float(compat[supported].double().sum()),
            l1_sum=float((query[supported] - mean[supported]).abs().double().sum()),
            dominant_agree=int((supported & (true == pred)).sum()),
            intact_queries=int(intact.sum()),
            intact_support_prefers_damaged=int((intact & (pred == 2)).sum()),
            intact_support_damaged_mass_sum=float(mean[intact, 2].double().sum()),
            damaged_queries=int(damaged.sum()),
            damaged_support_prefers_intact=int((damaged & (pred == 1)).sum()),
            damaged_support_intact_mass_sum=float(mean[damaged, 1].double().sum()),
        )
    return result


def support_provenance_changes(
    original_ids: torch.Tensor,
    original_bank: dict[str, Any],
    candidate_ids: torch.Tensor,
    candidate_bank: dict[str, Any],
) -> dict[str, int]:
    """Compare Train (image,region) sources, never unrelated bank row numbers."""
    old = original_ids.detach().cpu().long()
    new = candidate_ids.detach().cpu().long()
    if old.shape != new.shape or old.ndim != 2:
        raise ValueError("Support arrays must share [query,topk] shape")
    def sources(ids, bank):
        images = bank["images"].detach().cpu().long().tolist()
        regions = bank["regions"].detach().cpu().long().tolist()
        output = []
        for row in ids.tolist():
            if any(item < -1 or item >= len(images) for item in row):
                raise ValueError("Support ID is outside its bank")
            output.append(tuple(None if item < 0 else (images[item], regions[item])
                                for item in row))
        return output
    old_sources = sources(old, original_bank)
    new_sources = sources(new, candidate_bank)
    return dict(
        queries=len(old_sources), slots=int(old.numel()),
        changed_slots=sum(a != b for row_a, row_b in zip(old_sources, new_sources)
                          for a, b in zip(row_a, row_b)),
        queries_with_rank_change=sum(a != b for a, b in zip(old_sources, new_sources)),
        queries_with_set_change=sum(set(x for x in a if x is not None)
                                    != set(x for x in b if x is not None)
                                    for a, b in zip(old_sources, new_sources)),
    )
