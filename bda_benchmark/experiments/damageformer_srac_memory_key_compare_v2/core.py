"""Paired, Train-only case retrieval; query labels enter diagnostics only.

T128 and Q-mean F256 have exactly the same support rows. Cosine retrieval
uses the intersection of nonzero rows in the two key spaces so an invalid
representation cannot silently change the members being compared. Nonfinite
features fail validation rather than silently repairing a corrupt artifact.
The four-class soft distribution is a value, never an input query label.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor


KEY_WIDTHS = {"t128": 128, "f256": 256}
QUALITY_NAMES = (
    "top1_cosine", "cosine_gap", "support_fraction",
    "normalized_label_entropy", "weighted_support_purity", "event_diversity",
)


def _check_distribution(distribution: Tensor, rows: int, name: str) -> None:
    if not isinstance(distribution, Tensor) or distribution.shape != (rows, 4):
        raise ValueError(f"{name} must have shape [{rows}, 4]")
    if not distribution.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(distribution).all()):
        raise ValueError(f"{name} contains nonfinite values")
    if bool((distribution < 0).any()):
        raise ValueError(f"{name} contains negative class mass")
    if not bool(torch.allclose(distribution.sum(-1),
                               torch.ones(rows, device=distribution.device,
                                          dtype=distribution.dtype),
                               atol=1e-4, rtol=1e-4)):
        raise ValueError(f"{name} rows must sum to one")


def validate_paired_bank(bank: Mapping[str, Any]) -> None:
    """Validate row alignment; zero keys are excluded at retrieval.

    A zero feature is not mapped to an arbitrary neighbor. A nonfinite key
    fails validation. Labels and provenance must be valid because they identify the
    shared support members and train-label values.
    """
    if not isinstance(bank, Mapping):
        raise ValueError("bank must be a mapping")
    for field in (*KEY_WIDTHS, "distribution", "events", "images", "regions",
                  "event_names"):
        if field not in bank:
            raise ValueError(f"bank is missing {field}")
    first = bank["t128"]
    if not isinstance(first, Tensor) or first.ndim != 2:
        raise ValueError("t128 must be a two-dimensional tensor")
    rows = first.shape[0]
    for key, width in KEY_WIDTHS.items():
        value = bank[key]
        if not isinstance(value, Tensor) or value.shape != (rows, width):
            raise ValueError(f"{key} must have shape [{rows}, {width}]")
        if not value.is_floating_point():
            raise ValueError(f"{key} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} contains nonfinite values")
    _check_distribution(bank["distribution"], rows, "distribution")
    event_names = bank["event_names"]
    if not isinstance(event_names, (list, tuple)) or any(
            not isinstance(name, str) for name in event_names):
        raise ValueError("event_names must be a sequence of strings")
    if len(set(event_names)) != len(event_names):
        raise ValueError("event_names must be unique")
    for field in ("events", "images", "regions"):
        value = bank[field]
        if not isinstance(value, Tensor) or value.shape != (rows,):
            raise ValueError(f"{field} must have shape [{rows}]")
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{field} must contain integer identifiers")
        if bool((value < 0).any()):
            raise ValueError(f"{field} contains a negative identifier")
    if rows and int(bank["events"].max()) >= len(event_names):
        raise ValueError("events index is outside event_names")


def _valid_members(bank: Mapping[str, Any], device: torch.device) -> Tensor:
    valid = torch.ones(bank["t128"].shape[0], device=device, dtype=torch.bool)
    for key in KEY_WIDTHS:
        value = bank[key].to(device=device, dtype=torch.float32)
        valid &= torch.isfinite(value).all(-1) & (value.norm(dim=-1) > 1e-8)
    return valid


def _query_ids(value: int | Tensor, rows: int, device: torch.device,
               name: str) -> Tensor:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer identifier")
    if isinstance(value, int):
        result = torch.full((rows,), value, device=device, dtype=torch.long)
    elif isinstance(value, Tensor):
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{name} must contain integer identifiers")
        if value.numel() == 1:
            result = value.to(device=device, dtype=torch.long).reshape(1).expand(rows)
        elif value.shape == (rows,):
            result = value.to(device=device, dtype=torch.long)
        else:
            raise ValueError(f"{name} must be a scalar or shape [{rows}]")
    else:
        raise ValueError(f"{name} must be a scalar integer or integer tensor")
    if bool((result < -1).any()):
        raise ValueError(f"{name} must be nonnegative or the -1 heldout sentinel")
    return result


def _rank(scores: Tensor, keep: int) -> tuple[Tensor, Tensor]:
    """Stable ties favor the smaller original bank row; pad unavailable slots."""
    rows, members = scores.shape
    ids = torch.full((rows, keep), -1, device=scores.device, dtype=torch.long)
    chosen_scores = torch.full((rows, keep), -torch.inf,
                              device=scores.device, dtype=torch.float32)
    available = min(keep, members)
    if available:
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        chosen = order[:, :available]
        selected = scores.gather(1, chosen)
        ids[:, :available] = torch.where(torch.isfinite(selected), chosen, -1)
        chosen_scores[:, :available] = selected
    return ids, chosen_scores


@torch.no_grad()
def select_neighbors(query: Tensor, bank: Mapping[str, Any], key_name: str,
                     event_id: int | Tensor, image_id: int | Tensor,
                     topk: int = 8, policy: str = "global",
                     chunk_size: int = 256) -> tuple[Tensor, Tensor]:
    """Cosine neighbors with strict same-event AND same-image exclusions.

    `per_class` assigns K/4 slots per support dominant class (bank labels
    only), retains soft labels as values, and does not backfill a missing
    class with another. It changes the sampling prior; its label vote is
    evidence, not a calibrated posterior. Invalid query rows return only -1.
    Query ground truth and query predicted class are deliberately absent.
    event_id/image_id=-1 means the identity is known to be absent from the
    Train bank (e.g. a heldout image), not an unverified unknown identity.
    """
    validate_paired_bank(bank)
    if key_name not in KEY_WIDTHS:
        raise ValueError("key_name must be t128 or f256")
    if not isinstance(query, Tensor) or query.ndim != 2 or query.shape[1] != KEY_WIDTHS[key_name]:
        raise ValueError(f"query must have shape [N, {KEY_WIDTHS[key_name]}]")
    if not query.is_floating_point():
        raise ValueError("query must be floating point")
    if not bool(torch.isfinite(query).all()):
        raise ValueError("query contains nonfinite values")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk must be a positive integer")
    if policy not in ("global", "per_class"):
        raise ValueError("policy must be global or per_class")
    if policy == "per_class" and topk % 4:
        raise ValueError("per_class requires topk divisible by four")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    device, rows = query.device, query.shape[0]
    query_events = _query_ids(event_id, rows, device, "event_id")
    query_images = _query_ids(image_id, rows, device, "image_id")
    members_valid = _valid_members(bank, device)
    events = bank["events"].to(device=device)
    images = bank["images"].to(device=device)
    dominant = bank["distribution"].to(device=device).argmax(-1)
    keys = bank[key_name].to(device=device, dtype=torch.float32)
    keys = torch.nan_to_num(keys, nan=0.0, posinf=0.0, neginf=0.0)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    output_ids = torch.full((rows, topk), -1, device=device, dtype=torch.long)
    output_scores = torch.full((rows, topk), -torch.inf, device=device,
                               dtype=torch.float32)
    for start in range(0, rows, chunk_size):
        stop = min(rows, start + chunk_size)
        values = query[start:stop].float()
        query_valid = torch.isfinite(values).all(-1) & (values.norm(dim=-1) > 1e-8)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores = (values @ keys.T).clamp(-1.0, 1.0)
        eligible = members_valid[None, :] & query_valid[:, None]
        eligible &= events[None, :] != query_events[start:stop, None]
        eligible &= images[None, :] != query_images[start:stop, None]
        scores = scores.masked_fill(~eligible, -torch.inf)
        if policy == "global":
            ids, selected = _rank(scores, topk)
        else:
            class_results = [_rank(scores.masked_fill(dominant[None, :] != c,
                                                      -torch.inf), topk // 4)
                             for c in range(4)]
            ids = torch.cat([result[0] for result in class_results], dim=-1)
            selected = torch.cat([result[1] for result in class_results], dim=-1)
        output_ids[start:stop], output_scores[start:stop] = ids, selected
    return output_ids, output_scores


def _check_selected(bank: Mapping[str, Any], ids: Tensor,
                    scores: Tensor) -> Tensor:
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("ids must be a two-dimensional long tensor")
    if not isinstance(scores, Tensor) or scores.shape != ids.shape or not scores.is_floating_point():
        raise ValueError("scores must be a floating tensor with the same shape as ids")
    if ids.device != scores.device:
        raise ValueError("ids and scores must be on the same device")
    members = bank["t128"].shape[0]
    if bool(((ids < -1) | (ids >= members)).any()):
        raise ValueError("selected ids are outside the bank")
    valid = ids >= 0
    if bool((valid & ~torch.isfinite(scores)).any()):
        raise ValueError("selected support scores must be finite")
    return valid


@torch.no_grad()
def label_evidence(bank: Mapping[str, Any], ids: Tensor, scores: Tensor,
                   temperature: float = 0.1) -> dict[str, Tensor]:
    """Similarity-weighted soft labels and six observable quality summaries.

    `posterior` is retained as an interface name, but is a label-evidence
    distribution, not calibrated class probability. All summaries are zero
    for no support. There is no NaN-producing softmax of an all--inf row.
    """
    validate_paired_bank(bank)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    valid = _check_selected(bank, ids, scores)
    rows, slots = ids.shape
    supported = valid.any(-1)
    zero_posterior = torch.zeros((rows, 4), device=ids.device, dtype=torch.float32)
    if slots == 0 or bank["t128"].shape[0] == 0:
        return {"posterior": zero_posterior, "supported": supported,
                "valid": valid, "confidence": zero_posterior[:, 0],
                "quality": torch.zeros((rows, 6), device=ids.device),
                "weights": torch.zeros_like(scores, dtype=torch.float32)}
    safe_ids = ids.clamp_min(0)
    distributions = bank["distribution"].to(device=ids.device, dtype=torch.float32)[safe_ids]
    scaled = (scores.float() / temperature).masked_fill(~valid, -torch.inf)
    peak = scaled.max(-1, keepdim=True).values
    peak = torch.where(supported[:, None], peak, torch.zeros_like(peak))
    weights = torch.exp(scaled - peak).masked_fill(~valid, 0.0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    posterior = (weights[..., None] * distributions).sum(1)
    posterior = posterior.masked_fill(~supported[:, None], 0.0)
    ranked = torch.sort(scores.float().masked_fill(~valid, -torch.inf),
                        dim=-1, descending=True).values
    top1 = torch.where(supported, ranked[:, 0], torch.zeros_like(ranked[:, 0]))
    gap = torch.zeros_like(top1)
    if slots > 1:
        two = valid.sum(-1) >= 2
        gap[two] = ranked[two, 0] - ranked[two, 1]
    fraction = valid.sum(-1).float() / slots
    entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(-1) / math.log(4)
    purity = (weights * distributions.max(-1).values).sum(-1)
    selected_events = bank["events"].to(ids.device)[safe_ids]
    same = selected_events[:, :, None] == selected_events[:, None, :]
    prior = torch.tril(torch.ones(slots, slots, dtype=torch.bool,
                                 device=ids.device), diagonal=-1)
    seen_before = (same & prior[None] & valid[:, None, :]).any(-1)
    distinct = (valid & ~seen_before).sum(-1).float()
    diversity = distinct / valid.sum(-1).clamp_min(1).float()
    quality = torch.stack((top1, gap, fraction, entropy, purity, diversity), -1)
    quality = quality.masked_fill(~supported[:, None], 0.0)
    return {"posterior": posterior, "supported": supported, "valid": valid,
            "confidence": posterior.max(-1).values, "quality": quality,
            "weights": weights}


@torch.no_grad()
def paired_retrieval_diagnostics(query_distribution: Tensor,
                                 bank: Mapping[str, Any], ids: Tensor,
                                 scores: Tensor,
                                 temperature: float = 0.1) -> dict[str, Any]:
    """Additive post-hoc diagnostics; these soft overlaps are not Recall.

    Query labels are used ONLY after neighbors have been selected. Aggregate
    these counts/sums across the complete split before computing averages.
    Class coverage means a selected neighbor's *dominant* class exists; it
    does not imply its label is pure or its evidence is useful to a Reader.
    """
    _check_distribution(query_distribution, ids.shape[0], "query_distribution")
    evidence = label_evidence(bank, ids, scores, temperature)
    target = query_distribution.to(device=ids.device, dtype=torch.float32)
    valid, supported = evidence["valid"], evidence["supported"]
    posterior = evidence["posterior"]
    rows, slots = ids.shape
    if bank["t128"].shape[0] and slots:
        selected = bank["distribution"].to(device=ids.device,
                                            dtype=torch.float32)[ids.clamp_min(0)]
        compatibility = (selected * target[:, None, :]).sum(-1)
        average = (compatibility * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        best = compatibility.masked_fill(~valid, -torch.inf).max(-1).values
        best = torch.where(supported, best, torch.zeros_like(best))
        classes = selected.argmax(-1)
        covered = torch.stack([(valid & (classes == c)).any(-1)
                               for c in range(4)], -1)
    else:
        average = best = torch.zeros(rows, device=ids.device)
        covered = torch.zeros((rows, 4), device=ids.device, dtype=torch.bool)
    dominant = target.argmax(-1)
    dominant_covered = covered.gather(1, dominant[:, None]).squeeze(1)
    agreement = (posterior.argmax(-1) == dominant) & supported
    quality = evidence["quality"]

    def scalar(value: Tensor) -> float:
        return float(value.sum().cpu())

    def vector(value: Tensor) -> list[float]:
        return value.sum(0).detach().cpu().tolist()

    return {
        "query_count": rows,
        "supported_query_count": int(supported.sum().cpu()),
        "selected_slot_count": int(valid.sum().cpu()),
        "soft_compatibility_sum": scalar((target * posterior).sum(-1)),
        "best_neighbor_soft_compatibility_sum": scalar(best),
        "mean_neighbor_soft_compatibility_sum": scalar(average),
        "dominant_label_agreement_count": int(agreement.sum().cpu()),
        "dominant_class_covered_count": int(dominant_covered.sum().cpu()),
        "query_class_mass": vector(target),
        "evidence_class_mass": vector(posterior),
        "class_covered_query_count": vector(covered.float()),
        "covered_query_class_mass": vector(target * covered),
        "class_soft_compatibility_sum": vector(target * posterior),
        "quality_sum": vector(quality),
        "damage_query_mass_sum": scalar(target[:, 2]),
        "damage_evidence_mass_sum": scalar(posterior[:, 2]),
        "damage_shared_mass_sum": scalar(torch.minimum(target[:, 2], posterior[:, 2])),
        "damage_dominant_query_count": int((dominant == 2).sum().cpu()),
        "damage_dominant_covered_count": int(((dominant == 2) & covered[:, 2]).sum().cpu()),
    }


@torch.no_grad()
def matched_random_ids(bank: Mapping[str, Any], ids: Tensor, scores: Tensor,
                       event_id: int | Tensor, image_id: int | Tensor,
                       seed: int = 3407, policy: str = "global") -> tuple[Tensor, Tensor]:
    """Uniform identities with matched support count and score weights.

    Scores deliberately remain the true retrieval scores: this intervention
    changes identities while preserving confidence/temperature and missing
    slots. Returned scores are NOT cosine scores of the random identities.
    Per-class slots retain their class quota. Sampling is without replacement
    and respects the same event/image exclusions and paired-key validity.
    """
    validate_paired_bank(bank)
    valid = _check_selected(bank, ids, scores)
    if policy not in ("global", "per_class"):
        raise ValueError("policy must be global or per_class")
    if policy == "per_class" and ids.shape[1] % 4:
        raise ValueError("per_class requires slot count divisible by four")
    rows, slots = ids.shape
    q_events = _query_ids(event_id, rows, torch.device("cpu"), "event_id")
    q_images = _query_ids(image_id, rows, torch.device("cpu"), "image_id")
    events, images = bank["events"].cpu(), bank["images"].cpu()
    members_valid = _valid_members(bank, torch.device("cpu"))
    dominant = bank["distribution"].cpu().argmax(-1)
    result = torch.full_like(ids, -1, device="cpu")
    reference_valid = valid.cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for row in range(rows):
        eligible = members_valid & (events != q_events[row]) & (images != q_images[row])
        groups = range(4) if policy == "per_class" else (None,)
        for group in groups:
            if group is None:
                positions = reference_valid[row].nonzero(as_tuple=True)[0]
                candidates = eligible.nonzero(as_tuple=True)[0]
            else:
                start, stop = group * (slots // 4), (group + 1) * (slots // 4)
                positions = reference_valid[row, start:stop].nonzero(as_tuple=True)[0] + start
                candidates = (eligible & (dominant == group)).nonzero(as_tuple=True)[0]
            if candidates.numel() < positions.numel():
                raise ValueError("not enough eligible members to match reference support count")
            order = torch.randperm(candidates.numel(), generator=generator)
            result[row, positions] = candidates[order[:positions.numel()]]
    return result.to(ids.device), scores.clone()


def shuffled_label_bank(bank: Mapping[str, Any], seed: int = 3407) -> dict[str, Any]:
    """Fixed Train-label permutation; preserve every key and provenance row."""
    validate_paired_bank(bank)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(bank["t128"].shape[0], generator=generator)
    result = dict(bank)
    result["distribution"] = bank["distribution"][order.to(bank["distribution"].device)]
    return result
