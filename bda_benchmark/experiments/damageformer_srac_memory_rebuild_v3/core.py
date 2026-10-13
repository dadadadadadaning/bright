"""Independent FP32 paired-key retrieval and post-selection label diagnostics.

The bank is newly extracted Train evidence, not a replay of a historical FP16
archive. Both key spaces share exactly the same rows and jointly eligible
members. Query labels are deliberately absent from neighbor selection.
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


def _check_distribution(value: Tensor, rows: int, name: str) -> None:
    if not isinstance(value, Tensor) or value.shape != (rows, 4):
        raise ValueError(f"{name} must have shape [{rows}, 4]")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")
    if bool((value < 0).any()):
        raise ValueError(f"{name} contains negative class mass")
    if not bool(torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]),
                               atol=1e-4, rtol=1e-4)):
        raise ValueError(f"{name} rows must sum to one")


def validate_paired_bank(bank: Mapping[str, Any]) -> None:
    """Reject downcast, corrupt labels and ambiguous member provenance.

    Zero-norm features are valid stored observations, but neither retrieval
    space can use a row unless both keys have nonzero norms. The builder owns
    the image manifest and split checks; a stored split flag must say Train.
    """
    if not isinstance(bank, Mapping):
        raise ValueError("bank must be a mapping")
    for name in (*KEY_WIDTHS, "distribution", "events", "images", "regions",
                 "event_names"):
        if name not in bank:
            raise ValueError(f"bank is missing {name}")
    first = bank["t128"]
    if not isinstance(first, Tensor) or first.ndim != 2:
        raise ValueError("t128 must be a two-dimensional tensor")
    rows = first.shape[0]
    for key, width in KEY_WIDTHS.items():
        value = bank[key]
        if not isinstance(value, Tensor) or value.shape != (rows, width):
            raise ValueError(f"{key} must have shape [{rows}, {width}]")
        if value.dtype != torch.float32:
            raise ValueError(f"{key} must be float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} contains nonfinite values")
    _check_distribution(bank["distribution"], rows, "distribution")
    event_names = bank["event_names"]
    if not isinstance(event_names, (list, tuple)) or any(
            not isinstance(name, str) or not name for name in event_names):
        raise ValueError("event_names must be a sequence of nonempty strings")
    if len(set(event_names)) != len(event_names):
        raise ValueError("event_names must be unique")
    for field in ("events", "images", "regions", "original_rows", "member_ids"):
        if field not in bank:
            continue
        value = bank[field]
        if not isinstance(value, Tensor) or value.shape != (rows,):
            raise ValueError(f"{field} must have shape [{rows}]")
        if value.dtype != torch.long:
            raise ValueError(f"{field} must be torch.long")
        if bool((value < 0).any()):
            raise ValueError(f"{field} contains a negative identifier")
        if field in ("original_rows", "member_ids") and value.unique().numel() != rows:
            raise ValueError(f"{field} must be unique")
    if rows and int(bank["events"].max()) >= len(event_names):
        raise ValueError("events index is outside event_names")
    if "split" in bank and bank["split"] not in ("train", "Train"):
        raise ValueError("bank split must be Train")


def _normalize(value: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize finite FP32 features without overflowing their L2 norm."""
    peak = value.abs().amax(-1, keepdim=True)
    scaled = value / peak.clamp_min(torch.finfo(torch.float32).tiny)
    scaled_norm = scaled.norm(dim=-1, keepdim=True)
    valid = (peak.squeeze(-1) * scaled_norm.squeeze(-1)) > 1e-8
    unit = scaled / scaled_norm.clamp_min(torch.finfo(torch.float32).tiny)
    return unit, valid


def _valid_members(bank: Mapping[str, Any], device: torch.device) -> Tensor:
    valid = torch.ones(bank["t128"].shape[0], device=device, dtype=torch.bool)
    for key in KEY_WIDTHS:
        _, nonzero = _normalize(bank[key].to(device=device))
        valid &= nonzero
    return valid


def _query_ids(value: int | Tensor, rows: int, device: torch.device,
               name: str) -> Tensor:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer identifier")
    if isinstance(value, int):
        result = torch.full((rows,), value, device=device, dtype=torch.long)
    elif isinstance(value, Tensor):
        if value.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long")
        if value.numel() == 1:
            result = value.to(device=device).reshape(1).expand(rows)
        elif value.shape == (rows,):
            result = value.to(device=device)
        else:
            raise ValueError(f"{name} must be a scalar or shape [{rows}]")
    else:
        raise ValueError(f"{name} must be a scalar integer or long tensor")
    if bool((result < -1).any()):
        raise ValueError(f"{name} must be nonnegative or the -1 heldout sentinel")
    return result


def _rank(scores: Tensor, keep: int) -> tuple[Tensor, Tensor]:
    """Stable ties use the smaller bank row; unsupported slots have ID -1."""
    rows, members = scores.shape
    ids = torch.full((rows, keep), -1, device=scores.device, dtype=torch.long)
    selected_scores = torch.full((rows, keep), -torch.inf,
                                 device=scores.device, dtype=torch.float32)
    available = min(keep, members)
    if available:
        chosen = torch.argsort(scores, descending=True, stable=True)[:, :available]
        selected = scores.gather(1, chosen)
        ids[:, :available] = torch.where(torch.isfinite(selected), chosen, -1)
        selected_scores[:, :available] = selected
    return ids, selected_scores


def _cosine_dot(query_unit: Tensor, key_unit: Tensor) -> Tensor:
    """Use full FP32 CUDA retrieval GEMM, then restore front runtime flags.

    The historical front can retain its own TF32 policy. Only this new-bank
    retrieval dot product uses highest float32 matmul precision. CPU calls
    touch neither global setting, and exceptional CUDA calls restore both.
    """
    if query_unit.device.type != "cuda":
        return query_unit @ key_unit.T
    original_precision = torch.get_float32_matmul_precision()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        return query_unit @ key_unit.T
    finally:
        # PyTorch 2.2's allow_tf32 setter also changes matmul precision:
        # writing True after precision='medium' turns it into 'high'. Restore
        # the backend flag first and the precise original policy last.
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.set_float32_matmul_precision(original_precision)


@torch.no_grad()
def select_neighbors(query: Tensor, bank: Mapping[str, Any], key_name: str,
                     event_id: int | Tensor, image_id: int | Tensor,
                     topk: int = 8, policy: str = "global",
                     chunk_size: int = 256) -> tuple[Tensor, Tensor]:
    """Global cosine retrieval excludes both same-event and same-image rows.

    Chunking partitions queries only; every query still compares all eligible
    Train members. Both spaces use the joint key-validity mask. The sentinel
    -1 means an identity verified absent from Train, not an unknown identity.
    Query ground truth, predicted class and label values cannot alter ranking.
    """
    validate_paired_bank(bank)
    if key_name not in KEY_WIDTHS:
        raise ValueError("key_name must be t128 or f256")
    if not isinstance(query, Tensor) or query.ndim != 2 or query.shape[1] != KEY_WIDTHS[key_name]:
        raise ValueError(f"query must have shape [N, {KEY_WIDTHS[key_name]}]")
    if query.dtype != torch.float32:
        raise ValueError("query must be float32")
    if not bool(torch.isfinite(query).all()):
        raise ValueError("query contains nonfinite values")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk must be a positive integer")
    if policy != "global":
        raise ValueError("V3 only supports global retrieval")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    device, rows = query.device, query.shape[0]
    query_events = _query_ids(event_id, rows, device, "event_id")
    query_images = _query_ids(image_id, rows, device, "image_id")
    members_valid = _valid_members(bank, device)
    events, images = bank["events"].to(device), bank["images"].to(device)
    keys, _ = _normalize(bank[key_name].to(device))
    output_ids = torch.full((rows, topk), -1, device=device, dtype=torch.long)
    output_scores = torch.full((rows, topk), -torch.inf, device=device,
                               dtype=torch.float32)
    for start in range(0, rows, chunk_size):
        stop = min(rows, start + chunk_size)
        values, query_valid = _normalize(query[start:stop])
        scores = _cosine_dot(values, keys).clamp(-1.0, 1.0)
        eligible = members_valid[None, :] & query_valid[:, None]
        eligible &= events[None, :] != query_events[start:stop, None]
        eligible &= images[None, :] != query_images[start:stop, None]
        ids, chosen = _rank(scores.masked_fill(~eligible, -torch.inf), topk)
        output_ids[start:stop], output_scores[start:stop] = ids, chosen
    return output_ids, output_scores


def _check_selected(bank: Mapping[str, Any], ids: Tensor, scores: Tensor) -> Tensor:
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("ids must be a two-dimensional long tensor")
    if not isinstance(scores, Tensor) or scores.shape != ids.shape or scores.dtype != torch.float32:
        raise ValueError("scores must be float32 with the same shape as ids")
    if ids.device != scores.device:
        raise ValueError("ids and scores must be on the same device")
    if bool(((ids < -1) | (ids >= bank["t128"].shape[0])).any()):
        raise ValueError("selected ids are outside the bank")
    valid = ids >= 0
    if bool((valid & ~torch.isfinite(scores)).any()):
        raise ValueError("selected support scores must be finite")
    return valid


@torch.no_grad()
def label_evidence(bank: Mapping[str, Any], ids: Tensor, scores: Tensor,
                   temperature: float = 0.1) -> dict[str, Tensor]:
    """Cosine-weighted four-class soft values, with safe empty-support fallback.

    ``posterior`` is retained for compatibility but means label evidence,
    not a calibrated prediction. No-support rows have zero evidence/quality.
    """
    validate_paired_bank(bank)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    valid = _check_selected(bank, ids, scores)
    rows, slots = ids.shape
    supported = valid.any(-1)
    zero = torch.zeros((rows, 4), device=ids.device, dtype=torch.float32)
    if not slots or not bank["t128"].shape[0]:
        return {"posterior": zero, "supported": supported, "valid": valid,
                "confidence": zero[:, 0],
                "quality": torch.zeros((rows, 6), device=ids.device),
                "weights": torch.zeros_like(scores)}
    safe_ids = ids.clamp_min(0)
    distributions = bank["distribution"].to(ids.device)[safe_ids]
    # Subtract the unscaled peak first, so tiny positive temperatures cannot
    # turn the winning logit into inf and create an inf-minus-inf NaN.
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    peak = masked_scores.max(-1, keepdim=True).values
    peak = torch.where(supported[:, None], peak, torch.zeros_like(peak))
    scaled = ((scores.double() - peak.double()) / temperature).masked_fill(~valid, -torch.inf)
    weights = torch.exp(scaled).float().masked_fill(~valid, 0.0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    posterior = (weights[..., None] * distributions).sum(1)
    posterior = posterior.masked_fill(~supported[:, None], 0.0)
    ranked = scores.masked_fill(~valid, -torch.inf).sort(descending=True).values
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
    diversity = (valid & ~seen_before).sum(-1).float() / valid.sum(-1).clamp_min(1)
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
    """Additive post-selection label diagnostics; soft overlap is not Recall."""
    _check_distribution(query_distribution, ids.shape[0], "query_distribution")
    evidence = label_evidence(bank, ids, scores, temperature)
    target = query_distribution.to(ids.device)
    valid, supported = evidence["valid"], evidence["supported"]
    posterior = evidence["posterior"]
    rows, slots = ids.shape
    if bank["t128"].shape[0] and slots:
        selected = bank["distribution"].to(ids.device)[ids.clamp_min(0)]
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

    def scalar(value: Tensor) -> float:
        return float(value.sum().cpu())

    def vector(value: Tensor) -> list[float]:
        return value.sum(0).cpu().tolist()

    return {
        "query_count": rows,
        "supported_query_count": int(supported.sum().cpu()),
        "selected_slot_count": int(valid.sum().cpu()),
        "soft_compatibility_sum": scalar((target * posterior).sum(-1)),
        "best_neighbor_soft_compatibility_sum": scalar(best),
        "mean_neighbor_soft_compatibility_sum": scalar(average),
        "dominant_label_agreement_count": int(agreement.sum().cpu()),
        "dominant_label_conflict_count": int((supported & ~agreement).sum().cpu()),
        "dominant_class_covered_count": int(dominant_covered.sum().cpu()),
        "query_class_mass": vector(target),
        "evidence_class_mass": vector(posterior),
        "class_covered_query_count": vector(covered.float()),
        "covered_query_class_mass": vector(target * covered),
        "class_soft_compatibility_sum": vector(target * posterior),
        "quality_sum": vector(evidence["quality"]),
        "damage_query_mass_sum": scalar(target[:, 2]),
        "damage_evidence_mass_sum": scalar(posterior[:, 2]),
        "damage_shared_mass_sum": scalar(torch.minimum(target[:, 2], posterior[:, 2])),
        "damage_dominant_query_count": int((dominant == 2).sum().cpu()),
        "damage_dominant_covered_count": int(((dominant == 2) & covered[:, 2]).sum().cpu()),
        "damaged_destroyed_query_mass_sum": scalar(target[:, 2:].sum(-1)),
        "damaged_destroyed_evidence_mass_sum": scalar(posterior[:, 2:].sum(-1)),
        "damaged_destroyed_shared_mass_sum": scalar(torch.minimum(
            target[:, 2:].sum(-1), posterior[:, 2:].sum(-1))),
    }


@torch.no_grad()
def pair_neighbor_overlap(ids_a: Tensor, ids_b: Tensor) -> dict[str, Any]:
    """Additive overlap of unique support rows; unsupported pairs count as zero.

    Aggregate ``*_sum`` and their denominators before computing means.
    Neighbor overlap is not a label quality metric.
    """
    if not isinstance(ids_a, Tensor) or not isinstance(ids_b, Tensor):
        raise ValueError("neighbor ids must be tensors")
    if ids_a.ndim != 2 or ids_b.ndim != 2 or ids_a.shape[0] != ids_b.shape[0]:
        raise ValueError("neighbor ids must have equal query counts")
    if ids_a.dtype != torch.long or ids_b.dtype != torch.long:
        raise ValueError("neighbor ids must be torch.long")
    if bool((ids_a < -1).any()) or bool((ids_b < -1).any()):
        raise ValueError("neighbor ids must be nonnegative or -1")
    b = ids_b.to(ids_a.device)
    valid_a, valid_b = ids_a >= 0, b >= 0
    for values, valid in ((ids_a, valid_a), (b, valid_b)):
        duplicate = (values[:, :, None] == values[:, None, :])
        prior = torch.tril(torch.ones(values.shape[1], values.shape[1],
                                    device=values.device, dtype=torch.bool), diagonal=-1)
        if bool((duplicate & valid[:, :, None] & valid[:, None, :] & prior).any()):
            raise ValueError("neighbor rows must not contain duplicate member ids")
    common = ((ids_a[:, :, None] == b[:, None, :]) &
              valid_a[:, :, None] & valid_b[:, None, :]).any(-1).sum(-1)
    count_a, count_b = valid_a.sum(-1), valid_b.sum(-1)
    union = count_a + count_b - common
    both = (count_a > 0) & (count_b > 0)
    jaccard = common.float() / union.clamp_min(1)
    fraction = common.float() / torch.minimum(count_a, count_b).clamp_min(1)
    top1 = torch.zeros(ids_a.shape[0], dtype=torch.bool, device=ids_a.device)
    if ids_a.shape[1] and ids_b.shape[1]:
        top1 = (ids_a[:, 0] == b[:, 0]) & valid_a[:, 0] & valid_b[:, 0]
    return {"query_count": ids_a.shape[0],
            "both_supported_query_count": int(both.sum().cpu()),
            "intersection_slot_count": int(common.sum().cpu()),
            "union_slot_count": int(union.sum().cpu()),
            "neighbor_jaccard_sum": float(jaccard.sum().cpu()),
            "neighbor_overlap_fraction_sum": float(fraction.sum().cpu()),
            "top1_agreement_count": int(top1.sum().cpu())}


@torch.no_grad()
def matched_random_ids(bank: Mapping[str, Any], ids: Tensor, scores: Tensor,
                       event_id: int | Tensor, image_id: int | Tensor,
                       seed: int = 3407, policy: str = "global") -> tuple[Tensor, Tensor]:
    """Compatibility helper: random identities with original score weights.

    These scores are NOT the sampled rows' cosine scores. This intervention
    is excluded from the V3 primary paired-retrieval experiment.
    """
    validate_paired_bank(bank)
    valid = _check_selected(bank, ids, scores)
    if policy != "global":
        raise ValueError("V3 only supports global retrieval")
    rows = ids.shape[0]
    device = torch.device("cpu")
    q_events = _query_ids(event_id, rows, device, "event_id")
    q_images = _query_ids(image_id, rows, device, "image_id")
    events, images = bank["events"].cpu(), bank["images"].cpu()
    eligible_members = _valid_members(bank, device)
    result = torch.full_like(ids, -1, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for row in range(rows):
        eligible = eligible_members & (events != q_events[row]) & (images != q_images[row])
        positions = valid[row].cpu().nonzero(as_tuple=True)[0]
        candidates = eligible.nonzero(as_tuple=True)[0]
        if candidates.numel() < positions.numel():
            raise ValueError("not enough eligible members to match reference support count")
        chosen = torch.randperm(candidates.numel(), generator=generator)[:positions.numel()]
        result[row, positions] = candidates[chosen]
    return result.to(ids.device), scores.clone()


def shuffled_label_bank(bank: Mapping[str, Any], seed: int = 3407) -> dict[str, Any]:
    """Compatibility helper preserving the Train-label multiset and member rows."""
    validate_paired_bank(bank)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(bank["t128"].shape[0], generator=generator)
    result = dict(bank)
    result["distribution"] = bank["distribution"][order.to(bank["distribution"].device)]
    return result
