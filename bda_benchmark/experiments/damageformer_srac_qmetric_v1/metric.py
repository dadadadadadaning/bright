"""Train-only query-metric learning without changing the Memory payload.

The learned module in this file is deliberately small: a bias-free square
projection used only to rank supports.  Retrieval returns row IDs, and
``gather_support`` always reads the original parent-bank values so the reader
contract is unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_DIM = 128
NUM_CLASSES = 4


class RetrievalProjector(nn.Module):
    """Identity-initialized, bias-free projection for cosine retrieval keys."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        super().__init__()
        if int(dim) < 1:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, self.dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 2 or tokens.shape[-1] != self.dim:
            raise ValueError(f"Expected [..., {self.dim}] tokens")
        return F.normalize(self.linear(tokens.float()), dim=-1)


# A descriptive alias for callers that name the experiment rather than the
# operation.  It is the same class, not a second implementation/state format.
MetricProjector = RetrievalProjector


def encode_keys(
    tokens: torch.Tensor,
    projector: nn.Module | None = None,
    *,
    detach_input: bool = True,
) -> torch.Tensor:
    """Encode one tensor of frozen parent tokens as unit-length retrieval keys.

    ``detach_input=True`` is the protocol default: gradients may update the
    projector, but can never cross into the frozen front/Q/token encoder.
    """
    if not torch.is_tensor(tokens) or tokens.ndim < 2:
        raise ValueError("tokens must have shape [..., dim]")
    value = tokens.detach() if detach_input else tokens
    value = value.float()
    if projector is not None:
        value = projector(value)
        if value.shape != tokens.shape:
            raise ValueError("projector must preserve token shape")
    return F.normalize(value, dim=-1)


def encode_key_batches(
    tokens: torch.Tensor,
    projector: nn.Module | None = None,
    *,
    batch_size: int = 4096,
    detach_input: bool = True,
) -> torch.Tensor:
    """Chunked ``encode_keys`` with the same gradient boundary."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [count, dim]")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if len(tokens) == 0:
        return encode_keys(tokens, projector, detach_input=detach_input)
    return torch.cat(
        [
            encode_keys(
                tokens[start : start + int(batch_size)],
                projector,
                detach_input=detach_input,
            )
            for start in range(0, len(tokens), int(batch_size))
        ],
        dim=0,
    )


def project_batches(
    projector: nn.Module | None,
    *batches: torch.Tensor,
    detach_input: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Project several query/bank batches through one shared metric."""
    return tuple(
        encode_keys(batch, projector, detach_input=detach_input) for batch in batches
    )


# Small compatibility aliases for training/evaluation drivers.
project_keys = encode_keys
project_batch = encode_keys
project_in_batches = encode_key_batches


def _provenance_ids(
    values: torch.Tensor | Sequence[Any] | Any,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Make comparable integer IDs from tensor or hashable provenance values."""
    if torch.is_tensor(values):
        if values.ndim == 0:
            return values.detach().to(device=device, dtype=torch.long).expand(count)
        if values.numel() != count:
            raise ValueError(f"{name} must contain one value per token")
        if values.is_floating_point() and not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} contains non-finite values")
        return values.detach().reshape(-1).to(device=device, dtype=torch.long)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = [values] * count
    if len(values) != count:
        raise ValueError(f"{name} must contain one value per token")
    vocabulary: dict[tuple[str, str], int] = {}
    encoded = []
    for value in values:
        key = (type(value).__name__, repr(value))
        if key not in vocabulary:
            vocabulary[key] = len(vocabulary)
        encoded.append(vocabulary[key])
    return torch.tensor(encoded, dtype=torch.long, device=device)


def _labels_and_purity(
    labels: torch.Tensor,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not torch.is_tensor(labels) or labels.shape[0] != count:
        raise ValueError("labels must contain one row/value per token")
    labels = labels.detach().to(device)
    if labels.ndim == 2:
        if labels.shape[1] != NUM_CLASSES or not bool(torch.isfinite(labels).all()):
            raise ValueError("label distributions must have shape [count, 4]")
        return labels.argmax(-1).long(), labels.float().max(-1).values
    if labels.ndim != 1:
        raise ValueError("labels must be class IDs or [count, 4] distributions")
    if labels.is_floating_point():
        if not bool(torch.isfinite(labels).all()) or not bool(
            torch.equal(labels, labels.round())
        ):
            raise ValueError("class labels must be finite integers")
    labels = labels.long()
    if bool(((labels < 0) | (labels >= NUM_CLASSES)).any()):
        raise ValueError("class labels must be in [0, 3]")
    return labels, None


def high_purity_mask(
    distribution: torch.Tensor,
    minimum_purity: float,
) -> torch.Tensor:
    """Return the explicit Train-token quality gate used before metric loss."""
    if (
        distribution.ndim != 2
        or distribution.shape[1] != NUM_CLASSES
        or not 0.0 <= float(minimum_purity) <= 1.0
        or not bool(torch.isfinite(distribution).all())
    ):
        raise ValueError("Invalid distribution or minimum_purity")
    return distribution.detach().float().max(-1).values >= float(minimum_purity)


def _pair_availability(
    labels: torch.Tensor,
    events: torch.Tensor,
    images: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find usable anchors in linear memory, including unusual reused image IDs."""
    count = len(labels)
    if not count:
        empty = torch.zeros(0, dtype=torch.bool, device=labels.device)
        return empty, empty
    _, event_inverse = torch.unique(events, sorted=True, return_inverse=True)
    _, image_inverse = torch.unique(images, sorted=True, return_inverse=True)
    event_count = int(event_inverse.max()) + 1
    image_count = int(image_inverse.max()) + 1
    selected = eligible.nonzero().flatten()

    def counts(code: torch.Tensor, size: int) -> torch.Tensor:
        return torch.bincount(code[selected], minlength=size)

    class_count = counts(labels, NUM_CLASSES)
    event_counts = counts(event_inverse, event_count)
    image_counts = counts(image_inverse, image_count)
    class_event_code = labels * event_count + event_inverse
    class_image_code = labels * image_count + image_inverse
    image_event_code = image_inverse * event_count + event_inverse
    class_image_event_code = (
        (labels * image_count + image_inverse) * event_count + event_inverse
    )
    class_event_counts = counts(class_event_code, NUM_CLASSES * event_count)
    class_image_counts = counts(class_image_code, NUM_CLASSES * image_count)
    image_event_counts = counts(image_event_code, image_count * event_count)
    class_image_event_counts = counts(
        class_image_event_code, NUM_CLASSES * image_count * event_count
    )

    class_total = class_count[labels]
    event_total = event_counts[event_inverse]
    image_total = image_counts[image_inverse]
    class_event = class_event_counts[class_event_code]
    class_image = class_image_counts[class_image_code]
    image_event = image_event_counts[image_event_code]
    class_image_event = class_image_event_counts[class_image_event_code]

    # Same class outside the event, excluding any malformed/reused image ID.
    positive_count = (class_total - class_event) - (class_image - class_image_event)
    # Other class outside the event, again excluding reused image IDs exactly.
    cross_event_other_class = (
        int(eligible.sum()) - event_total - (class_total - class_event)
    )
    same_image_cross_event_other_class = (
        (image_total - image_event) - (class_image - class_image_event)
    )
    negative_count = cross_event_other_class - same_image_cross_event_other_class
    return eligible & (positive_count > 0), eligible & (negative_count > 0)


def _anchor_pair_masks(
    anchor_indices: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    images: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build only [anchors, candidates] masks, never a full-bank square mask."""
    anchor_labels = labels[anchor_indices]
    cross_event = events[anchor_indices, None] != events[None, :]
    other_image = images[anchor_indices, None] != images[None, :]
    available = cross_event & other_image & eligible[anchor_indices, None] & eligible[None, :]
    same_class = anchor_labels[:, None] == labels[None, :]
    return available & same_class, available & ~same_class


def _sample_pool(
    pool: torch.Tensor,
    count: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, bool]:
    """Sample on the generator's device and return indices on ``pool.device``."""
    if count <= 0 or len(pool) == 0:
        return pool[:0], False
    generator_device = pool.device if generator is None else torch.device(generator.device)
    if len(pool) >= count:
        order = torch.randperm(len(pool), generator=generator, device=generator_device)
        return pool[order[:count].to(pool.device)], False
    draw = torch.randint(
        len(pool), (count,), generator=generator, device=generator_device
    )
    return pool[draw.to(pool.device)], True


@torch.no_grad()
def balanced_anchor_indices(
    labels: torch.Tensor,
    events: torch.Tensor | Sequence[Any],
    images: torch.Tensor | Sequence[Any],
    per_class: int | None = None,
    generator: torch.Generator | None = None,
    *,
    eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select equally many usable anchors from every available class.

    An anchor is usable only when it has both a strict cross-event positive and
    a strict cross-event negative.  Missing classes/event-pairs are reported and
    skipped.  When an explicit ``per_class`` exceeds a minority pool, sampling
    with replacement preserves class balance without indexing failures.
    """
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    device, count = labels.device, labels.shape[0]
    label_ids, _ = _labels_and_purity(labels, count, device)
    event_ids = _provenance_ids(events, count, device, "events")
    image_ids = _provenance_ids(images, count, device, "images")
    if per_class is not None and int(per_class) < 1:
        raise ValueError("per_class must be positive or None")
    if eligible is None:
        eligible_mask = torch.ones(count, dtype=torch.bool, device=device)
    else:
        if eligible.shape != (count,):
            raise ValueError("eligible must have shape [count]")
        eligible_mask = eligible.detach().to(device=device, dtype=torch.bool)

    has_positive, has_negative = _pair_availability(
        label_ids, event_ids, image_ids, eligible_mask
    )
    usable = eligible_mask & has_positive & has_negative
    candidate_by_class = []
    usable_by_class = []
    event_count_by_class = []
    pools = []
    skipped_event_class_pairs = []
    for class_id in range(NUM_CLASSES):
        class_candidates = eligible_mask & (label_ids == class_id)
        pool = torch.where(usable & (label_ids == class_id))[0]
        pools.append(pool)
        candidate_by_class.append(int(class_candidates.sum()))
        usable_by_class.append(len(pool))
        class_events = torch.unique(event_ids[class_candidates])
        event_count_by_class.append(len(class_events))
        for event_id in class_events.tolist():
            group = class_candidates & (event_ids == int(event_id))
            missing = group & ~has_positive
            if bool(missing.any()):
                skipped_event_class_pairs.append(
                    {
                        "class_id": class_id,
                        "event_id": int(event_id),
                        "anchors": int(missing.sum()),
                    }
                )

    active = [class_id for class_id, pool in enumerate(pools) if len(pool)]
    if per_class is None:
        target = min((len(pools[class_id]) for class_id in active), default=0)
    else:
        target = int(per_class)
    selected, replacement_by_class = [], [False] * NUM_CLASSES
    for class_id in active:
        take, replaced = _sample_pool(pools[class_id], target, generator)
        selected.append(take)
        replacement_by_class[class_id] = replaced
    indices = (
        torch.cat(selected)
        if selected
        else torch.empty(0, dtype=torch.long, device=device)
    )
    # Avoid a fixed class ordering while preserving exact class counts.
    if len(indices) > 1:
        order_device = indices.device if generator is None else torch.device(generator.device)
        order = torch.randperm(len(indices), generator=generator, device=order_device)
        indices = indices[order.to(indices.device)]
    sampled_by_class = torch.bincount(
        label_ids[indices], minlength=NUM_CLASSES
    ).tolist() if len(indices) else [0] * NUM_CLASSES
    stats: dict[str, Any] = {
        "token_count": count,
        "eligible_count": int(eligible_mask.sum()),
        "candidate_by_class": candidate_by_class,
        "usable_by_class": usable_by_class,
        "sampled_by_class": [int(value) for value in sampled_by_class],
        "event_count_by_class": event_count_by_class,
        "active_class_ids": active,
        "missing_class_ids": [
            class_id for class_id, value in enumerate(candidate_by_class) if value == 0
        ],
        "classes_without_usable_anchor": [
            class_id
            for class_id, value in enumerate(candidate_by_class)
            if value > 0 and usable_by_class[class_id] == 0
        ],
        "skipped_no_positive": int((eligible_mask & ~has_positive).sum()),
        "skipped_no_negative": int((eligible_mask & ~has_negative).sum()),
        "skipped_event_class_pairs": skipped_event_class_pairs,
        "requested_per_class": None if per_class is None else int(per_class),
        "effective_per_class": target,
        "sampled_anchor_count": len(indices),
        "replacement_by_class": replacement_by_class,
    }
    return indices, stats


def cross_event_contrastive_loss(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor | Sequence[Any],
    images: torch.Tensor | Sequence[Any],
    projector: nn.Module | None = None,
    temperature: float = 0.10,
    anchor_indices: torch.Tensor | None = None,
    *,
    anchors_per_class: int | None = None,
    generator: torch.Generator | None = None,
    purity: torch.Tensor | None = None,
    minimum_purity: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Strict cross-event supervised contrastive loss and logging statistics.

    Positives share the GT dominant class and come from another event/image.
    Negatives have another class and also come from another event/image.  Only
    high-purity eligible tokens participate when ``purity`` (or a distribution
    passed as ``labels``) is available.  Invalid anchors are skipped rather than
    relaxed to same-event pairs.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [count, dim]")
    if not 0.0 < float(temperature):
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(minimum_purity) <= 1.0:
        raise ValueError("minimum_purity must be in [0, 1]")
    device, count = tokens.device, len(tokens)
    label_ids, inferred_purity = _labels_and_purity(labels, count, device)
    event_ids = _provenance_ids(events, count, device, "events")
    image_ids = _provenance_ids(images, count, device, "images")
    quality = purity
    if quality is None:
        quality = inferred_purity
    if quality is None:
        eligible = torch.ones(count, dtype=torch.bool, device=device)
        purity_filter_applied = False
    else:
        if quality.shape != (count,) or not bool(torch.isfinite(quality).all()):
            raise ValueError("purity must be finite with shape [count]")
        eligible = quality.detach().to(device).float() >= float(minimum_purity)
        purity_filter_applied = True

    sample_stats: dict[str, Any]
    if anchor_indices is None:
        anchor_indices, sample_stats = balanced_anchor_indices(
            label_ids,
            event_ids,
            image_ids,
            anchors_per_class,
            generator,
            eligible=eligible,
        )
    else:
        if not torch.is_tensor(anchor_indices) or anchor_indices.ndim != 1:
            raise ValueError("anchor_indices must be a flat tensor")
        anchor_indices = anchor_indices.detach().to(device=device, dtype=torch.long)
        if len(anchor_indices) and bool(
            ((anchor_indices < 0) | (anchor_indices >= count)).any()
        ):
            raise ValueError("anchor_indices are outside the token batch")
        sample_stats = {
            "sampled_anchor_count": len(anchor_indices),
            "sampled_by_class": (
                [int(value) for value in torch.bincount(
                    label_ids[anchor_indices], minlength=NUM_CLASSES
                ).tolist()]
                if len(anchor_indices)
                else [0] * NUM_CLASSES
            ),
            "sampling": "caller_provided",
        }

    keys = encode_keys(tokens, projector, detach_input=True)
    anchor_positive, anchor_negative = _anchor_pair_masks(
        anchor_indices, label_ids, event_ids, image_ids, eligible
    )
    valid_anchor = (
        eligible[anchor_indices]
        & anchor_positive.any(1)
        & anchor_negative.any(1)
    )
    used = anchor_indices[valid_anchor]
    used_positive = anchor_positive[valid_anchor]
    used_negative = anchor_negative[valid_anchor]

    if len(used):
        logits = keys[used] @ keys.T / float(temperature)
        comparison = used_positive | used_negative
        denominator = torch.logsumexp(logits.masked_fill(~comparison, -torch.inf), dim=1)
        positive_log_probability = logits - denominator[:, None]
        per_anchor = -(
            positive_log_probability.masked_fill(~used_positive, 0.0).sum(1)
            / used_positive.sum(1).clamp_min(1)
        )
        loss = per_anchor.mean()
    else:
        # Graph-connected zero: finite, backward-safe, and creates zero projector
        # gradients while leaving the frozen token tensor entirely detached.
        loss = keys.sum() * 0.0

    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite cross-event contrastive loss")
    stats = dict(sample_stats)
    stats.update(
        token_count=count,
        eligible_count=int(eligible.sum()),
        purity_filter_applied=purity_filter_applied,
        minimum_purity=float(minimum_purity),
        selected_anchor_count=len(anchor_indices),
        used_anchor_count=len(used),
        skipped_selected_no_positive=int((~anchor_positive.any(1)).sum()),
        skipped_selected_no_negative=int((~anchor_negative.any(1)).sum()),
        skipped_selected_low_purity=int((~eligible[anchor_indices]).sum()),
        positive_pair_count=int(used_positive.sum()),
        negative_pair_count=int(used_negative.sum()),
        used_by_class=(
            [int(value) for value in torch.bincount(
                label_ids[used], minlength=NUM_CLASSES
            ).tolist()]
            if len(used)
            else [0] * NUM_CLASSES
        ),
        zero_loss=not len(used),
    )
    return loss, stats


supervised_cross_event_contrastive = cross_event_contrastive_loss


def _query_provenance(
    value: int | torch.Tensor,
    query_count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.ndim == 0 or value.numel() == 1:
            return value.detach().to(device=device, dtype=torch.long).reshape(1).expand(
                query_count
            )
        if value.numel() == query_count:
            return value.detach().to(device=device, dtype=torch.long).reshape(-1)
        raise ValueError(f"{name} must be scalar or one value per query")
    return torch.full((query_count,), int(value), dtype=torch.long, device=device)


@torch.no_grad()
def select_ids(
    query: torch.Tensor,
    bank: dict[str, Any],
    event: int | torch.Tensor,
    image: int | torch.Tensor,
    topk: int,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    """Global projected-cosine top-k with strict event/image exclusion.

    This inference API intentionally accepts no query GT, class, target, or
    class-quota argument.  Missing rows are represented by ``-1`` IDs.
    """
    if query.ndim != 2 or int(topk) < 1:
        raise ValueError("query must be [count, dim] and topk must be positive")
    required = ("tokens", "events", "images")
    if any(name not in bank for name in required):
        raise ValueError("bank requires tokens/events/images")
    tokens = bank["tokens"]
    if tokens.ndim != 2 or tokens.shape[1] != query.shape[1]:
        raise ValueError("query and bank token dimensions disagree")
    if tokens.device != query.device:
        raise ValueError("query and bank must be on the same device")
    if bank["events"].shape != (len(tokens),) or bank["images"].shape != (
        len(tokens),
    ):
        raise ValueError("bank provenance dimensions disagree")
    if len(tokens) == 0:
        return torch.full(
            (len(query), int(topk)), -1, dtype=torch.long, device=query.device
        )

    query_key, bank_key = project_batches(projector, query, tokens)
    scores = query_key @ bank_key.T
    query_events = _query_provenance(event, len(query), query.device, "event")
    query_images = _query_provenance(image, len(query), query.device, "image")
    allowed = (
        bank["events"].to(query.device)[None] != query_events[:, None]
    ) & (bank["images"].to(query.device)[None] != query_images[:, None])
    scores = scores.masked_fill(~allowed, -torch.inf)
    width = min(int(topk), len(tokens))
    values, indices = scores.topk(width, dim=1)
    indices = indices.masked_fill(~torch.isfinite(values), -1)
    if width < int(topk):
        indices = torch.cat(
            [
                indices,
                indices.new_full((len(query), int(topk) - width), -1),
            ],
            dim=1,
        )
    return indices


def gather_support(
    indices: torch.Tensor,
    bank: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Gather the unchanged parent support payload from selected row IDs."""
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be a rank-2 integer tensor")
    required = ("tokens", "distribution", "geometry", "events", "disasters")
    if any(name not in bank for name in required):
        raise ValueError("bank is missing the parent support payload")
    count = len(bank["tokens"])
    if count < 1:
        raise ValueError("Cannot gather from an empty bank")
    if bool(((indices < -1) | (indices >= count)).any()):
        raise ValueError("support index outside bank")
    valid = indices >= 0
    safe = indices.clamp_min(0).long()
    return (
        bank["tokens"][safe].float(),
        bank["distribution"][safe].float(),
        valid,
        {
            "geometry": bank["geometry"][safe].float(),
            "events": bank["events"][safe],
            "disasters": bank["disasters"][safe],
        },
    )


__all__ = [
    "DEFAULT_DIM",
    "MetricProjector",
    "RetrievalProjector",
    "balanced_anchor_indices",
    "cross_event_contrastive_loss",
    "encode_key_batches",
    "encode_keys",
    "gather_support",
    "high_purity_mask",
    "project_batch",
    "project_batches",
    "project_in_batches",
    "project_keys",
    "select_ids",
    "supervised_cross_event_contrastive",
]
