from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def stable_hash(values: Sequence[str]) -> torch.Tensor:
    """Deterministic event identifiers; Python's hash is process-randomised."""
    output = []
    for value in values:
        number = 1469598103934665603
        for byte in value.encode("utf-8"):
            number ^= byte
            number = (number * 1099511628211) & ((1 << 63) - 1)
        output.append(number)
    return torch.tensor(output, dtype=torch.long)


def affected_target(labels: torch.Tensor) -> torch.Tensor:
    return (labels.long() >= 2).long()


def semantic_label_and_purity(target: torch.Tensor, mask: torch.Tensor) -> tuple[int, float]:
    """Evaluation-only majority label; never use this value to decide cache entry."""
    values = target[mask]
    values = values[(values >= 0) & (values <= 3)]
    if values.numel() == 0:
        return 0, 0.0
    counts = torch.bincount(values.long(), minlength=4)
    return int(counts.argmax()), float(counts.max() / counts.sum().clamp_min(1))


def region_geometry(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool().cpu()
    y, x = torch.nonzero(mask, as_tuple=True)
    if y.numel() == 0:
        raise ValueError("empty structural region")
    height, width = mask.shape
    area = float(mask.sum())
    bh = float(y.max() - y.min() + 1); bw = float(x.max() - x.min() + 1)
    return torch.tensor((
        torch.log1p(torch.tensor(area)).item(),
        bh / max(height, 1), bw / max(width, 1),
        bw / max(bh, 1.0), area / max(bh * bw, 1.0),
        float(y.float().mean()) / max(height - 1, 1),
        float(x.float().mean()) / max(width - 1, 1),
    ), dtype=torch.float32)


def pool_feature(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean/std pool without using labels or predicted semantic classes."""
    if feature.ndim != 3:
        raise ValueError("feature must be [C,H,W]")
    weight = F.interpolate(mask.float()[None, None].to(feature.device), size=feature.shape[-2:], mode="area")[0, 0]
    denominator = weight.sum().clamp_min(1e-6)
    mean = (feature.float() * weight[None]).sum((1, 2)) / denominator
    std = (((feature.float() - mean[:, None, None]).square() * weight[None]).sum((1, 2)) / denominator).sqrt()
    return torch.cat((mean, std)).detach().cpu().float()


def edge_statistics(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Label-free local RGB structure descriptor used by the retrieval key."""
    if rgb.ndim != 3:
        raise ValueError("rgb must be [C,H,W]")
    gray = rgb.float().mean(0, keepdim=True)[None]
    kx = torch.tensor(((-3., 0., 3.), (-10., 0., 10.), (-3., 0., 3.)), device=rgb.device)[None, None] / 16.
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)[0, 0]; gy = F.conv2d(gray, ky, padding=1)[0, 0]
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
    values = magnitude[mask.to(magnitude.device)]
    if values.numel() == 0:
        return torch.zeros(4)
    return torch.tensor((values.mean(), values.std(unbiased=False), values.quantile(.75), values.quantile(.95)), dtype=torch.float32)


def structural_regions(rgb: torch.Tensor, *, n_segments: int, compactness: float) -> torch.Tensor:
    """Pre-RGB-only SLIC regions with a deterministic dependency-free fallback.

    The fallback is deliberately reported in the cache metadata. It is a smoke-test
    fallback, not a replacement for SLIC in the research result.
    """
    image = rgb.detach().float().cpu().permute(1, 2, 0).numpy()
    try:
        from skimage.segmentation import slic
        labels = slic(image, n_segments=int(n_segments), compactness=float(compactness),
                      start_label=1, channel_axis=-1, convert2lab=False)
        return torch.from_numpy(labels.astype("int32", copy=False))
    except ImportError:
        height, width = image.shape[:2]
        side = max(4, int((height * width / max(int(n_segments), 1)) ** .5))
        yy = torch.arange(height)[:, None] // side; xx = torch.arange(width)[None] // side
        return (yy * (int((width + side - 1) // side)) + xx + 1).to(torch.int32)


def normalise_from_train(value: torch.Tensor, statistics: dict | None = None) -> tuple[torch.Tensor, dict]:
    statistics = {} if statistics is None else dict(statistics)
    value = torch.nan_to_num(value.float(), nan=0., posinf=65504., neginf=-65504.)
    if "mean" not in statistics:
        statistics["mean"] = value.mean(0)
        statistics["std"] = value.std(0, unbiased=False).clamp_min(1e-4)
    z = ((value - statistics["mean"]) / statistics["std"]).clamp(-8, 8)
    return F.normalize(z, dim=1, eps=1e-8), statistics


def balanced_event_class_rows(labels: torch.Tensor, events: Sequence[str], *, maximum_per_event_class: int, seed: int,
                              eligible: torch.Tensor | None = None) -> torch.Tensor:
    """Bound a labelled *training* Memory bank without using query labels.

    This does not discard entries while building the cache.  It only fixes the
    labelled training library size used in a retrieval experiment, preventing a
    common class or a large event from dominating both runtime and neighbours.
    """
    if maximum_per_event_class <= 0:
        return torch.nonzero(torch.ones(len(labels), dtype=torch.bool) if eligible is None else eligible.bool(), as_tuple=False).flatten()
    eligible = torch.ones(len(labels), dtype=torch.bool) if eligible is None else eligible.bool()
    generator = torch.Generator().manual_seed(int(seed))
    result = []
    event_tensor = torch.tensor([value for value in stable_hash(events).tolist()], dtype=torch.long)
    for event in torch.unique(event_tensor, sorted=True).tolist():
        for label in range(4):
            rows = torch.nonzero(eligible & (event_tensor == event) & (labels.long() == label), as_tuple=False).flatten()
            if rows.numel() > int(maximum_per_event_class):
                rows = rows[torch.randperm(rows.numel(), generator=generator)[:int(maximum_per_event_class)]]
            result.append(rows)
    return torch.cat(result).sort().values if result else torch.empty(0, dtype=torch.long)


@dataclass
class Retrieval:
    indices: torch.Tensor
    scores: torch.Tensor
    coverage: torch.Tensor


@torch.no_grad()
def cross_event_topk(query_key: torch.Tensor, memory_key: torch.Tensor,
                     query_events: Sequence[str], memory_events: Sequence[str], *, topk: int,
                     maximum_per_event: int = 2, device: str = "cpu") -> Retrieval:
    """Exact cosine retrieval excluding every entry from the query's disaster event."""
    qevent, mevent = stable_hash(query_events), stable_hash(memory_events)
    memory_device = memory_key.float().to(device)
    all_indices, all_scores = [], []
    for start in range(0, len(query_key), 128):
        stop = min(start + 128, len(query_key))
        score = query_key[start:stop].float().to(device) @ memory_device.T
        score.masked_fill_(qevent[start:stop, None].to(device) == mevent[None].to(device), float("-inf"))
        count = min(max(int(topk) * 8, int(topk)), score.shape[1])
        values, indices = score.topk(count, dim=1)
        for row in range(stop - start):
            selected_i, selected_s, used = [], [], {}
            for index, value in zip(indices[row].tolist(), values[row].tolist()):
                if value == float("-inf"):
                    continue
                event = int(mevent[index])
                if used.get(event, 0) >= int(maximum_per_event):
                    continue
                selected_i.append(index); selected_s.append(value); used[event] = used.get(event, 0) + 1
                if len(selected_i) == int(topk):
                    break
            selected_i += [0] * (int(topk) - len(selected_i))
            selected_s += [float("-inf")] * (int(topk) - len(selected_s))
            all_indices.append(torch.tensor(selected_i)[None]); all_scores.append(torch.tensor(selected_s)[None])
    indices, scores = torch.cat(all_indices), torch.cat(all_scores)
    return Retrieval(indices, scores, torch.isfinite(scores).any(1))


def weighted_binary_vote(retrieval: Retrieval, values: torch.Tensor) -> torch.Tensor:
    valid = retrieval.indices >= 0
    weights = torch.softmax(retrieval.scores.masked_fill(~valid, -1e9), dim=1) * valid
    return (values.long()[retrieval.indices.clamp_min(0)].float() * weights).sum(1)


def binary_auroc(target: torch.Tensor, score: torch.Tensor) -> float:
    target, score = target.long().flatten(), score.double().flatten()
    pos, neg = target == 1, target == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = score.argsort(stable=True); ordered = score[order]
    rank_ordered = torch.empty(len(score), dtype=torch.double)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and ordered[stop] == ordered[start]: stop += 1
        rank_ordered[start:stop] = (start + 1 + stop) / 2
        start = stop
    rank = torch.empty_like(rank_ordered); rank[order] = rank_ordered
    return float((rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def binary_f1(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target, prediction = target.bool(), prediction.bool()
    tp = (target & prediction).sum().float(); fp = (~target & prediction).sum().float(); fn = (target & ~prediction).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def event_auc(target: torch.Tensor, score: torch.Tensor, events: Sequence[str], coverage: torch.Tensor) -> dict[str, float]:
    result = {}
    for event in sorted(set(events)):
        mask = coverage & torch.tensor([x == event for x in events])
        value = binary_auroc(target[mask], score[mask])
        if value == value: result[event] = value
    return result
