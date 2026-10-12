"""Selection and validation for a quality-eligible, diverse Train bank.

Q remains the frozen SRAC partition.  A label is the original four-way region
fraction, not a building identity or a hardened class assigned to every pixel.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from pathlib import Path

import torch


FORMAT = "SRAC_MEMORY_BANK_DIVERSE_V1"
FIELDS = ("tokens", "distribution", "events", "images", "regions", "geometry", "disasters")


def source_code_hash():
    """Hash only this builder's executable source, with stable relative names."""
    root = Path(__file__).parent
    sha = hashlib.sha256()
    for name in ("bank.py", "build.py"):
        sha.update(name.encode("utf-8"))
        sha.update((root / name).read_bytes())
    return sha.hexdigest()


def stable_tie(seed: int, image_name: str, region_id: int) -> int:
    raw = f"{seed}\0{image_name}\0{region_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def bucket_counts(bank: dict, events: int) -> list[list[int]]:
    distribution = bank["distribution"].float()
    event_ids = bank["events"].long()
    if distribution.ndim != 2 or distribution.shape[1] != 4 or event_ids.shape != (len(distribution),):
        raise ValueError("Invalid bank distribution/events shape")
    if not torch.isfinite(distribution).all() or (distribution < 0).any():
        raise ValueError("Invalid bank soft labels")
    if not torch.allclose(distribution.sum(-1), torch.ones(len(distribution)), atol=1e-5, rtol=0):
        raise ValueError("Bank soft labels do not sum to one")
    if ((event_ids < 0) | (event_ids >= events)).any():
        raise ValueError("Bank event ID out of range")
    code = event_ids * 4 + distribution.argmax(-1)
    return torch.bincount(code, minlength=events * 4).reshape(events, 4).tolist()


def validate_shard(shard: dict, *, image_id: int, image_name: str,
                   event_id: int, expected_dim: int, min_purity: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return eligible mask, dominant class and purity; reject stale/corrupt shards."""
    if not isinstance(shard, dict) or shard.get("name") != image_name:
        raise ValueError("Train shard name/order mismatch")
    if int(shard.get("image", -1)) != image_id or int(shard.get("event", -1)) != event_id:
        raise ValueError("Train shard image/event mismatch")
    tokens, distribution, geometry = (shard[key] for key in ("tokens", "distribution", "geometry"))
    weights, mass, region_ids = (shard[key] for key in ("weights", "mass", "region_ids"))
    if not all(torch.is_tensor(x) for x in (tokens, distribution, geometry, weights, mass, region_ids)):
        raise ValueError("Train shard fields must be tensors")
    n = len(tokens)
    if (tokens.ndim != 2 or tokens.shape[1] != expected_dim or distribution.shape != (n, 4)
            or geometry.ndim != 2 or len(geometry) != n or weights.shape != (n,)
            or mass.shape != (n,) or region_ids.shape != (n,)):
        raise ValueError("Train shard tensor shape mismatch")
    if not torch.equal(region_ids.long(), torch.arange(n)):
        raise ValueError("Train shard region IDs disagree with row indices")
    if any(not bool(torch.isfinite(x).all()) for x in (tokens, distribution, geometry, weights, mass)):
        raise ValueError("Non-finite Train shard values")
    if bool((weights < 0).any()) or bool((mass < 0).any()) or bool((distribution < 0).any()):
        raise ValueError("Negative Train shard mass, weight or soft label")
    valid = (weights > 0) & (mass > 1e-6)
    purity, dominant = distribution.float().max(-1)
    if bool(valid.any()) and not torch.allclose(distribution[valid].float().sum(-1),
                                                torch.ones(int(valid.sum())), atol=1e-5, rtol=0):
        raise ValueError("Valid Train shard labels do not sum to one")
    eligible = valid & (purity >= min_purity)
    return eligible, dominant, purity


def quality_key(purity: float, mass: float, seed: int, name: str, region_id: int):
    """Purity first; capped Q mass and seeded hash only resolve near-equal cases."""
    if not math.isfinite(purity) or not math.isfinite(mass) or mass < 0:
        raise ValueError("Invalid candidate quality")
    return (float(purity), float(min(mass, 16.0)), stable_tie(seed, name, region_id))


def bounded_push(heap: list, item: tuple, limit: int) -> None:
    if limit <= 0:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def choose_bucket(capped: list, unrestricted: list, quota: int):
    """Respect the image cap first, then relax only enough to fill exact quota."""
    if quota < 0:
        raise ValueError("Invalid quota")
    if quota == 0:
        return []
    chosen = []
    seen = set()
    for item in sorted(capped, reverse=True):
        row = item[1:3]
        if row not in seen:
            chosen.append((item, False))
            seen.add(row)
            if len(chosen) == quota:
                return chosen
    for item in sorted(unrestricted, reverse=True):
        row = item[1:3]
        if row not in seen:
            chosen.append((item, True))
            seen.add(row)
            if len(chosen) == quota:
                return chosen
    if quota:
        raise ValueError(f"Insufficient eligible Train regions: need {quota}, found {len(chosen)}")
    return chosen


def choose_diverse_bucket(capped: list, unrestricted: list, quota: int,
                          keys: dict, per_image_cap: int,
                          quality_weight: float = 0.2):
    """Choose real rows by projected-key coverage within an exact bucket quota.

    The candidate pool has already been bounded by Train-label purity/Q mass.
    Greedy cosine-distance coverage starts with the best-quality candidate.
    An image cap is relaxed only when no unselected candidate obeys it.  The
    returned row identities still refer to untouched Train-shard payloads.
    """
    if quota < 0 or per_image_cap < 1 or not math.isfinite(quality_weight) or quality_weight < 0:
        raise ValueError("Invalid diverse-bank selection argument")
    if quota == 0:
        return []
    candidates = {}
    for item in (*capped, *unrestricted):
        identity = (item[1], item[2])
        if identity not in candidates or item > candidates[identity]:
            candidates[identity] = item
    if len(candidates) < quota:
        raise ValueError(f"Insufficient eligible Train candidate rows: need {quota}, found {len(candidates)}")
    ordered = sorted(candidates.values(), reverse=True)
    if any((item[1], item[2]) not in keys for item in ordered):
        raise ValueError("Missing projected key for diverse-bank candidate")
    matrix = torch.stack([keys[(item[1], item[2])].float() for item in ordered])
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("Invalid projected candidate keys")
    norms = matrix.norm(dim=-1)
    if bool((norms < 1e-8).any()):
        raise ValueError("Zero-length projected candidate key")
    matrix = torch.nn.functional.normalize(matrix, dim=-1)
    similarity = (matrix @ matrix.T).clamp(-1.0, 1.0)
    # The first item is the highest-quality anchor. Quality remains a small,
    # fixed contribution; source diversity is imposed by the image cap.
    chosen = [(ordered[0], False)]
    image_counts = {ordered[0][1]: 1}
    remaining = set(range(1, len(ordered)))
    closest = similarity[0].clone()
    while len(chosen) < quota:
        allowed = [i for i in sorted(remaining)
                   if image_counts.get(ordered[i][1], 0) < per_image_cap]
        relaxed = not allowed
        pool = allowed if allowed else sorted(remaining)
        # Ranking is deterministic: score, original quality rank, then source.
        # Cosine distance is 1 - nearest selected similarity in [0, 2].
        best = max(pool, key=lambda i: (
            float(1.0 - closest[i]) + quality_weight * ordered[i][0][0],
            ordered[i][0], -ordered[i][1], -ordered[i][2]))
        chosen.append((ordered[best], relaxed))
        remaining.remove(best)
        image_counts[ordered[best][1]] = image_counts.get(ordered[best][1], 0) + 1
        closest = torch.maximum(closest, similarity[best])
    return chosen


def purity_summary(distribution: torch.Tensor) -> dict:
    p = distribution.float().max(-1).values
    if len(p) == 0:
        return dict(count=0, mean=None, min=None, ge080=0, ge095=0)
    return dict(count=len(p), mean=float(p.mean()), min=float(p.min()),
                ge080=int((p >= .8).sum()), ge095=int((p >= .95).sum()))


def key_redundancy(keys: torch.Tensor) -> dict:
    """Nearest same-bucket cosine; lower means less key duplication, not better IoU."""
    if keys.ndim != 2 or not bool(torch.isfinite(keys).all()):
        raise ValueError("Invalid projected keys for redundancy summary")
    n = len(keys)
    if n < 2:
        return dict(count=n, mean_nearest_cosine=None, max_nearest_cosine=None)
    normalized = torch.nn.functional.normalize(keys.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-float("inf"))
    nearest = similarity.max(-1).values
    return dict(count=n, mean_nearest_cosine=float(nearest.mean()),
                max_nearest_cosine=float(nearest.max()))
