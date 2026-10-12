"""Pure selection and validation helpers for a quality-prioritized Train bank.

Q remains the frozen SRAC partition.  A label is the original four-way region
fraction, not a building identity or a hardened class assigned to every pixel.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from pathlib import Path

import torch


FORMAT = "SRAC_MEMORY_BANK_V2"
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


def purity_summary(distribution: torch.Tensor) -> dict:
    p = distribution.float().max(-1).values
    if len(p) == 0:
        return dict(count=0, mean=None, min=None, ge080=0, ge095=0)
    return dict(count=len(p), mean=float(p.mean()), min=float(p.min()),
                ge080=int((p >= .8).sum()), ge095=int((p >= .95).sum()))
