from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

# Building-level labels in the current Stage1 cache.
BUILDING_UNKNOWN = -1
BUILDING_I = 0
BUILDING_D = 1
BUILDING_X = 2

# Pixel-level four-class confusion matrix labels.
PIXEL_BG = 0
PIXEL_I = 1
PIXEL_D = 2
PIXEL_X = 3

EXPECTED_SHA = "0b76a887a16d4c1c5371cba1570a3a6c641435ef2c97494363323b73aa4239f7"


def load_stage1_cache(path: str, require_sha: bool = True) -> dict:
    x = torch.load(path, map_location="cpu", weights_only=False)

    required = [
        "features", "labels", "label_purity", "names", "events",
        "pred_building_ids", "base_probs", "base_pred", "uncertainty",
        "peak_mean_delta", "region_count", "region_offsets",
        "region_tokens", "region_critical", "region_logits",
        "base_pixel_cm", "building_pixel_cm",
    ]
    missing = [k for k in required if k not in x]
    if missing:
        raise KeyError(f"Missing cache fields: {missing}")

    if require_sha:
        sha = x.get("source_checkpoint_sha256")
        if sha != EXPECTED_SHA:
            raise RuntimeError(f"Unexpected Stage1 SHA: {sha}")

    if x.get("test_accessed", False):
        raise RuntimeError("Refusing cache with test_accessed=True")

    offsets = x["region_offsets"]
    counts = x["region_count"]
    if len(offsets) != len(x["labels"]) + 1:
        raise RuntimeError("region_offsets length mismatch")
    if int(offsets[-1]) != len(x["region_tokens"]):
        raise RuntimeError("region_offsets tail mismatch")
    if not torch.equal(offsets[1:] - offsets[:-1], counts):
        raise RuntimeError("region_offsets/count mismatch")

    return x


def encode_events(events: List[str]) -> Tuple[torch.Tensor, Dict[str, int]]:
    names = sorted(set(events))
    mapping = {name: i for i, name in enumerate(names)}
    ids = torch.tensor([mapping[e] for e in events], dtype=torch.long)
    return ids, mapping


def global_keys(cache: dict) -> torch.Tensor:
    """C1-stable key = first 128 dims = RegionToken mean."""
    f = cache["features"].float()
    if f.shape[1] != 512:
        raise ValueError(f"Expected 512D features, got {tuple(f.shape)}")
    return F.normalize(f[:, :128], dim=-1)


def split_c3_features(cache: dict) -> Dict[str, torch.Tensor]:
    f = cache["features"].float()
    return {
        "region_mean": f[:, 0:128],
        "region_std": f[:, 128:256],
        "top1_critical": f[:, 256:384],
        "top2_critical": f[:, 384:512],
        "peak_mean_delta": cache["peak_mean_delta"].float(),
    }


def get_regions(cache: dict, building_index: int) -> Dict[str, torch.Tensor]:
    o = cache["region_offsets"]
    s = int(o[building_index])
    e = int(o[building_index + 1])
    return {
        "tokens": cache["region_tokens"][s:e].float(),
        "critical": cache["region_critical"][s:e].float(),
        "logits": cache["region_logits"][s:e].float(),
    }


def region_evidence(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Current region_logits are 3-way building-state evidence I/D/X."""
    p = torch.softmax(logits.float(), dim=-1)
    p_i, p_d, p_x = p[:, 0], p[:, 1], p[:, 2]
    entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(-1)
    return {
        "prob": p,
        "p_i": p_i,
        "p_d": p_d,
        "p_x": p_x,
        "delta_di": p_d - p_i,
        "entropy": entropy,
    }


def region_weights(
    critical: torch.Tensor,
    logits: torch.Tensor,
    temperature: float = 0.25,
    beta_di: float = 0.50,
    gamma_entropy: float = 0.25,
) -> torch.Tensor:
    e = region_evidence(logits)
    score = critical + beta_di * e["delta_di"] - gamma_entropy * e["entropy"]
    return torch.softmax(score / temperature, dim=0)


def symmetric_local_score(
    q_tokens: torch.Tensor,
    q_critical: torch.Tensor,
    q_logits: torch.Tensor,
    m_tokens: torch.Tensor,
    m_critical: torch.Tensor,
    m_logits: torch.Tensor,
    weighted: bool = True,
) -> torch.Tensor:
    q = F.normalize(q_tokens.float(), dim=-1)
    m = F.normalize(m_tokens.float(), dim=-1)
    sim = q @ m.T

    if weighted:
        qw = region_weights(q_critical, q_logits)
        mw = region_weights(m_critical, m_logits)
    else:
        qw = torch.full((len(q),), 1.0 / max(1, len(q)), device=q.device)
        mw = torch.full((len(m),), 1.0 / max(1, len(m)), device=m.device)

    q2m = (qw * sim.max(dim=1).values).sum()
    m2q = (mw * sim.max(dim=0).values).sum()
    return 0.5 * (q2m + m2q)


@dataclass
class PackedMemory:
    global_keys: torch.Tensor
    support_rows: torch.Tensor
    labels: torch.Tensor
    purity: torch.Tensor
    event_ids: torch.Tensor
    event_map: Dict[str, int]
    names: List[str]
    pred_building_ids: torch.Tensor
    source_cache: dict


def build_train_memory(
    cache: dict,
    min_purity: float = 0.70,
    include_destroyed: bool = True,
) -> PackedMemory:
    labels = cache["labels"].long()
    purity = cache["label_purity"].float()

    valid = labels >= 0
    valid &= purity >= min_purity
    if not include_destroyed:
        valid &= labels != BUILDING_X

    rows = torch.where(valid)[0]
    eids, event_map = encode_events(cache["events"])
    keys = global_keys(cache)

    return PackedMemory(
        global_keys=keys[rows],
        support_rows=rows,
        labels=labels[rows],
        purity=purity[rows],
        event_ids=eids[rows],
        event_map=event_map,
        names=[cache["names"][int(i)] for i in rows],
        pred_building_ids=cache["pred_building_ids"][rows].long(),
        source_cache=cache,
    )


@torch.no_grad()
def coarse_topk(
    query_key: torch.Tensor,
    query_event_id: int,
    memory: PackedMemory,
    topk: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(query_key.float(), dim=-1)
    s = memory.global_keys @ q
    s = s.masked_fill(memory.event_ids == int(query_event_id), float("-inf"))
    valid = int(torch.isfinite(s).sum())
    if valid == 0:
        return s.new_empty((0,)), torch.empty(0, dtype=torch.long)
    vals, pos = torch.topk(s, min(topk, valid))
    return vals, pos


@torch.no_grad()
def rerank_topk(
    query_cache: dict,
    query_row: int,
    memory: PackedMemory,
    coarse_scores: torch.Tensor,
    coarse_pos: torch.Tensor,
    fine_k: int = 20,
    global_weight: float = 0.35,
    weighted_local: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qr = get_regions(query_cache, query_row)
    scores = []

    for gs, p in zip(coarse_scores, coarse_pos):
        source_row = int(memory.support_rows[int(p)])
        mr = get_regions(memory.source_cache, source_row)
        ls = symmetric_local_score(
            qr["tokens"], qr["critical"], qr["logits"],
            mr["tokens"], mr["critical"], mr["logits"],
            weighted=weighted_local,
        )
        scores.append(global_weight * gs + (1.0 - global_weight) * ls)

    if not scores:
        return coarse_scores, coarse_pos

    scores = torch.stack(scores)
    vals, order = torch.topk(scores, min(fine_k, len(scores)))
    return vals, coarse_pos[order]


@torch.no_grad()
def support_summary(
    scores: torch.Tensor,
    positions: torch.Tensor,
    memory: PackedMemory,
    temperature: float = 0.10,
) -> Dict[str, torch.Tensor]:
    if len(scores) == 0:
        z = torch.tensor(0.0)
        return {
            "support_i": z, "support_d": z, "support_x": z,
            "delta_di": z, "x_veto": z,
            "agreement": z, "event_diversity": z,
            "effective_neighbors": z,
        }

    labels = memory.labels[positions]
    events = memory.event_ids[positions]
    w = torch.softmax(scores.float() / temperature, dim=0)

    def sup(c):
        return (w * (labels == c).float()).sum()

    si, sd, sx = sup(BUILDING_I), sup(BUILDING_D), sup(BUILDING_X)
    return {
        "support_i": si,
        "support_d": sd,
        "support_x": sx,
        "delta_di": sd - si,
        "x_veto": sx - sd,
        "agreement": torch.stack([si, sd, sx]).max(),
        "event_diversity": scores.new_tensor(
            torch.unique(events).numel() / max(1, len(events))
        ),
        "effective_neighbors": 1.0 / (w.square().sum() + 1e-8),
    }
