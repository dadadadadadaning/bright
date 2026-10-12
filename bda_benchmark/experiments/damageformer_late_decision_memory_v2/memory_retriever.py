from __future__ import annotations
import torch
import torch.nn.functional as F
from .schema import LABEL_I, LABEL_D, LABEL_X
from .local_matcher import symmetric_region_score

def masked_mean(x, mask, eps=1e-6):
    w = mask.float().unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp_min(eps)

def build_global_keys(region_tokens, region_mask):
    return F.normalize(masked_mean(region_tokens.float(), region_mask.bool()), dim=-1)

@torch.no_grad()
def coarse_retrieve(query_key, query_event, bank, topk=50):
    q = F.normalize(query_key.float(), dim=-1)
    scores = F.normalize(bank.keys.float(), dim=-1) @ q
    scores = scores.masked_fill(bank.event_ids == int(query_event), float("-inf"))
    valid = int(torch.isfinite(scores).sum())
    if valid == 0:
        return scores.new_empty((0,)), torch.empty(0, dtype=torch.long, device=scores.device)
    vals, idx = torch.topk(scores, k=min(topk, valid))
    return vals, idx

@torch.no_grad()
def local_rerank(q_tokens, q_mask, q_meta, coarse_indices, coarse_scores, bank, topk=20, global_weight=0.35):
    if coarse_indices.numel() == 0:
        return coarse_scores, coarse_indices
    combined = []
    for gs, idx_t in zip(coarse_scores, coarse_indices):
        idx = int(idx_t)
        ls = symmetric_region_score(
            q_tokens, q_mask, q_meta,
            bank.region_tokens[idx], bank.region_mask[idx], bank.region_meta[idx]
        )
        combined.append(global_weight * gs + (1-global_weight) * ls)
    combined = torch.stack(combined)
    vals, order = torch.topk(combined, k=min(topk, combined.numel()))
    return vals, coarse_indices[order]

@torch.no_grad()
def memory_support(scores, indices, bank, temperature=0.10):
    z = bank.keys.new_tensor(0.0)
    if indices.numel() == 0:
        return dict(
            support_intact=z, support_damaged=z, support_destroyed=z,
            delta_di=z, x_veto=z, agreement=z, event_diversity=z,
            effective_neighbors=z, score_margin=z
        )
    labels, events = bank.labels[indices], bank.event_ids[indices]
    w = torch.softmax(scores.float()/temperature, dim=0)
    sup = lambda lab: (w * (labels == lab).float()).sum()
    si, sd, sx = sup(LABEL_I), sup(LABEL_D), sup(LABEL_X)
    unique_events = torch.unique(events).numel()
    sorted_scores = torch.sort(scores.float(), descending=True).values
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else z
    return dict(
        support_intact=si, support_damaged=sd, support_destroyed=sx,
        delta_di=sd-si, x_veto=sx-sd,
        agreement=torch.stack([si,sd,sx]).max(),
        event_diversity=scores.new_tensor(unique_events/max(1,int(indices.numel()))),
        effective_neighbors=1.0/(w.square().sum()+1e-8),
        score_margin=margin,
    )

@torch.no_grad()
def query_memory(query_key, query_tokens, query_mask, query_meta, query_event, bank,
                 coarse_k=50, fine_k=20, global_weight=0.35, temperature=0.10):
    cs, ci = coarse_retrieve(query_key, query_event, bank, coarse_k)
    fs, fi = local_rerank(query_tokens, query_mask, query_meta, ci, cs, bank, fine_k, global_weight)
    out = memory_support(fs, fi, bank, temperature)
    out["neighbor_indices"], out["neighbor_scores"] = fi, fs
    return out
