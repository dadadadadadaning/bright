from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class PaddedRegions:
    tokens: torch.Tensor      # [N,K,128], CPU float16
    mask: torch.Tensor        # [N,K], bool
    critical: torch.Tensor    # [N,K], float16
    logits: torch.Tensor      # [N,K,3], float16

def build_padded_regions(cache: dict, max_regions: int | None = None) -> PaddedRegions:
    n = len(cache["labels"])
    kmax = int(cache["region_count"].max()) if max_regions is None else int(max_regions)
    d = cache["region_tokens"].shape[1]
    tokens = torch.zeros((n,kmax,d), dtype=torch.float16)
    critical = torch.zeros((n,kmax), dtype=torch.float16)
    logits = torch.zeros((n,kmax,3), dtype=torch.float16)
    mask = torch.zeros((n,kmax), dtype=torch.bool)
    off = cache["region_offsets"]
    for i in range(n):
        s,e = int(off[i]), int(off[i+1])
        k = min(kmax, e-s)
        if k <= 0: continue
        tokens[i,:k] = cache["region_tokens"][s:s+k].to(torch.float16)
        critical[i,:k] = cache["region_critical"][s:s+k].to(torch.float16)
        logits[i,:k] = cache["region_logits"][s:s+k].to(torch.float16)
        mask[i,:k] = True
    return PaddedRegions(tokens,mask,critical,logits)

def region_weights(critical, logits, mask, temperature=0.25, beta_di=0.50, gamma_entropy=0.25):
    """region_logits order is assumed [I,D,X], consistent with current 3-way head."""
    p = torch.softmax(logits.float(), dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(-1)
    delta_di = p[...,1] - p[...,0]
    score = critical.float() + beta_di*delta_di - gamma_entropy*ent
    score = score.masked_fill(~mask, -1e9)
    w = torch.softmax(score/temperature, dim=-1) * mask.float()
    return w / w.sum(-1, keepdim=True).clamp_min(1e-8)

def local_pair_scores(
    q_tokens, q_mask, q_critical, q_logits,
    m_tokens, m_mask, m_critical, m_logits,
    weighted: bool,
    temperature=0.25, beta_di=0.50, gamma_entropy=0.25,
):
    """Vectorized B query x M candidate region-set scores.

    q_tokens [B,K,D]
    m_tokens [B,M,K,D]
    returns [B,M]
    """
    q = F.normalize(q_tokens.float(), dim=-1)
    m = F.normalize(m_tokens.float(), dim=-1)
    sim = torch.einsum("bkd,bmld->bmkl", q, m)
    valid = q_mask[:,None,:,None] & m_mask[:,:,None,:]
    sim = sim.masked_fill(~valid, -1e4)

    qbest = sim.max(-1).values   # [B,M,Kq]
    mbest = sim.max(-2).values   # [B,M,Km]

    if weighted:
        qw = region_weights(q_critical, q_logits, q_mask, temperature, beta_di, gamma_entropy)
        mw = region_weights(m_critical, m_logits, m_mask, temperature, beta_di, gamma_entropy)
        q2m = (qbest * qw[:,None,:]).sum(-1)
        m2q = (mbest * mw).sum(-1)
    else:
        qw = q_mask.float() / q_mask.float().sum(-1,keepdim=True).clamp_min(1)
        mw = m_mask.float() / m_mask.float().sum(-1,keepdim=True).clamp_min(1)
        q2m = (qbest * qw[:,None,:]).sum(-1)
        m2q = (mbest * mw).sum(-1)
    return 0.5*(q2m+m2q)
