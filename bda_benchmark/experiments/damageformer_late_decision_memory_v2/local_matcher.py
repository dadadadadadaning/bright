from __future__ import annotations
import torch
import torch.nn.functional as F

def region_reliability_weights(meta, mask, tau=0.25, entropy_penalty=0.25, area_log_weight=0.10):
    area = meta[...,0].clamp_min(0)
    conf = meta[...,1].clamp(0,1)
    ent = meta[...,2].clamp(0,1)
    peak = meta[...,3]
    score = peak * conf - entropy_penalty * ent + area_log_weight * torch.log(area + 1e-6)
    score = score.masked_fill(~mask.bool(), float("-inf"))
    all_invalid = ~mask.bool().any(dim=-1, keepdim=True)
    score = torch.where(all_invalid, torch.zeros_like(score), score)
    w = torch.softmax(score / tau, dim=-1) * mask.float()
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

def symmetric_region_score(q_tokens, q_mask, q_meta, m_tokens, m_mask, m_meta, tau=0.25):
    qmask, mmask = q_mask.bool(), m_mask.bool()
    if qmask.sum() == 0 or mmask.sum() == 0:
        return q_tokens.new_tensor(-1.0, dtype=torch.float32)
    q = F.normalize(q_tokens[qmask].float(), dim=-1)
    m = F.normalize(m_tokens[mmask].float(), dim=-1)
    sim = q @ m.T
    qw = region_reliability_weights(q_meta.unsqueeze(0), qmask.unsqueeze(0), tau=tau)[0][qmask]
    mw = region_reliability_weights(m_meta.unsqueeze(0), mmask.unsqueeze(0), tau=tau)[0][mmask]
    q2m = (qw * sim.max(1).values).sum()
    m2q = (mw * sim.max(0).values).sum()
    return 0.5 * (q2m + m2q)
