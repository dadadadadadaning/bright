from __future__ import annotations
import torch

def pool_soft_superpixels_for_memory(
    pixel_feat, assignment, align_confidence,
    match_entropy, evidence_score, tau_peak=0.25, eps=1e-6,
):
    """Convert soft superpixels Q_pk into local Memory tokens."""
    q = assignment.float().clamp_min(0)
    mass = q.sum(0).clamp_min(eps)
    token = (q.T @ pixel_feat.float()) / mass[:, None]
    evidence_mean = (q.T @ evidence_score.float()) / mass
    conf_mean = (q.T @ align_confidence.float()) / mass
    ent_mean = (q.T @ match_entropy.float()) / mass

    logits = evidence_score.float()[:, None] / tau_peak
    peak_w = q * torch.exp(logits - logits.max())
    peak_w = peak_w / peak_w.sum(0, keepdim=True).clamp_min(eps)
    peak = (peak_w * evidence_score.float()[:, None]).sum(0)
    peak_delta = peak - evidence_mean
    area_ratio = mass / mass.sum().clamp_min(eps)

    meta = torch.stack([area_ratio, conf_mean, ent_mean, peak_delta], dim=-1)
    return {
        "region_tokens": token,
        "region_meta": meta,
        "region_mass": mass,
        "region_mask": mass > eps * 10,
    }

def building_region_membership(building_mask_flat, assignment, eps=1e-6):
    mass = building_mask_flat.float() @ assignment.float()
    return mass / mass.sum(1, keepdim=True).clamp_min(eps)
