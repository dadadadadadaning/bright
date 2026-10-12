from __future__ import annotations
import torch
import torch.nn as nn

class SpecialistMLP(nn.Module):
    """Ranks Base=Intact candidates by missed-Damaged evidence."""
    def __init__(self, in_dim, hidden1=256, hidden2=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden1), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden1, hidden2), nn.GELU(),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def build_specialist_features(
    variant, base_probs_id_x, uncertainty,
    region_mean=None, region_std=None,
    critical_top1=None, critical_top2=None, peak_mean_delta=None,
):
    if uncertainty.ndim == 1:
        uncertainty = uncertainty[:, None]
    parts = [base_probs_id_x, uncertainty]
    variant = variant.upper()
    if variant in {"C1","C2","C3"}:
        if region_mean is None: raise ValueError("region_mean required")
        parts.append(region_mean)
    if variant in {"C2","C3"}:
        if region_std is None: raise ValueError("region_std required")
        parts.append(region_std)
    if variant == "C3":
        if any(x is None for x in (critical_top1, critical_top2, peak_mean_delta)):
            raise ValueError("C3 requires top1/top2/peak_mean_delta")
        if peak_mean_delta.ndim == 1:
            peak_mean_delta = peak_mean_delta[:, None]
        parts += [critical_top1, critical_top2, peak_mean_delta]
    if variant not in {"C0","C1","C2","C3"}:
        raise ValueError(variant)
    return torch.cat(parts, dim=-1)
