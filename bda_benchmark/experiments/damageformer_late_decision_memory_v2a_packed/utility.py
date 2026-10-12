from __future__ import annotations
import torch
import torch.nn as nn

PIXEL_I = 1
PIXEL_D = 2

def iou_from_cm(cm):
    cm = cm.float()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    return tp / union.clamp_min(1.0)

def valid_miou(cm):
    return iou_from_cm(cm)[1:4].mean()

def intact_only_override_cm(building_cm):
    out = building_cm.clone()
    out[:, PIXEL_D] += out[:, PIXEL_I]
    out[:, PIXEL_I] = 0
    return out

def exact_delta(global_cm, building_cm):
    base = valid_miou(global_cm)
    new_b = intact_only_override_cm(building_cm)
    return valid_miou(global_cm - building_cm + new_b) - base

class UtilityVerifier(nn.Module):
    def __init__(self, in_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
