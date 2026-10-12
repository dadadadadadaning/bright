from __future__ import annotations
import torch
import torch.nn as nn
from .schema import LABEL_I, LABEL_D

class UtilityVerifier(nn.Module):
    def __init__(self, in_dim=12, hidden1=32, hidden2=16, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden1), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden1, hidden2), nn.GELU(),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def override_intact_to_damaged(building_conf, intact_col=LABEL_I, damaged_col=LABEL_D):
    out = building_conf.clone()
    out[:, damaged_col] += out[:, intact_col]
    out[:, intact_col] = 0
    return out

def iou_from_confusion(conf):
    conf = conf.float()
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    return tp / union.clamp_min(1.0)

def valid_miou_from_confusion(conf):
    return iou_from_confusion(conf)[1:4].mean()

def exact_utility_delta(global_conf, building_conf):
    base = valid_miou_from_confusion(global_conf)
    new_b = override_intact_to_damaged(building_conf)
    after = valid_miou_from_confusion(global_conf - building_conf + new_b)
    return after - base

def make_utility_features(specialist_score, base_uncertainty, proto, memory, mean_align_conf, peak_delta):
    vals = [
        specialist_score, base_uncertainty,
        proto["proto_delta_DI"], proto["proto_X_veto"],
        memory["delta_di"], memory["x_veto"], memory["agreement"],
        memory["event_diversity"], memory["effective_neighbors"], memory["score_margin"],
        mean_align_conf, peak_delta
    ]
    return torch.stack([v.float().reshape(()) for v in vals])
