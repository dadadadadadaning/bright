from __future__ import annotations
import hashlib, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import building_area
from .metrics import exact_single_utility

class UtilityVerifier(nn.Module):
    """Small action-value verifier.

    It has:
      sign_head: P(delta utility > 0)
      delta_head: regression of standardized exact pixel utility.
    """
    def __init__(self,in_dim):
        super().__init__()
        self.trunk=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64,32), nn.GELU(),
        )
        self.sign_head=nn.Linear(32,1)
        self.delta_head=nn.Linear(32,1)
    def forward(self,x):
        h=self.trunk(x)
        return self.sign_head(h).squeeze(-1), self.delta_head(h).squeeze(-1)

def utility_features(cache, rows, specialist_scores, hist_part):
    bp=cache["base_probs"].float()[rows]
    area=torch.log1p(building_area(cache)[rows].float())
    cols=[
        specialist_scores[rows].float(),
        hist_part["prototype"]["delta_di"].float(),
        hist_part["prototype"]["x_veto"].float(),
        hist_part["global_memory"]["delta_di"].float(),
        hist_part["global_memory"]["x_veto"].float(),
        hist_part["local_memory"]["delta_di"].float(),
        hist_part["local_memory"]["x_veto"].float(),
        hist_part["weighted_local_memory"]["delta_di"].float(),
        hist_part["weighted_local_memory"]["x_veto"].float(),
        cache["uncertainty"].float()[rows],
        cache["peak_mean_delta"].float()[rows],
        bp[:,1]-bp[:,0],
        area,
    ]
    return torch.stack(cols,dim=1)

def utility_targets(cache,rows):
    return torch.tensor([
        100.0*exact_single_utility(cache["base_pixel_cm"],cache["building_pixel_cm"][int(i)])
        for i in rows
    ],dtype=torch.float32)

def split_rows_by_image(cache, rows, seed=3407, train_ratio=0.80):
    # Utility gets only Train-Cal rows; split again by image for checkpoint selection.
    names=sorted(set(cache["names"][int(i)] for i in rows))
    rng=random.Random(seed+99173); rng.shuffle(names)
    cut=max(1,min(len(names)-1,int(round(len(names)*train_ratio)))) if len(names)>1 else len(names)
    trn=set(names[:cut])
    m=torch.tensor([cache["names"][int(i)] in trn for i in rows],dtype=torch.bool)
    return m,~m

def normalize_fit(x,fit_mask):
    mean=x[fit_mask].mean(0)
    std=x[fit_mask].std(0).clamp_min(1e-5)
    return (x-mean)/std,mean,std
