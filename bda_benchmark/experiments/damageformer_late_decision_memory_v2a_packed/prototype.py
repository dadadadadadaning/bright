from __future__ import annotations
import torch
import torch.nn.functional as F
from .packed_memory import BUILDING_I, BUILDING_D, BUILDING_X

@torch.no_grad()
def event_balanced_class_prototype(query_event_id, memory, label):
    valid = (memory.labels == label) & (memory.event_ids != int(query_event_id))
    events = torch.unique(memory.event_ids[valid])
    protos = []
    for e in events:
        idx = valid & (memory.event_ids == e)
        if idx.any():
            protos.append(F.normalize(memory.global_keys[idx].float().mean(0), dim=0))
    if not protos:
        raise RuntimeError(f"No support for label={label}")
    return F.normalize(torch.stack(protos).mean(0), dim=0)

@torch.no_grad()
def prototype_scores(query_key, query_event_id, memory):
    q = F.normalize(query_key.float(), dim=-1)
    pi = event_balanced_class_prototype(query_event_id, memory, BUILDING_I)
    pd = event_balanced_class_prototype(query_event_id, memory, BUILDING_D)
    px = event_balanced_class_prototype(query_event_id, memory, BUILDING_X)
    si, sd, sx = q @ pi, q @ pd, q @ px
    return {
        "s_i": si, "s_d": sd, "s_x": sx,
        "delta_di": sd - si,
        "x_veto": sx - sd,
    }
