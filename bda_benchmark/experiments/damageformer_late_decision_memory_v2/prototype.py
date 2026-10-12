from __future__ import annotations
import torch
import torch.nn.functional as F
from .schema import LABEL_I, LABEL_D, LABEL_X

@torch.no_grad()
def event_balanced_prototype(bank, query_event, label):
    valid = (bank.labels == int(label)) & (bank.event_ids != int(query_event))
    events = torch.unique(bank.event_ids[valid])
    protos = []
    for e in events:
        idx = valid & (bank.event_ids == e)
        if idx.any():
            protos.append(F.normalize(bank.keys[idx].float().mean(0), dim=0))
    if not protos:
        raise RuntimeError(f"No support for label={label}, event={query_event}")
    return F.normalize(torch.stack(protos).mean(0), dim=0)

@torch.no_grad()
def prototype_scores(query_key, query_event, bank):
    q = F.normalize(query_key.float(), dim=-1)
    pi = event_balanced_prototype(bank, query_event, LABEL_I)
    pd = event_balanced_prototype(bank, query_event, LABEL_D)
    px = event_balanced_prototype(bank, query_event, LABEL_X)
    si, sd, sx = q @ pi, q @ pd, q @ px
    return {
        "proto_I": si, "proto_D": sd, "proto_X": sx,
        "proto_delta_DI": sd - si,
        "proto_X_veto": sx - sd,
    }
