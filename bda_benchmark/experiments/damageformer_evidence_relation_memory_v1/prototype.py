from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .cache import global_region_mean
from .config import BLD_INTACT, BLD_DAMAGED, BLD_DESTROYED

@dataclass
class PrototypeBank:
    keys: torch.Tensor
    labels: torch.Tensor
    event_ids: torch.Tensor
    support_rows: torch.Tensor

def build_prototype_bank(cache, fit_mask, event_ids, min_purity=0.70):
    y = cache["labels"].long()
    purity = cache["label_purity"].float()
    valid = fit_mask & (purity >= min_purity) & (y >= 0)
    rows = torch.where(valid)[0]
    keys = F.normalize(global_region_mean(cache)[rows], dim=-1)
    return PrototypeBank(keys, y[rows], event_ids[rows], rows)

@torch.no_grad()
def _class_proto(bank, query_event, label, event_balanced=False):
    valid = (bank.labels == label) & (bank.event_ids != int(query_event))
    if not valid.any():
        raise RuntimeError(f"No prototype support for class={label}, event={query_event}")
    if not event_balanced:
        return F.normalize(bank.keys[valid].mean(0), dim=0)
    protos = []
    for e in torch.unique(bank.event_ids[valid]):
        idx = valid & (bank.event_ids == e)
        protos.append(F.normalize(bank.keys[idx].mean(0), dim=0))
    return F.normalize(torch.stack(protos).mean(0), dim=0)

@torch.no_grad()
def score_queries(query_keys, query_event_ids, bank, event_balanced=False):
    q = F.normalize(query_keys.float(), dim=-1)
    out_i = torch.empty(len(q)); out_d = torch.empty(len(q)); out_x = torch.empty(len(q))
    # event-specific prototypes are reusable.
    cache = {}
    for e in torch.unique(query_event_ids).tolist():
        cache[e] = {
            c:_class_proto(bank,e,c,event_balanced)
            for c in (BLD_INTACT,BLD_DAMAGED,BLD_DESTROYED)
        }
    for e in torch.unique(query_event_ids).tolist():
        rows = torch.where(query_event_ids == e)[0]
        qe = q[rows]
        out_i[rows] = qe @ cache[e][BLD_INTACT]
        out_d[rows] = qe @ cache[e][BLD_DAMAGED]
        out_x[rows] = qe @ cache[e][BLD_DESTROYED]
    return {
        "support_i":out_i, "support_d":out_d, "support_x":out_x,
        "delta_di":out_d-out_i, "x_veto":out_x-out_d,
    }
