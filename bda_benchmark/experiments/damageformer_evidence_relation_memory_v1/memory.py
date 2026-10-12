from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .cache import global_region_mean
from .config import BLD_INTACT, BLD_DAMAGED, BLD_DESTROYED, DEFAULT_PROTOCOL
from .regions import PaddedRegions, local_pair_scores

@dataclass
class ClassMemory:
    support_rows: torch.Tensor      # rows in train cache
    keys: torch.Tensor              # normalized [Ns,128]
    labels: torch.Tensor
    event_ids: torch.Tensor

def build_class_memory(train_cache, fit_mask, event_ids, min_purity=0.70, control="none", seed=3407):
    y = train_cache["labels"].long().clone()
    ev = event_ids.clone()
    purity = train_cache["label_purity"].float()
    valid = fit_mask & (purity >= min_purity) & (y >= 0)
    rows = torch.where(valid)[0]

    g = torch.Generator().manual_seed(seed)
    if control == "shuffle_labels":
        perm = torch.randperm(len(rows), generator=g)
        y_rows = y[rows][perm]
    elif control == "swapped_id":
        y_rows = y[rows].clone()
        i = y_rows == BLD_INTACT; d = y_rows == BLD_DAMAGED
        y_rows[i] = BLD_DAMAGED; y_rows[d] = BLD_INTACT
    else:
        y_rows = y[rows]

    if control == "shuffle_events":
        perm = torch.randperm(len(rows), generator=g)
        ev_rows = ev[rows][perm]
    else:
        ev_rows = ev[rows]

    keys = F.normalize(global_region_mean(train_cache)[rows], dim=-1)
    return ClassMemory(rows, keys, y_rows, ev_rows)

def _class_positions(bank: ClassMemory, cls: int):
    return torch.where(bank.labels == cls)[0]

@torch.no_grad()
def _global_topk_batch(qkeys, qevents, bank, cls, topk, device):
    pos = _class_positions(bank, cls)
    k = bank.keys[pos].to(device, non_blocking=True)
    ev = bank.event_ids[pos].to(device, non_blocking=True)
    q = F.normalize(qkeys.to(device).float(), dim=-1)
    sim = q @ k.T
    sim = sim.masked_fill(ev[None,:] == qevents.to(device)[:,None], -1e9)
    vals, local_pos = torch.topk(sim, k=min(topk, sim.shape[1]), dim=1)
    bank_pos = pos[local_pos.cpu()]
    return vals.cpu(), bank_pos

@torch.no_grad()
def global_memory_scores(query_keys, query_events, bank, coarse_k=50, batch_size=512, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(query_keys)
    supports = {c: torch.empty(n) for c in (BLD_INTACT,BLD_DAMAGED,BLD_DESTROYED)}
    coarse = {c: [] for c in supports}
    for s in range(0,n,batch_size):
        e = min(n,s+batch_size)
        for c in supports:
            vals,pos = _global_topk_batch(query_keys[s:e], query_events[s:e], bank, c, coarse_k, device)
            supports[c][s:e] = vals.mean(1)
            coarse[c].append(pos)
    coarse = {c:torch.cat(v,0) for c,v in coarse.items()}
    return {
        "support_i":supports[BLD_INTACT],
        "support_d":supports[BLD_DAMAGED],
        "support_x":supports[BLD_DESTROYED],
        "delta_di":supports[BLD_DAMAGED]-supports[BLD_INTACT],
        "x_veto":supports[BLD_DESTROYED]-supports[BLD_DAMAGED],
        "coarse_positions":coarse,
    }

@torch.no_grad()
def local_memory_scores(
    query_rows, query_keys, query_events,
    train_cache, query_cache, bank: ClassMemory,
    train_regions: PaddedRegions, query_regions: PaddedRegions,
    weighted=False, coarse_k=50, fine_k=20, global_weight=0.35,
    control="none", seed=3407, batch_size=64, device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    n = len(query_rows)
    support = {c:torch.empty(n) for c in (BLD_INTACT,BLD_DAMAGED,BLD_DESTROYED)}

    # Coarse class-conditional candidates.
    global_out = global_memory_scores(query_keys, query_events, bank, coarse_k, batch_size=512, device=device)
    coarse = global_out["coarse_positions"]

    for s in range(0,n,batch_size):
        e = min(n,s+batch_size)
        qr = query_rows[s:e]
        qtok = query_regions.tokens[qr].to(device)
        qmask = query_regions.mask[qr].to(device)
        qcrit = query_regions.critical[qr].to(device)
        qlog = query_regions.logits[qr].to(device)

        for c in support:
            pos = coarse[c][s:e]                  # positions in bank
            source_rows = bank.support_rows[pos]  # rows in train cache
            if control == "wrong_regions":
                flat = source_rows.flatten()
                perm = torch.randperm(len(flat), generator=g)
                source_rows = flat[perm].view_as(source_rows)

            mtok = train_regions.tokens[source_rows].to(device)
            mmask = train_regions.mask[source_rows].to(device)
            mcrit = train_regions.critical[source_rows].to(device)
            mlog = train_regions.logits[source_rows].to(device)

            local = local_pair_scores(
                qtok,qmask,qcrit,qlog,mtok,mmask,mcrit,mlog,
                weighted=weighted,
                temperature=DEFAULT_PROTOCOL.region_temperature,
                beta_di=DEFAULT_PROTOCOL.region_beta_di,
                gamma_entropy=DEFAULT_PROTOCOL.region_gamma_entropy,
            )
            # Get matching global values for the same candidates.
            mk = bank.keys[pos].to(device)
            qn = F.normalize(query_keys[s:e].to(device).float(), dim=-1)
            gs = torch.einsum("bd,bmd->bm", qn, mk.float())
            total = global_weight*gs + (1.0-global_weight)*local
            vals = torch.topk(total, k=min(fine_k,total.shape[1]), dim=1).values
            support[c][s:e] = vals.mean(1).cpu()

    return {
        "support_i":support[BLD_INTACT],
        "support_d":support[BLD_DAMAGED],
        "support_x":support[BLD_DESTROYED],
        "delta_di":support[BLD_DAMAGED]-support[BLD_INTACT],
        "x_veto":support[BLD_DESTROYED]-support[BLD_DAMAGED],
    }
