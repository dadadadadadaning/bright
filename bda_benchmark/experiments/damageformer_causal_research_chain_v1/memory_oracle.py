#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .cache_features import region_slice, wrong_region_feature
from .core import D, I, binary_auroc, cache_fingerprint, load_cache, normalize, write_json
from .representation import transform_features


def cosine_topk_chunked(
    query: torch.Tensor,
    memory: torch.Tensor,
    k: int,
    *,
    query_events: List[str] | None = None,
    memory_events: List[str] | None = None,
    exclude_same_event: bool = False,
    device: str = "cuda:0",
    query_chunk: int = 128,
    memory_chunk: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k cosine with bounded memory; never materializes QxM."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    q_all = normalize(query.float()).cpu()
    m_all = normalize(memory.float()).cpu()
    k = min(int(k), int(m_all.shape[0]))
    all_score, all_idx = [], []
    for qs in range(0, len(q_all), query_chunk):
        qe = min(len(q_all), qs + query_chunk)
        q = q_all[qs:qe].to(dev)
        best_s = torch.full((len(q), k), -1e4, device=dev)
        best_i = torch.full((len(q), k), -1, dtype=torch.long, device=dev)
        for ms in range(0, len(m_all), memory_chunk):
            me = min(len(m_all), ms + memory_chunk)
            m = m_all[ms:me].to(dev)
            sim = q @ m.T
            if exclude_same_event:
                if query_events is None or memory_events is None:
                    raise ValueError("events required for same-event exclusion")
                invalid = torch.tensor(
                    [[str(query_events[qs + qi]) == str(memory_events[mi]) for mi in range(ms, me)] for qi in range(qe - qs)],
                    dtype=torch.bool, device=dev,
                )
                sim = sim.masked_fill(invalid, -1e4)
            idx = torch.arange(ms, me, device=dev).view(1, -1).expand(len(q), -1)
            cand_s = torch.cat((best_s, sim), dim=1)
            cand_i = torch.cat((best_i, idx), dim=1)
            best_s, order = cand_s.topk(k=k, dim=1)
            best_i = cand_i.gather(1, order)
        all_score.append(best_s.cpu())
        all_idx.append(best_i.cpu())
    return torch.cat(all_score), torch.cat(all_idx)


def local_set_score(qtok: torch.Tensor, mtok: torch.Tensor, qcrit: torch.Tensor, mcrit: torch.Tensor, top_regions: int = 4) -> float:
    qorder = torch.argsort(qcrit, descending=True)[:top_regions]
    morder = torch.argsort(mcrit, descending=True)[:top_regions]
    q = normalize(qtok[qorder].float())
    m = normalize(mtok[morder].float())
    if len(q) == 0 or len(m) == 0:
        return -1.0
    sim = q @ m.T
    # symmetric Chamfer-like cosine
    return float(0.5 * (sim.max(1).values.mean() + sim.max(0).values.mean()))


def feature_matrix(cache: Dict[str, object], mode: str) -> torch.Tensor:
    if mode == "building":
        return cache["features"].float()
    rows = []
    for i in range(len(cache["labels"])):
        tok, crit = region_slice(cache, i)
        rows.append(wrong_region_feature(tok, crit))
    return torch.stack(rows)


def apply_control(train: Dict[str, object], control: str, seed: int) -> Tuple[torch.Tensor, List[str]]:
    labels = train["labels"].clone().long()
    events = list(train["events"])
    g = torch.Generator().manual_seed(seed)
    if control == "shuffle_labels":
        labels = labels[torch.randperm(len(labels), generator=g)]
    elif control == "shuffle_events":
        perm = torch.randperm(len(events), generator=g).tolist()
        events = [events[i] for i in perm]
    elif control not in ("none", "wrong_regions"):
        raise ValueError(control)
    return labels, events


def retrieval_delta(scores: torch.Tensor, idx: torch.Tensor, memory_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = idx.shape[0]
    delta = torch.empty(n)
    agreement = torch.empty(n)
    top1_label = torch.empty(n, dtype=torch.long)
    for i in range(n):
        lab = memory_labels[idx[i]]
        sim = scores[i]
        d = sim[lab == D]
        ii = sim[lab == I]
        sd = float(d.mean()) if len(d) else -1.0
        si = float(ii.mean()) if len(ii) else -1.0
        delta[i] = sd - si
        agreement[i] = float((lab == D).float().mean())
        top1_label[i] = int(lab[0])
    return delta, agreement, top1_label


def main() -> None:
    p = argparse.ArgumentParser("Region-aware instance memory oracle")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--coarse_k", type=int, default=50)
    p.add_argument("--local_top_regions", type=int, default=4)
    p.add_argument("--local_weight", type=float, default=0.35)
    p.add_argument("--exclude_same_event", action="store_true")
    p.add_argument("--control", choices=("none", "shuffle_labels", "shuffle_events", "wrong_regions"), default="none")
    p.add_argument("--tail_ckpt", default="", help="Optional Stage-B checkpoint; coarse memory retrieval uses its projected embedding.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    tr = load_cache(args.train_cache, expected_split="train")
    va = load_cache(args.val_cache, expected_split="val")
    feature_mode = "wrong_regions" if args.control == "wrong_regions" else "building"
    xtr_raw = feature_matrix(tr, feature_mode)
    xva_raw = feature_matrix(va, feature_mode)
    if args.control == "wrong_regions":
        xtr, xva = xtr_raw, xva_raw
    else:
        xtr, _ = transform_features(xtr_raw, args.tail_ckpt, args.device)
        xva, _ = transform_features(xva_raw, args.tail_ckpt, args.device)
    memory_labels, memory_events = apply_control(tr, args.control, args.seed)
    yva = va["labels"].long()
    base_pred = va["base_probs"].argmax(1)
    candidate = (base_pred == I) & ((yva == I) | (yva == D))
    qids = torch.nonzero(candidate, as_tuple=False).flatten()
    if qids.numel() == 0:
        raise RuntimeError("no base-Intact I/D validation candidates")

    qfeat = xva[qids]
    coarse_scores, coarse_idx = cosine_topk_chunked(
        qfeat, xtr, args.coarse_k,
        query_events=[va["events"][i] for i in qids.tolist()],
        memory_events=memory_events,
        exclude_same_event=args.exclude_same_event,
        device=args.device,
    )

    # optional local reranking using superpixel region sets
    final_scores = []
    final_indices = []
    for qi, vid in enumerate(qids.tolist()):
        cidx = coarse_idx[qi]
        gscore = coarse_scores[qi]
        if args.control == "wrong_regions":
            comb = gscore
        else:
            qtok, qcrit = region_slice(va, vid)
            local = []
            for mid in cidx.tolist():
                mtok, mcrit = region_slice(tr, mid)
                local.append(local_set_score(qtok, mtok, qcrit, mcrit, args.local_top_regions))
            local_t = torch.tensor(local, dtype=torch.float32)
            comb = (1.0 - args.local_weight) * gscore + args.local_weight * local_t
        order = torch.argsort(comb, descending=True)[: min(args.topk, len(comb))]
        final_scores.append(comb[order])
        final_indices.append(cidx[order])
    final_scores = torch.stack(final_scores)
    final_indices = torch.stack(final_indices)
    delta, agreement, top1_label = retrieval_delta(final_scores, final_indices, memory_labels)
    qlabels = yva[qids]
    auc = binary_auroc(delta, (qlabels == D).long())
    missed_mask = qlabels == D

    recall = {}
    for k in (1, 3, 5, 10, 20):
        kk = min(k, final_indices.shape[1])
        has_d = []
        for i in torch.nonzero(missed_mask, as_tuple=False).flatten().tolist():
            lab = memory_labels[final_indices[i, :kk]]
            has_d.append(bool((lab == D).any()))
        recall[f"recall_at_{k}"] = float(sum(has_d) / max(1, len(has_d)))

    event_diversity = []
    for i in range(len(qids)):
        es = [memory_events[j] for j in final_indices[i].tolist()]
        event_diversity.append(len(set(es)))

    out = {
        "protocol": "CAUSAL_CHAIN_STAGE_E_MEMORY_ORACLE",
        "train_cache": cache_fingerprint(args.train_cache),
        "val_cache": cache_fingerprint(args.val_cache),
        "control": args.control,
        "exclude_same_event": bool(args.exclude_same_event),
        "feature_mode": feature_mode,
        "coarse_k": args.coarse_k,
        "topk": args.topk,
        "local_weight": args.local_weight,
        "tail_checkpoint": args.tail_ckpt or None,
        "candidate_count": int(len(qids)),
        "candidate_missed_damaged": int((qlabels == D).sum()),
        "candidate_correct_intact": int((qlabels == I).sum()),
        "memory_delta_rescue_auroc": float(auc),
        "delta_mean_missed_damaged": float(delta[qlabels == D].mean()) if bool((qlabels == D).any()) else float("nan"),
        "delta_mean_correct_intact": float(delta[qlabels == I].mean()) if bool((qlabels == I).any()) else float("nan"),
        "topk_damaged_fraction_mean": float(agreement.mean()),
        "top1_damaged_rate_missed": float((top1_label[qlabels == D] == D).float().mean()) if bool((qlabels == D).any()) else float("nan"),
        "retrieved_event_diversity_mean": float(sum(event_diversity) / max(1, len(event_diversity))),
        **recall,
        "test_accessed": False,
    }
    write_json(args.output, out)
    torch.save({
        "query_indices": qids,
        "neighbor_indices": final_indices,
        "neighbor_scores": final_scores.half(),
        "delta_di": delta.half(),
        "damaged_fraction": agreement.half(),
        "labels": qlabels,
        "names": [va["names"][i] for i in qids.tolist()],
        "events": [va["events"][i] for i in qids.tolist()],
    }, str(Path(args.output).with_suffix(".pt")))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
