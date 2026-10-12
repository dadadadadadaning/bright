#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Tuple

import torch

from bda_benchmark.experiments.damageformer_causal_research_chain_v1.cache_features import region_slice, wrong_region_feature
from bda_benchmark.experiments.damageformer_causal_research_chain_v1.memory_oracle import cosine_topk_chunked, local_set_score, retrieval_delta

from .late_core import D, I, candidate_mask, fixed_budget_rows, write_json, zscore_on_mask


def normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def main():
    p = argparse.ArgumentParser("E3: Region-aware memory verification for late decision")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--specialist_scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--control", choices=("none", "shuffle_labels", "shuffle_events", "wrong_regions"), default="none")
    p.add_argument("--coarse_k", type=int, default=50)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--local_top_regions", type=int, default=4)
    p.add_argument("--local_weight", type=float, default=0.35)
    p.add_argument("--min_train_purity", type=float, default=0.70)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    tr = torch.load(args.train_cache, map_location="cpu", weights_only=False, mmap=True)
    va = torch.load(args.val_cache, map_location="cpu", weights_only=False, mmap=True)
    sp = torch.load(args.specialist_scores, map_location="cpu", weights_only=False)
    candidate = candidate_mask(va, "base_intact")
    qids = torch.nonzero(candidate, as_tuple=False).flatten()

    ytr = tr["labels"].long(); purity = tr["label_purity"].float()
    support_mask = (purity >= args.min_train_purity) & ((ytr == I) | (ytr == D))
    support_ids = torch.nonzero(support_mask, as_tuple=False).flatten()
    memory_labels = ytr[support_ids].clone()
    memory_events = [str(tr["events"][i]) for i in support_ids.tolist()]

    g = torch.Generator().manual_seed(args.seed)
    if args.control == "shuffle_labels":
        memory_labels = memory_labels[torch.randperm(len(memory_labels), generator=g)]
    if args.control == "shuffle_events":
        perm = torch.randperm(len(memory_events), generator=g).tolist()
        memory_events = [memory_events[i] for i in perm]

    if args.control == "wrong_regions":
        xtr_all = torch.stack([wrong_region_feature(*region_slice(tr, i)) for i in range(len(ytr))])
        xva_all = torch.stack([wrong_region_feature(*region_slice(va, i)) for i in range(len(va["labels"]))])
    else:
        xtr_all = tr["features"].float()
        xva_all = va["features"].float()
    xmem = xtr_all[support_ids]
    xq = xva_all[qids]

    coarse_scores, coarse_idx = cosine_topk_chunked(
        xq, xmem, args.coarse_k,
        query_events=[str(va["events"][i]) for i in qids.tolist()],
        memory_events=memory_events,
        exclude_same_event=True,
        device=args.device,
    )

    final_scores, final_indices = [], []
    for qi, vid in enumerate(qids.tolist()):
        cidx = coarse_idx[qi]
        gscore = coarse_scores[qi]
        if args.control == "wrong_regions":
            comb = gscore
        else:
            qtok, qcrit = region_slice(va, vid)
            local = []
            for sid in cidx.tolist():
                mid = int(support_ids[sid])
                mtok, mcrit = region_slice(tr, mid)
                local.append(local_set_score(qtok, mtok, qcrit, mcrit, args.local_top_regions))
            local_t = torch.tensor(local, dtype=torch.float32)
            comb = (1.0 - args.local_weight) * gscore + args.local_weight * local_t
        order = torch.argsort(comb, descending=True)[: min(args.topk, len(comb))]
        final_scores.append(comb[order])
        final_indices.append(cidx[order])
    final_scores = torch.stack(final_scores)
    final_indices = torch.stack(final_indices)
    delta, damaged_fraction, top1_label = retrieval_delta(final_scores, final_indices, memory_labels)

    memory_delta_all = torch.zeros(len(va["labels"]), dtype=torch.float32)
    memory_delta_all[qids] = delta
    spec = sp["score"].float()
    combined = zscore_on_mask(spec, candidate) + zscore_on_mask(memory_delta_all, candidate)

    out = {
        "protocol": "LATE_DECISION_E3_MEMORY_VERIFY",
        "control": args.control,
        "same_event_excluded": True,
        "coarse_k": args.coarse_k,
        "topk": args.topk,
        "local_weight": args.local_weight,
        "fusion": "z(Specialist)+z(MemoryDelta), lambda=1 fixed",
        "specialist_only": fixed_budget_rows(spec, va, candidate=candidate),
        "memory_only": fixed_budget_rows(memory_delta_all, va, candidate=candidate),
        "specialist_plus_memory": fixed_budget_rows(combined, va, candidate=candidate),
        "mean_topk_damaged_fraction": float(damaged_fraction.mean()),
        "test_accessed": False,
    }
    write_json(args.output, out)
    torch.save({
        "query_indices": qids, "neighbor_support_indices": final_indices,
        "neighbor_scores": final_scores.half(), "memory_delta": memory_delta_all.half(),
        "combined_score": combined.half(), "candidate_mask": candidate,
    }, str(Path(args.output).with_suffix(".pt")))
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
