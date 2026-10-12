#!/usr/bin/env python3
"""Train-only calibration verifier for the region-aware memory branch.

The verifier never trains on Validation. Train cache is split deterministically
into fit-memory and calibration-query subsets. Validation is read only after the
verifier has been trained and frozen.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache_features import region_slice
from .core import D, I, binary_auroc, cache_fingerprint, coverage_curve, deterministic_partition, load_cache, seed_everything, write_json
from .memory_oracle import cosine_topk_chunked, local_set_score
from .representation import transform_features


class Verifier(nn.Module):
    def __init__(self, dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _rerank_features(
    memory_cache: Dict[str, object],
    query_cache: Dict[str, object],
    memory_ids: torch.Tensor,
    query_ids: torch.Tensor,
    memory_feat: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    coarse_k: int,
    topk: int,
    local_weight: float,
    local_top_regions: int,
    device: str,
) -> torch.Tensor:
    mem_events = [memory_cache["events"][i] for i in memory_ids.tolist()]
    qry_events = [query_cache["events"][i] for i in query_ids.tolist()]
    coarse_s, coarse_local_idx = cosine_topk_chunked(
        query_feat, memory_feat, coarse_k,
        query_events=qry_events,
        memory_events=mem_events,
        exclude_same_event=True,
        device=device,
    )
    rows = []
    mem_labels_all = memory_cache["labels"].long()
    for qi, q_global_id in enumerate(query_ids.tolist()):
        local_mem = coarse_local_idx[qi]
        global_mem = memory_ids[local_mem]
        qtok, qcrit = region_slice(query_cache, q_global_id)
        local_scores = []
        for mid in global_mem.tolist():
            mtok, mcrit = region_slice(memory_cache, mid)
            local_scores.append(local_set_score(qtok, mtok, qcrit, mcrit, local_top_regions))
        local_t = torch.tensor(local_scores, dtype=torch.float32)
        combined = (1.0 - local_weight) * coarse_s[qi] + local_weight * local_t
        order = torch.argsort(combined, descending=True)[: min(topk, len(combined))]
        final_scores = combined[order]
        final_ids = global_mem[order]
        lab = mem_labels_all[final_ids]
        d = final_scores[lab == D]
        ii = final_scores[lab == I]
        sd = float(d.mean()) if len(d) else -1.0
        si = float(ii.mean()) if len(ii) else -1.0
        frac_d = float((lab == D).float().mean())
        top1 = float(final_scores[0])
        event_div = float(len(set(memory_cache["events"][j] for j in final_ids.tolist())) / max(1, len(final_ids)))
        score_std = float(final_scores.std(unbiased=False))
        probs = query_cache["base_probs"][q_global_id].float()
        uncertainty = float(1.0 - probs.max())
        base_margin = float(probs[D] - probs[I])
        rows.append([uncertainty, base_margin, sd - si, frac_d, top1, event_div, score_std])
    return torch.tensor(rows, dtype=torch.float32)


def _candidate_ids(cache: Dict[str, object], mask: torch.Tensor | None = None) -> torch.Tensor:
    labels = cache["labels"].long()
    pred = cache["base_probs"].argmax(1)
    c = (pred == I) & ((labels == I) | (labels == D))
    if mask is not None:
        c &= mask
    return torch.nonzero(c, as_tuple=False).flatten()


def main() -> None:
    p = argparse.ArgumentParser("Train/evaluate region-aware memory utility verifier")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tail_ckpt", default="")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--calibration_fraction", type=float, default=0.20)
    p.add_argument("--coarse_k", type=int, default=50)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--local_weight", type=float, default=0.35)
    p.add_argument("--local_top_regions", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"completed output exists: {out}")

    tr = load_cache(args.train_cache, expected_split="train")
    va = load_cache(args.val_cache, expected_split="val")
    fit_mask, cal_mask = deterministic_partition(
        tr["names"], tr["events"], tr["labels"], args.calibration_fraction, args.seed
    )
    fit_ids = torch.nonzero(fit_mask, as_tuple=False).flatten()
    cal_ids = _candidate_ids(tr, cal_mask)
    if len(cal_ids) < 10:
        raise RuntimeError("too few train-only calibration candidates")

    tr_repr, _ = transform_features(tr["features"].float(), args.tail_ckpt, args.device)
    va_repr, _ = transform_features(va["features"].float(), args.tail_ckpt, args.device)

    xcal = _rerank_features(
        tr, tr, fit_ids, cal_ids, tr_repr[fit_ids], tr_repr[cal_ids],
        coarse_k=args.coarse_k, topk=args.topk, local_weight=args.local_weight,
        local_top_regions=args.local_top_regions, device=args.device,
    )
    ycal = (tr["labels"][cal_ids] == D).float()
    if not bool((ycal == 1).any() and (ycal == 0).any()):
        raise RuntimeError("calibration candidates must contain both Intact and Damaged")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Verifier(xcal.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    pos = torch.nonzero(ycal == 1, as_tuple=False).flatten().tolist()
    neg = torch.nonzero(ycal == 0, as_tuple=False).flatten().tolist()

    for step in range(1, args.steps + 1):
        b = min(args.batch_size, 2 * min(len(pos), len(neg)))
        ids = [rng.choice(pos if j % 2 else neg) for j in range(max(2, b))]
        rng.shuffle(ids)
        ids_t = torch.tensor(ids)
        score = model(xcal[ids_t].to(device))
        loss = F.binary_cross_entropy_with_logits(score, ycal[ids_t].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss)}), flush=True)

    # Validation is read-only after verifier parameters are fixed.
    val_ids = _candidate_ids(va)
    xv = _rerank_features(
        tr, va, torch.arange(len(tr["labels"])), val_ids, tr_repr, va_repr[val_ids],
        coarse_k=args.coarse_k, topk=args.topk, local_weight=args.local_weight,
        local_top_regions=args.local_top_regions, device=args.device,
    )
    with torch.no_grad():
        score = model(xv.to(device)).cpu()
    yv = va["labels"][val_ids]
    auc = binary_auroc(score, (yv == D).long())
    curve = coverage_curve(
        score,
        torch.ones(len(score), dtype=torch.bool),
        yv == D,
        yv == I,
    )
    torch.save(
        {"model": model.state_dict(), "input_dim": int(xcal.shape[1]), "topk": args.topk},
        out / "verifier.pth",
    )
    summary = {
        "protocol": "CAUSAL_CHAIN_STAGE_E_REGION_MEMORY_VERIFIER",
        "train_cache": cache_fingerprint(args.train_cache),
        "val_cache": cache_fingerprint(args.val_cache),
        "tail_checkpoint": args.tail_ckpt or None,
        "train_fit_count": int(fit_mask.sum()),
        "train_calibration_candidate_count": int(len(cal_ids)),
        "val_candidate_count": int(len(val_ids)),
        "val_di_auroc": float(auc),
        "coverage_curve": curve,
        "coarse_k": args.coarse_k,
        "topk": args.topk,
        "local_weight": args.local_weight,
        "verifier_training_uses_validation": False,
        "memory_excludes_same_event": True,
        "test_accessed": False,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
