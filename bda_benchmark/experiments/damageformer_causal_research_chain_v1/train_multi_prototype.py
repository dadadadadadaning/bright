#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .core import D, I, cache_fingerprint, classification_metrics, event_entropy, load_cache, seed_everything, separation_ratio, write_json
from .tail_models import MultiPrototypeTail


def build_groups(labels: torch.Tensor, events: List[str]) -> Dict[Tuple[int, str], List[int]]:
    g: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for i, (y, e) in enumerate(zip(labels.tolist(), events)):
        g[(int(y), str(e))].append(i)
    return g


def class_event_batch(groups: Dict[Tuple[int, str], List[int]], batch_size: int, rng: random.Random) -> torch.Tensor:
    by_class: Dict[int, List[str]] = defaultdict(list)
    for (c, e), ids in groups.items():
        if ids:
            by_class[c].append(e)
    ids: List[int] = []
    for _ in range(batch_size):
        c = rng.choice(sorted(by_class))
        e = rng.choice(by_class[c])
        ids.append(rng.choice(groups[(c, e)]))
    return torch.tensor(ids, dtype=torch.long)


def diversity_loss(proto: torch.Tensor, threshold: float = 0.75) -> torch.Tensor:
    # proto C,K,D normalized
    c, k, _ = proto.shape
    if k <= 1:
        return proto.sum() * 0.0
    vals = []
    for cls in range(c):
        s = proto[cls] @ proto[cls].T
        mask = ~torch.eye(k, dtype=torch.bool, device=proto.device)
        vals.append(F.relu(s[mask] - threshold).mean())
    return torch.stack(vals).mean()


def main() -> None:
    p = argparse.ArgumentParser("Event-balanced multi-prototype tail separation")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prototypes", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--margin_weight", type=float, default=0.50)
    p.add_argument("--diversity_weight", type=float, default=0.05)
    p.add_argument("--diversity_threshold", type=float, default=0.75)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"completed output exists: {out}")
    tr = load_cache(args.train_cache, expected_split="train")
    va = load_cache(args.val_cache, expected_split="val")
    xtr, ytr = tr["features"].float(), tr["labels"].long()
    xva, yva = va["features"].float(), va["labels"].long()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MultiPrototypeTail(xtr.shape[1], args.embed_dim, args.prototypes, args.temperature).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps))
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    groups = build_groups(ytr, tr["events"])
    best, best_payload = -float("inf"), None

    def evaluate():
        model.eval()
        scores, embeds, assigns = [], [], []
        with torch.no_grad():
            prot = model.normalized_prototypes()
            for i in range(0, len(xva), 4096):
                xb = xva[i:i+4096].to(device)
                s, h = model(xb)
                scores.append(s.cpu())
                embeds.append(h.cpu())
                sim = torch.einsum("nd,ckd->nck", h, prot)
                # chosen prototype within predicted/GT-independent best class-prototype pair
                flat = sim.reshape(sim.shape[0], -1).argmax(1)
                assigns.append(flat.cpu())
        scores = torch.cat(scores)
        embeds = torch.cat(embeds)
        assigns = torch.cat(assigns)
        m = classification_metrics(scores, yva, va["events"])
        m["projected_di_separation_ratio"] = separation_ratio(embeds, yva)
        # audit prototype event shortcut by actual assigned flat prototype id
        m["prototype_event_entropy"] = event_entropy(assigns, va["events"], 3 * args.prototypes)
        return m, scores, embeds, assigns

    for step in range(1, args.steps + 1):
        model.train()
        ids = class_event_batch(groups, min(args.batch_size, len(xtr)), rng)
        xb, yb = xtr[ids].to(device), ytr[ids].to(device)
        scores, h = model(xb)
        ce = F.cross_entropy(scores, yb)
        di_mask = yb == D
        if bool(di_mask.any()):
            margin = F.relu(args.margin - scores[di_mask, D] + scores[di_mask, I]).mean()
        else:
            margin = ce * 0.0
        div = diversity_loss(model.normalized_prototypes(), args.diversity_threshold)
        loss = ce + args.margin_weight * margin + args.diversity_weight * div
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss), "ce": float(ce), "di_margin": float(margin), "div": float(div)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m, _, _, _ = evaluate()
            score = float(m["di_auroc"])
            print("VAL", step, json.dumps({k: m[k] for k in ("macro_f1", "damaged_f1", "damaged_to_intact", "di_auroc", "projected_di_separation_ratio")}), flush=True)
            payload = {
                "model": model.state_dict(), "input_dim": int(xtr.shape[1]), "embed_dim": args.embed_dim,
                "prototypes": args.prototypes, "temperature": args.temperature, "step": step,
            }
            torch.save(payload, out / "latest.pth")
            if score > best:
                best = score
                best_payload = payload
                torch.save(payload, out / "best.pth")
    if best_payload is None:
        raise RuntimeError("no evaluation")
    model.load_state_dict(best_payload["model"])
    metrics, scores, embeds, assigns = evaluate()
    summary = {
        "protocol": "CAUSAL_CHAIN_STAGE_B_MULTI_PROTOTYPE",
        "prototypes_per_class": args.prototypes,
        "steps": args.steps,
        "seed": args.seed,
        "train_cache": cache_fingerprint(args.train_cache),
        "val_cache": cache_fingerprint(args.val_cache),
        "metrics": metrics,
        "selection_metric": "val DI AUROC",
        "test_accessed": False,
    }
    write_json(out / "summary.json", summary)
    torch.save({"scores": scores.half(), "embeds": embeds.half(), "assignments": assigns, "labels": yva, "names": va["names"], "events": va["events"]}, out / "val_predictions.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
