#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .core import cache_fingerprint, classification_metrics, load_cache, seed_everything, separation_ratio, write_json
from .tail_losses import balanced_ce, blv_loss, class_counts, logit_adjust_ce, seesaw_ce
from .tail_models import LinearProbe, MLPProbe


def sample_batch(n: int, batch_size: int, g: torch.Generator) -> torch.Tensor:
    return torch.randint(0, n, (batch_size,), generator=g)


def main() -> None:
    p = argparse.ArgumentParser("Frozen-representation tail diagnostic")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", choices=("ce", "balanced_ce", "blv", "seesaw", "logit_adjust"), required=True)
    p.add_argument("--head", choices=("linear", "mlp"), default="linear")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--blv_sigma", type=float, default=4.0)
    p.add_argument("--logit_tau", type=float, default=1.0)
    p.add_argument("--seesaw_p", type=float, default=0.8)
    p.add_argument("--seesaw_q", type=float, default=2.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"completed output exists: {out}")
    train = load_cache(args.train_cache, expected_split="train")
    val = load_cache(args.val_cache, expected_split="val")
    xtr = train["features"].float()
    ytr = train["labels"].long()
    xva = val["features"].float()
    yva = val["labels"].long()
    counts = class_counts(ytr)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dim = int(xtr.shape[1])
    model = LinearProbe(dim) if args.head == "linear" else MLPProbe(dim)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps))
    g = seed_everything(args.seed)
    best = -float("inf")
    best_payload = None

    def evaluate():
        model.eval()
        rows = []
        with torch.no_grad():
            for i in range(0, len(xva), 4096):
                rows.append(model(xva[i:i+4096].to(device)).cpu())
        logits = torch.cat(rows)
        metrics = classification_metrics(logits, yva, val["events"])
        metrics["raw_feature_di_separation_ratio"] = separation_ratio(xva, yva)
        return metrics, logits

    for step in range(1, args.steps + 1):
        model.train()
        ids = sample_batch(len(xtr), min(args.batch_size, len(xtr)), g)
        xb, yb = xtr[ids].to(device), ytr[ids].to(device)
        logits = model(xb)
        if args.method == "ce":
            loss = torch.nn.functional.cross_entropy(logits, yb)
        elif args.method == "balanced_ce":
            loss = balanced_ce(logits, yb, counts)
        elif args.method == "blv":
            loss = blv_loss(logits, yb, counts, args.blv_sigma)
        elif args.method == "seesaw":
            loss = seesaw_ce(logits, yb, counts, args.seesaw_p, args.seesaw_q)
        else:
            loss = logit_adjust_ce(logits, yb, counts, args.logit_tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss), "lr": sched.get_last_lr()[0]}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m, _ = evaluate()
            score = float(m["di_auroc"])
            print("VAL", step, json.dumps({k: m[k] for k in ("macro_f1", "damaged_f1", "damaged_to_intact", "di_auroc")}), flush=True)
            payload = {"model": model.state_dict(), "step": step, "method": args.method, "head": args.head, "dim": dim}
            torch.save(payload, out / "latest.pth")
            if score > best:
                best = score
                best_payload = payload
                torch.save(payload, out / "best.pth")
    if best_payload is None:
        raise RuntimeError("no evaluation performed")
    model.load_state_dict(best_payload["model"])
    metrics, logits = evaluate()
    summary = {
        "protocol": "CAUSAL_CHAIN_STAGE_B_FROZEN_TAIL_PROBE",
        "method": args.method,
        "head": args.head,
        "steps": args.steps,
        "seed": args.seed,
        "train_cache": cache_fingerprint(args.train_cache),
        "val_cache": cache_fingerprint(args.val_cache),
        "train_class_counts": counts.tolist(),
        "metrics": metrics,
        "selection_metric": "val DI AUROC",
        "test_accessed": False,
    }
    write_json(out / "summary.json", summary)
    torch.save({"logits": logits.half(), "labels": yva, "names": val["names"], "events": val["events"]}, out / "val_predictions.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
