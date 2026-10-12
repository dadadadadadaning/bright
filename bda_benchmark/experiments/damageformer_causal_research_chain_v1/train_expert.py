#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import D, I, binary_auroc, cache_fingerprint, coverage_curve, load_cache, seed_everything, top2_is_id, write_json
from .expert_model import DamagedSpecialist
from .representation import transform_features


def make_inputs(cache, tail_ckpt: str, device: str):
    feat_raw = cache["features"].float()
    feat, tail_scores = transform_features(feat_raw, tail_ckpt, device)
    probs = cache["base_probs"].float()
    uncertainty = cache["uncertainty"].float().unsqueeze(1)
    peak = cache["peak_mean_delta"].float().unsqueeze(1)
    parts = [feat, probs, uncertainty, peak]
    if tail_scores is not None:
        parts.append(tail_scores.float())
        parts.append((tail_scores[:, 1] - tail_scores[:, 0]).unsqueeze(1).float())
    return torch.cat(parts, dim=1)


def candidate_mask(cache, rule: str, uncertainty_min: float):
    labels = cache["labels"].long()
    probs = cache["base_probs"].float()
    pred = probs.argmax(1)
    a = pred == I
    b = top2_is_id(probs)
    c = a if rule == "base_intact" else b if rule == "top2_id" else (a | b)
    return c & ((labels == I) | (labels == D)) & (cache["uncertainty"].float() >= uncertainty_min)


def balanced_ids(labels: torch.Tensor, mask: torch.Tensor, batch: int, rng: random.Random) -> torch.Tensor:
    ii = torch.nonzero(mask & (labels == I), as_tuple=False).flatten().tolist()
    dd = torch.nonzero(mask & (labels == D), as_tuple=False).flatten().tolist()
    if not ii or not dd:
        raise RuntimeError("expert train candidates must contain both Intact and Damaged")
    out = []
    for j in range(batch):
        pool = dd if j % 2 else ii
        out.append(rng.choice(pool))
    rng.shuffle(out)
    return torch.tensor(out, dtype=torch.long)


def main() -> None:
    p = argparse.ArgumentParser("Train conditional Damaged specialist on frozen evidence")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--candidate", choices=("base_intact", "top2_id", "union"), default="union")
    p.add_argument("--uncertainty_min", type=float, default=0.0)
    p.add_argument("--tail_ckpt", default="", help="Optional Stage-B multi-prototype checkpoint; specialist consumes projected evidence and prototype margin.")
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
    xtr, xva = make_inputs(tr, args.tail_ckpt, args.device), make_inputs(va, args.tail_ckpt, args.device)
    ytr, yva = tr["labels"].long(), va["labels"].long()
    mtr = candidate_mask(tr, args.candidate, args.uncertainty_min)
    mva = candidate_mask(va, args.candidate, args.uncertainty_min)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamagedSpecialist(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps))
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    best, best_payload = -float("inf"), None

    def evaluate():
        model.eval()
        with torch.no_grad():
            score = torch.empty(len(xva))
            for a in range(0, len(xva), 4096):
                score[a:a+4096] = model(xva[a:a+4096].to(device)).cpu()
        auc = binary_auroc(score[mva], (yva[mva] == D).long()) if bool(mva.any()) else float("nan")
        curve = coverage_curve(score, mva, yva == D, yva == I)
        return float(auc), curve, score

    for step in range(1, args.steps + 1):
        ids = balanced_ids(ytr, mtr, min(args.batch_size, int(mtr.sum())), rng)
        xb = xtr[ids].to(device)
        target = (ytr[ids] == D).float().to(device)
        model.train()
        score = model(xb)
        loss = F.binary_cross_entropy_with_logits(score, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss), "lr": sched.get_last_lr()[0]}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            auc, curve, _ = evaluate()
            print("VAL", step, "auc", auc, "curve", json.dumps(curve), flush=True)
            payload = {"model": model.state_dict(), "input_dim": int(xtr.shape[1]), "step": step, "candidate": args.candidate}
            torch.save(payload, out / "latest.pth")
            if auc > best:
                best = auc; best_payload = payload; torch.save(payload, out / "best.pth")
    if best_payload is None:
        raise RuntimeError("no expert eval")
    model.load_state_dict(best_payload["model"])
    auc, curve, score = evaluate()
    summary = {
        "protocol": "CAUSAL_CHAIN_STAGE_C_DAMAGED_SPECIALIST",
        "train_cache": cache_fingerprint(args.train_cache),
        "val_cache": cache_fingerprint(args.val_cache),
        "candidate_rule": args.candidate,
        "tail_checkpoint": args.tail_ckpt or None,
        "train_candidates": int(mtr.sum()),
        "val_candidates": int(mva.sum()),
        "val_candidate_damaged_prevalence": float((yva[mva] == D).float().mean()) if bool(mva.any()) else 0.0,
        "val_di_auroc": auc,
        "coverage_curve": curve,
        "selection_metric": "candidate DI AUROC",
        "test_accessed": False,
    }
    write_json(out / "summary.json", summary)
    torch.save({"score": score.half(), "candidate_mask": mva, "labels": yva, "names": va["names"], "events": va["events"]}, out / "val_scores.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
