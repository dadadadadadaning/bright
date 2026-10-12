#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .late_core import (
    D, I, binary_auroc, candidate_mask, fixed_budget_rows,
    seed_everything, supervised_train_mask, write_json,
)


class Specialist(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def feature_parts(cache, variant: str) -> torch.Tensor:
    f = cache["features"].float()
    token_dim = f.shape[1] // 4
    probs = cache["base_probs"].float()
    unc = cache["uncertainty"].float().unsqueeze(1)
    peak = cache["peak_mean_delta"].float().unsqueeze(1)
    if variant == "C0":
        return torch.cat((probs, unc), 1)
    if variant == "C1":
        return torch.cat((probs, unc, f[:, :token_dim]), 1)
    if variant == "C2":
        return torch.cat((probs, unc, f[:, : 2 * token_dim]), 1)
    if variant == "C3":
        return torch.cat((probs, unc, f, peak), 1)
    raise ValueError(variant)


def sample_balanced(labels, mask, batch, rng):
    ii = torch.nonzero(mask & (labels == I), as_tuple=False).flatten().tolist()
    dd = torch.nonzero(mask & (labels == D), as_tuple=False).flatten().tolist()
    if not ii or not dd:
        raise RuntimeError("train candidates need both I and D")
    ids = [rng.choice(dd if j % 2 else ii) for j in range(batch)]
    rng.shuffle(ids)
    return torch.tensor(ids, dtype=torch.long)


def main():
    p = argparse.ArgumentParser("E1/E4: train late-decision Damaged trigger")
    p.add_argument("--train_cache", required=True)
    p.add_argument("--val_cache", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_variant", choices=("C0", "C1", "C2", "C3"), default="C3")
    p.add_argument("--candidate", choices=("base_intact", "top2_id", "union"), default="base_intact")
    p.add_argument("--min_train_purity", type=float, default=0.70)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"completed output exists: {out}")
    tr = torch.load(args.train_cache, map_location="cpu", weights_only=False, mmap=True)
    va = torch.load(args.val_cache, map_location="cpu", weights_only=False, mmap=True)
    xtr, xva = feature_parts(tr, args.feature_variant), feature_parts(va, args.feature_variant)
    ytr, yva = tr["labels"].long(), va["labels"].long()
    mtr = supervised_train_mask(tr, args.candidate, args.min_train_purity)
    mva_obs = candidate_mask(va, args.candidate)  # IMPORTANT: no GT filter.
    mva_diag = mva_obs & ((yva == I) | (yva == D)) & (va["label_purity"].float() >= 0.70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Specialist(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps))
    seed_everything(args.seed); rng = random.Random(args.seed)
    best_auc, best_payload = -float("inf"), None

    @torch.no_grad()
    def score_all():
        model.eval(); score = torch.empty(len(xva))
        for a in range(0, len(xva), 4096):
            score[a:a+4096] = model(xva[a:a+4096].to(device)).cpu()
        auc = binary_auroc(score[mva_diag], (yva[mva_diag] == D).long()) if bool(mva_diag.any()) else float("nan")
        rows = fixed_budget_rows(score, va, candidate=mva_obs, override_mode="intact_only")
        return score, auc, rows

    for step in range(1, args.steps + 1):
        ids = sample_balanced(ytr, mtr, min(args.batch_size, int(mtr.sum())), rng)
        xb = xtr[ids].to(device); target = (ytr[ids] == D).float().to(device)
        model.train(); s = model(xb); loss = F.binary_cross_entropy_with_logits(s, target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "lr": sched.get_last_lr()[0]}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            score, auc, rows = score_all()
            print(json.dumps({"VAL": step, "diag_id_auc": auc, "pixel_rows": rows}, ensure_ascii=False), flush=True)
            payload = {"model": model.state_dict(), "input_dim": int(xtr.shape[1]), "step": step,
                       "feature_variant": args.feature_variant, "candidate": args.candidate}
            torch.save(payload, out / "latest.pth")
            if auc > best_auc:
                best_auc, best_payload = auc, payload
                torch.save(payload, out / "best.pth")

    if best_payload is None:
        raise RuntimeError("no validation evaluation")
    model.load_state_dict(best_payload["model"])
    score, auc, rows = score_all()
    summary = {
        "protocol": "LATE_DECISION_E1_SPECIALIST",
        "feature_variant": args.feature_variant,
        "feature_definition": {
            "C0": "A1 probs + uncertainty",
            "C1": "C0 + RegionToken mean",
            "C2": "C0 + RegionToken mean/std",
            "C3": "C0 + mean/std + top1/top2 critical Region + Peak-Mean delta",
        }[args.feature_variant],
        "candidate_rule": args.candidate,
        "val_candidate_filter_uses_gt": False,
        "train_gt_filter": f"I/D and purity>={args.min_train_purity}",
        "best_diag_id_auroc": float(auc),
        "pixel_late_override_rows": rows,
        "test_accessed": False,
    }
    write_json(out / "summary.json", summary)
    torch.save({
        "score": score.half(), "candidate_mask": mva_obs,
        "labels": yva, "purity": va["label_purity"],
        "names": va["names"], "events": va["events"],
    }, out / "val_scores.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
