from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import (
    D, Specialist, balanced_sample, candidate_mask, clean_id_candidate,
    evaluate_coverages, feature_matrix, load_stage1_cache, score_diagnostics,
    score_model, seed_all, stable_image_split, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser("Leakage-safe C0-C3 specialist")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--variant", choices=("C0", "C1", "C2", "C3"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--coverage", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_stage1_cache(args.train_cache, "train")
    val = load_stage1_cache(args.val_cache, "val")
    fit, cal = stable_image_split(train, 0.80, args.seed)
    train_rows = clean_id_candidate(train, fit)
    cal_candidate = cal & candidate_mask(train)
    val_candidate = candidate_mask(val)
    x_train = feature_matrix(train, args.variant)
    x_val = feature_matrix(val, args.variant)
    labels = train["labels"].long()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Specialist(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))
    rng = random.Random(args.seed)
    best_key = (-float("inf"), -float("inf"))
    best = None

    for step in range(1, args.steps + 1):
        rows = balanced_sample(labels, train_rows, min(args.batch_size, int(train_rows.sum())), rng)
        xb = x_train[rows].to(device)
        target = (labels[rows] == D).float().to(device)
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(xb), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps:
            train_score = score_model(model, x_train, device)
            result = evaluate_coverages(train, train_score, cal_candidate, "train_cal")[str(args.coverage)]
            diagnostic = score_diagnostics(train, train_score, cal)
            key = (float(result["delta_valid_miou_pp"]), float(diagnostic["ap"] or -1.0))
            print(json.dumps({
                "variant": args.variant, "step": step, "loss": float(loss),
                "cal_delta_miou_pp": key[0], "cal_ap": diagnostic["ap"],
            }), flush=True)
            payload = {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "input_dim": x_train.shape[1], "step": step, "variant": args.variant,
                "fit_mask": fit, "cal_mask": cal, "selection_metric": "Train-Cal delta_valid_miou_pp@2%",
            }
            torch.save(payload, output / "latest.pth")
            if key > best_key:
                best_key, best = key, payload
                torch.save(payload, output / "best.pth")

    if best is None:
        raise RuntimeError("no specialist checkpoint was evaluated")
    model.load_state_dict(best["state_dict"])
    model.to(device)
    train_score = score_model(model, x_train, device)
    val_score = score_model(model, x_val, device)
    summary = {
        "protocol": "DAMAGEFORMER_DECISION_CHAIN_V2_SPECIALIST",
        "variant": args.variant,
        "best_step": best["step"],
        "checkpoint_selected_on": "Train-Cal fixed-coverage pixel utility; never Val",
        "train_fit_diagnostics": score_diagnostics(train, train_score, fit),
        "train_cal_diagnostics": score_diagnostics(train, train_score, cal),
        "train_cal_action": evaluate_coverages(train, train_score, cal_candidate, "train_cal"),
        "val_diagnostics": score_diagnostics(val, val_score, torch.ones(len(val_score), dtype=torch.bool)),
        "val_action": evaluate_coverages(val, val_score, val_candidate, "val"),
        "test_accessed": False,
    }
    write_json(output / "summary.json", summary)
    torch.save({
        "protocol": summary["protocol"], "variant": args.variant,
        "train_scores": train_score.half(), "val_scores": val_score.half(),
        "fit_mask": fit, "cal_mask": cal, "candidate_rule": "base_pred==Intact",
        "test_accessed": False,
    }, output / "scores.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

