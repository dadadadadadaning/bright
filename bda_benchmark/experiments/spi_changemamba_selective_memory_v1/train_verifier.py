#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import PROTOCOL
from .verifier_core import (
    VARIANT_FEATURES, balanced_event_folds, filter_rows, fit, matrix,
    search_threshold, standardize,
)


def main() -> None:
    parser = argparse.ArgumentParser("Train safe low-capacity action verifier")
    parser.add_argument("--utility", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_FEATURES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--maximum_acceptance", type=float, default=0.70)
    parser.add_argument("--conflict_margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    cache = torch.load(args.utility, map_location="cpu", weights_only=False)
    if cache.get("protocol") != PROTOCOL + "_UTILITY":
        raise RuntimeError("unexpected utility protocol")
    if cache.get("split") != "train" or not cache.get("allowed_for_verifier_training"):
        raise RuntimeError("Verifier training requires Train utility; Val/Test are forbidden")
    rows = filter_rows(cache["rows"], args.variant)
    names = VARIANT_FEATURES[args.variant]
    raw = matrix(rows, names)
    labels = torch.tensor([float(row["accept_label"]) for row in rows])
    utility = torch.tensor([float(row["utility_miou_pp"]) for row in rows])
    events = [str(row["event"]) for row in rows]
    oof = torch.full((len(rows),), float("nan"))
    folds = balanced_event_folds(events, args.folds)
    for fold_index, held_events in enumerate(folds):
        held = torch.tensor([event in held_events for event in events], dtype=torch.bool)
        train = ~held
        train_x, mean, std = standardize(raw[train])
        model = fit(
            train_x, labels[train], utility[train], steps=args.steps,
            lr=args.lr, seed=args.seed + fold_index,
        )
        with torch.no_grad():
            oof[held] = torch.sigmoid(model((raw[held] - mean) / std))
        print(
            f"OOF {fold_index + 1}/{len(folds)} held_events={sorted(held_events)} "
            f"rows={int(held.sum())}", flush=True,
        )
    if bool(torch.isnan(oof).any()):
        raise RuntimeError("OOF scores contain NaN")
    threshold = search_threshold(
        cache, rows, oof.tolist(), args.maximum_acceptance, args.conflict_margin
    )
    all_x, mean, std = standardize(raw)
    final_model = fit(all_x, labels, utility, steps=args.steps, lr=args.lr, seed=args.seed)
    payload = {
        "protocol": PROTOCOL + "_VERIFIER", "variant": args.variant,
        "feature_names": list(names), "mean": mean, "std": std,
        "model": final_model.state_dict(), "threshold": threshold["threshold"],
        "conflict_margin": args.conflict_margin, "oof": threshold,
        "training": {
            "rows": len(rows), "positive_rows": int(labels.sum()),
            "folds": len(folds), "fold_policy": "candidate-count-balanced event folds",
            "safe_reject_all_threshold": True, "steps": args.steps, "seed": args.seed,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(
        f"Saved: {destination} threshold={threshold['threshold']:.2f} "
        f"gain={threshold['gain_pp']:+.4f}pp accept={100*threshold['acceptance_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()

