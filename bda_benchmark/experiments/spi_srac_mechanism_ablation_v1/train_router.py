#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import PROTOCOL
from .router_core import (
    VARIANT_FEATURES, balanced_event_folds, filter_rows, fit, matrix,
    search_threshold, standardize,
)


def main() -> None:
    parser = argparse.ArgumentParser("Train metric-aligned Keep/Direct/RAG router")
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
        raise RuntimeError("router training requires Train utility; Val/Test are forbidden")
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
        model = fit(train_x, labels[train], utility[train], steps=args.steps, lr=args.lr, seed=args.seed + fold_index)
        with torch.no_grad():
            oof[held] = torch.sigmoid(model((raw[held] - mean) / std))
        print(f"OOF {fold_index + 1}/{len(folds)} held={sorted(held_events)} rows={int(held.sum())}", flush=True)
    threshold = search_threshold(cache, rows, oof.tolist(), args.maximum_acceptance, args.conflict_margin)
    all_x, mean, std = standardize(raw)
    model = fit(all_x, labels, utility, steps=args.steps, lr=args.lr, seed=args.seed)
    payload = {
        "protocol": PROTOCOL + "_ROUTER",
        "variant": args.variant,
        "feature_names": list(names),
        "mean": mean,
        "std": std,
        "model": model.state_dict(),
        "threshold": threshold["threshold"],
        "conflict_margin": args.conflict_margin,
        "oof": threshold,
        "safe_reject_all_threshold": True,
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} threshold={threshold['threshold']:.2f} gain={threshold['gain_pp']:+.4f}pp")


if __name__ == "__main__":
    main()

