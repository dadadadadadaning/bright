#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (
    feature_names, fit_logistic, matrix_from_rows, proposal_filter,
    standardize, threshold_search,
)


def event_folds(events: list[str], count: int) -> list[set[str]]:
    unique = sorted(set(events))
    return [set(unique[index::count]) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser("Train a low-capacity metric-aligned verifier")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--variant", choices=("m2_direct", "m2_union_basic", "m3_class_memory"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--maximum_acceptance", type=float, default=0.70)
    parser.add_argument("--conflict_margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    cache = torch.load(args.targets, map_location="cpu", weights_only=False)
    if cache.get("protocol") != PROTOCOL + "_METRIC_TARGETS":
        raise RuntimeError(f"unexpected target protocol: {cache.get('protocol')!r}")
    if cache.get("split") != "train" or not cache.get("allowed_for_verifier_training", False):
        raise RuntimeError("Verifier fitting requires an explicit Train cache; Val/Test caches are forbidden")
    rows = proposal_filter(cache["rows"], args.variant)
    if not rows:
        raise RuntimeError("no proposal rows after variant filtering")
    names = feature_names(args.variant)
    raw = matrix_from_rows(rows, names)
    labels = torch.tensor([float(row["accept_label"]) for row in rows], dtype=torch.float32)
    utility = torch.tensor([float(row["utility_miou_pp"]) for row in rows], dtype=torch.float32)
    events = [str(row["event"]) for row in rows]

    oof = torch.full((len(rows),), float("nan"))
    folds = event_folds(events, max(2, int(args.folds)))
    for fold_index, held_events in enumerate(folds):
        held = torch.tensor([event in held_events for event in events], dtype=torch.bool)
        train = ~held
        train_x, mean, std = standardize(raw[train])
        model = fit_logistic(
            train_x, labels[train], utility[train], steps=args.steps, lr=args.lr,
            seed=args.seed + fold_index,
        )
        with torch.no_grad():
            oof[held] = torch.sigmoid(model((raw[held] - mean) / std))
        print(f"OOF fold {fold_index + 1}/{len(folds)} held_events={sorted(held_events)} rows={int(held.sum())}")
    if bool(torch.isnan(oof).any()):
        raise RuntimeError("OOF scoring left unassigned rows")

    threshold = threshold_search(
        cache["base_hist"], rows, oof.tolist(), maximum_acceptance=args.maximum_acceptance,
        conflict_margin=args.conflict_margin,
    )
    all_x, mean, std = standardize(raw)
    final_model = fit_logistic(all_x, labels, utility, steps=args.steps, lr=args.lr, seed=args.seed)
    payload = {
        "protocol": PROTOCOL + "_CHECKPOINT", "variant": args.variant,
        "feature_names": list(names), "mean": mean, "std": std,
        "model": final_model.state_dict(), "threshold": threshold["threshold"],
        "conflict_margin": args.conflict_margin, "oof_train_gain_pp": threshold["gain_pp"],
        "oof_acceptance_rate": threshold["acceptance_rate"],
        "training": {
            "rows": len(rows), "positive_rows": int(labels.sum().item()), "steps": args.steps,
            "folds": len(folds), "split_unit": "event", "maximum_acceptance": args.maximum_acceptance,
            "seed": args.seed, "targets": str(Path(args.targets).resolve()),
        },
        "test_accessed": False,
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(
        f"Saved: {destination} variant={args.variant} threshold={threshold['threshold']:.2f} "
        f"OOF_gain={threshold['gain_pp']:+.4f}pp accept={100*threshold['acceptance_rate']:.2f}%"
    )


if __name__ == "__main__": main()
