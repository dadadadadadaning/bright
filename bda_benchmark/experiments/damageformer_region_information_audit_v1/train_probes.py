#!/usr/bin/env python3
"""Parameter-matched R0--R4 linear probes on frozen region records."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import add_hist, confusion_metrics


VARIANTS = ("r0_a1", "r1_real_region", "r2_shuffled_region", "r3_region_only", "r4_grid_region")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def compose(record: Dict, variant: str, seed: int) -> torch.Tensor:
    a1 = record["a1"].float()
    region = record["region"].float()
    zeros_a1, zeros_region = torch.zeros_like(a1), torch.zeros_like(region)
    if variant == "r0_a1":
        return torch.cat((a1, zeros_region), 1)
    if variant == "r1_real_region":
        return torch.cat((a1, region), 1)
    if variant == "r2_shuffled_region":
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(region), generator=generator)
        # A derangement is not required statistically, but prevent the trivial
        # identity case for tiny smoke caches.
        if len(region) > 1 and bool((permutation == torch.arange(len(region))).all()):
            permutation = permutation.roll(1)
        return torch.cat((a1, region[permutation]), 1)
    if variant == "r3_region_only":
        return torch.cat((zeros_a1, region), 1)
    if variant == "r4_grid_region":
        return torch.cat((a1, record["grid"].float()), 1)
    raise ValueError(variant)


class LinearProbe(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(dimension, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def subgroup_metrics(target, prediction, values: List[str]) -> Dict[str, object]:
    output = {}
    for value in sorted(set(values)):
        index = torch.tensor([x == value for x in values], dtype=torch.bool)
        hist = torch.zeros(3, 3, dtype=torch.float64)
        add_hist(hist, target[index], prediction[index])
        output[value] = {**confusion_metrics(hist), "count": int(index.sum())}
    return output


@torch.no_grad()
def evaluate(model, x, record, device) -> Dict[str, object]:
    prediction_parts = []
    for start in range(0, len(x), 8192):
        prediction_parts.append(model(x[start:start + 8192].to(device)).argmax(1).cpu())
    prediction = torch.cat(prediction_parts) if prediction_parts else torch.empty(0, dtype=torch.long)
    target = record["target"].long()
    hist = torch.zeros(3, 3, dtype=torch.float64)
    add_hist(hist, target, prediction)
    base = record["a1_prediction"].long()
    base_wrong = base != target
    base_correct = ~base_wrong
    result = confusion_metrics(hist)
    result.update({
        "count": len(target),
        "a1_region_accuracy": float((base == target).float().mean()) if len(target) else 0.0,
        "a1_error_count": int(base_wrong.sum()),
        "a1_error_recovery_count": int((base_wrong & (prediction == target)).sum()),
        "a1_error_recovery_rate": float((prediction[base_wrong] == target[base_wrong]).float().mean()) if bool(base_wrong.any()) else 0.0,
        "a1_correct_harm_count": int((base_correct & (prediction != target)).sum()),
        "a1_correct_harm_rate": float((prediction[base_correct] != target[base_correct]).float().mean()) if bool(base_correct.any()) else 0.0,
        "net_correction_count": int((base_wrong & (prediction == target)).sum() - (base_correct & (prediction != target)).sum()),
        "by_event": subgroup_metrics(target, prediction, record["event"]),
        "by_building_size": subgroup_metrics(target, prediction, record["size_bin"]),
        "by_damage_ratio": subgroup_metrics(target, prediction, record["damage_ratio_bin"]),
    })
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_records", required=True)
    p.add_argument("--val_records", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = p.parse_args()
    seed_all(args.seed)
    train = torch.load(args.train_records, map_location="cpu", weights_only=False)
    val = torch.load(args.val_records, map_location="cpu", weights_only=False)
    if train.get("test_accessed") or val.get("test_accessed"):
        raise RuntimeError("test-derived records are forbidden")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for variant in args.variants:
        seed_all(args.seed)
        x_train = compose(train, variant, args.seed)
        x_val = compose(val, variant, args.seed + 1)
        y_train = train["target"].long()
        mean = x_train.mean(0)
        std = x_train.std(0, unbiased=False).clamp_min(1e-5)
        # Constant zero slots stay zero after normalization.
        mean[std <= 1.00001e-5] = 0.0
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        model = LinearProbe(x_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
        counts = torch.bincount(y_train, minlength=3).float().clamp_min(1)
        class_weight = (counts.sum() / (3 * counts)).to(device)
        generator = torch.Generator().manual_seed(args.seed)
        for step in range(1, args.steps + 1):
            index = torch.randint(0, len(x_train), (min(args.batch_size, len(x_train)),), generator=generator)
            logits = model(x_train[index].to(device))
            loss = F.cross_entropy(logits, y_train[index].to(device), weight=class_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
            if step % 500 == 0 or step == args.steps:
                print(f"{variant} step={step}/{args.steps} loss={float(loss):.6f}", flush=True)
        result = evaluate(model, x_val, val, device)
        result.update({
            "variant": variant,
            "input_dimension": int(x_train.shape[1]),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "train_class_count": counts.long().tolist(),
            "test_accessed": False,
        })
        all_results[variant] = result
        torch.save({"model": model.state_dict(), "mean": mean, "std": std, "variant": variant,
                    "input_dimension": x_train.shape[1], "test_accessed": False}, output_dir / f"{variant}.pth")
        (output_dir / f"{variant}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{variant}: acc={100*result['accuracy']:.3f}% macroF1={100*result['macro_f1']:.3f}% "
              f"recovery={100*result['a1_error_recovery_rate']:.3f}% harm={100*result['a1_correct_harm_rate']:.3f}%")

    def pp(left: str, right: str) -> float:
        return 100 * (all_results[left]["macro_f1"] - all_results[right]["macro_f1"])
    summary = {
        "protocol": "DAMAGEFORMER_REGION_INFORMATION_AUDIT_V1_PROBES",
        "results": all_results,
        "comparisons_pp": {
            "r1_minus_r0": pp("r1_real_region", "r0_a1") if {"r1_real_region", "r0_a1"} <= all_results.keys() else None,
            "r1_minus_r2": pp("r1_real_region", "r2_shuffled_region") if {"r1_real_region", "r2_shuffled_region"} <= all_results.keys() else None,
            "r1_minus_r4": pp("r1_real_region", "r4_grid_region") if {"r1_real_region", "r4_grid_region"} <= all_results.keys() else None,
        },
        "same_capacity_contract": {
            "layout": "18 A1 slots + 128 region slots",
            "unused_slots": "zero-filled",
            "classifier": "Linear(146,3) for every R0-R4 variant",
        },
        "admission": {
            "minimum_increment_pp": 0.20,
            "r1_above_r0": None,
            "r1_above_shuffled": None,
            "r1_above_grid": None,
        },
        "test_accessed": False,
    }
    c = summary["comparisons_pp"]
    summary["admission"]["r1_above_r0"] = c["r1_minus_r0"] is not None and c["r1_minus_r0"] >= 0.20
    summary["admission"]["r1_above_shuffled"] = c["r1_minus_r2"] is not None and c["r1_minus_r2"] >= 0.20
    summary["admission"]["r1_above_grid"] = c["r1_minus_r4"] is not None and c["r1_minus_r4"] >= 0.20
    summary["pass"] = all(summary["admission"][k] for k in ("r1_above_r0", "r1_above_shuffled", "r1_above_grid"))
    (output_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons_pp": c, "admission": summary["admission"], "pass": summary["pass"]}, indent=2))


if __name__ == "__main__":
    main()

