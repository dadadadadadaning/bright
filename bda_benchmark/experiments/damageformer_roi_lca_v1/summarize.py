#!/usr/bin/env python3
"""Summarise clean gains and misalignment drops against D0."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


MODES = ("none", "shift_x4", "shift_y4", "shift_x8", "shift_y8", "shift_x16", "shift_y16", "pair_shuffle", "zero")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def event_bootstrap(deltas, repeats=2000, seed=3407):
    if not deltas:
        return None
    generator = random.Random(seed)
    samples = []
    for _ in range(repeats):
        draw = [deltas[generator.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return {
        "mean_pp": 100.0 * sum(deltas) / len(deltas),
        "ci95_low_pp": 100.0 * samples[int(0.025 * repeats)],
        "ci95_high_pp": 100.0 * samples[min(repeats - 1, int(0.975 * repeats))],
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_root", required=True)
    parser.add_argument("--tag", default="probe5000")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    root = Path(args.audit_root)
    variants = ("d0", "a2_lca3", "a3_conf3", "a4_orient3", "a5_conf5")
    rows = {}
    for variant in variants:
        directory = root / (f"d0_seed{args.seed}" if variant == "d0" else f"{variant}_{args.tag}_seed{args.seed}")
        if not (directory / "none.json").is_file():
            continue
        clean = load(directory / "none.json")
        clean_miou = clean["metrics"]["valid_miou"]
        row = {"clean_valid_miou": clean_miou, "clean_damaged_f1": clean["metrics"].get("damaged_f1"),
               "event_miou": clean["metrics"].get("event_miou_by_name", {}), "drops_pp": {}}
        for mode in MODES[1:]:
            path = directory / f"{mode}.json"
            if path.is_file():
                value = load(path)
                row["drops_pp"][mode] = 100.0 * (clean_miou - value["metrics"]["valid_miou"])
        rows[variant] = row
    if "d0" in rows:
        base = rows["d0"]["clean_valid_miou"]
        base_f1 = rows["d0"]["clean_damaged_f1"]
        base_events = rows["d0"]["event_miou"]
        for variant, row in rows.items():
            row["gain_vs_d0_pp"] = 100.0 * (row["clean_valid_miou"] - base)
            row["damaged_f1_gain_vs_d0_pp"] = (
                100.0 * (row["clean_damaged_f1"] - base_f1)
                if row["clean_damaged_f1"] is not None and base_f1 is not None else None
            )
            common = sorted(set(base_events) & set(row["event_miou"]))
            deltas = [row["event_miou"][name] - base_events[name] for name in common]
            row["event_nonnegative"] = sum(value >= 0.0 for value in deltas)
            row["event_total"] = len(deltas)
            row["event_bootstrap"] = event_bootstrap(deltas, seed=args.seed)
            critical = ("shift_x4", "shift_y4", "shift_x8", "shift_y8")
            comparable = [mode for mode in critical if mode in row["drops_pp"] and mode in rows["d0"]["drops_pp"]]
            row["mean_shift_drop_improvement_pp"] = (
                sum(rows["d0"]["drops_pp"][mode] - row["drops_pp"][mode] for mode in comparable) / len(comparable)
                if comparable else None
            )
            if variant != "d0":
                row["quick_admission"] = {
                    "clean_gain_ge_0pp": row["gain_vs_d0_pp"] >= 0.0,
                    "damaged_f1_nonnegative": row["damaged_f1_gain_vs_d0_pp"] is not None and row["damaged_f1_gain_vs_d0_pp"] >= 0.0,
                    "events_nonnegative_ge_10": row["event_nonnegative"] >= 10,
                    "shift_drop_better": row["mean_shift_drop_improvement_pp"] is not None and row["mean_shift_drop_improvement_pp"] > 0.0,
                }
                row["quick_pass"] = all(row["quick_admission"].values())
    result = {"protocol": "DAMAGEFORMER_ROI_LCA_SUMMARY_V1", "rows": rows, "test_accessed": False}
    destination = Path(args.output_json); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
