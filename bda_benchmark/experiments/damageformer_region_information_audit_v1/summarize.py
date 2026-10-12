#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", required=True)
    args = p.parse_args()
    root = Path(args.eval_root)
    partition = json.loads((root / "partition_validation.json").read_text(encoding="utf-8"))
    probes = json.loads((root / "probes/probe_summary.json").read_text(encoding="utf-8"))
    oracle = partition["oracle_region_upper_bound"]
    print("===== Gate 1: partition =====")
    print(f"Boundary recall             : {100*partition['boundary_recall']:.4f}%")
    print(f"Internal class boundary     : {100*partition['internal_class_boundary_recall']:.4f}%")
    print(f"Region purity               : {100*oracle['pixel_weighted_region_purity']:.4f}%")
    print(f"Under-segmentation error    : {100*oracle['undersegmentation_error']:.4f}%")
    print(f"Oracle accuracy             : {100*oracle['accuracy']:.4f}%")
    print(f"Oracle macro-F1             : {100*oracle['macro_f1']:.4f}%")
    print(f"Small affected preservation: {100*partition['small_affected_component_preservation']:.4f}%")
    print("\n===== Gate 2: frozen information probes =====")
    for key in ("r0_a1", "r1_real_region", "r2_shuffled_region", "r3_region_only", "r4_grid_region"):
        row = probes["results"][key]
        print(f"{key:20s} Acc={100*row['accuracy']:.4f}% Macro-F1={100*row['macro_f1']:.4f}% "
              f"D-F1={100*row['f1'][1]:.4f}% X-F1={100*row['f1'][2]:.4f}% "
              f"Recovery={100*row['a1_error_recovery_rate']:.4f}% Harm={100*row['a1_correct_harm_rate']:.4f}%")
    print("\nComparisons (Macro-F1):")
    for key, value in probes["comparisons_pp"].items():
        print(f"{key:20s}: {value:+.4f} pp")
    print("PASS:", probes["pass"])


if __name__ == "__main__":
    main()

