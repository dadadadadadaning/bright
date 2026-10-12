#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); args = p.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    base = data["base_identity_observed"]
    print("Frozen Base")
    print(f" valid mIoU : {100*base['valid_miou']:.4f}%")
    print(f" mIoU4      : {100*base['miou4']:.4f}%")
    print(f" Damaged F1 : {100*base['damaged_f1']:.4f}%")
    print("\nFixed 2% strategies")
    for name, row in data["results"].items():
        m = row["metrics"]
        ci = row["bootstrap"]["event_stratified_image_micro"]
        print(f"{name:30s} valid={100*m['valid_miou']:.4f}% "
              f"gain={row['delta_valid_miou_pp']:+.4f}pp D-F1={100*m['damaged_f1']:.4f}% "
              f"CI=[{ci['ci95_low_pp']:+.4f},{ci['ci95_high_pp']:+.4f}] "
              f"events+={row['event_positive_count']}/{row['event_count']}")
    print("\nRandom controls")
    for name, row in data["random_controls"].items():
        print(f"{name:24s} {row['mean_pp']:+.4f} ± {row['std_pp']:.4f} pp")
    print("\nComparisons")
    for name, value in data["comparisons_pp"].items():
        print(f"{name:32s}: {value:+.4f} pp")
    print("\nAdmission")
    for name, value in data["admission"].items():
        if name != "rules": print(f"{name:36s}: {value}")


if __name__ == "__main__": main()
