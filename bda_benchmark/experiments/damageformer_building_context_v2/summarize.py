#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

BUILDING_VARIANTS = (
    "full_joint",
    "bir_on_frozen_c5",
    "bir_shuffle_on_frozen_c5",
    "rgb_only_building",
    "raw_sar_building",
    "no_conf_building",
    "shifted_sar_building",
    "wrong_building_joint",
)

OLD_VARIANTS = (
    "c1_lca",
    "c2_slico",
    "c3_rgb_assoc",
    "c4_raw_sar_assoc",
    "c5_aligned_assoc",
    "c6_no_confidence",
    "c7_shuffled_sar",
    "c8_damage_hint",
    "c9_boundary",
)


def load_metrics(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("best_metrics", {})


def print_row(group, name, m):
    if m is None:
        print(f"{group}\t{name}\tMISSING")
        return
    print(
        f"{group}\t{name}\t"
        f"{100*float(m.get('valid_miou',0)):.4f}\t"
        f"{100*float(m.get('miou4',0)):.4f}\t"
        f"{100*float(m.get('damaged_f1',0)):.4f}\t"
        f"{100*float(m.get('intact_iou',0)):.4f}\t"
        f"{100*float(m.get('damaged_iou',0)):.4f}\t"
        f"{100*float(m.get('destroyed_iou',0)):.4f}\t"
        f"{100*float(m.get('event_mean_miou4',0)):.4f}\t"
        f"{100*float(m.get('event_worst_miou4',0)):.4f}\t"
        f"{float(m.get('bir_gain',0)):.6f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old_root", required=True)
    p.add_argument("--new_root", required=True)
    p.add_argument("--tag", default="probe5000")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    print(
        "group\tvariant\tvalid_mIoU\tmIoU4\tDamagedF1\t"
        "IntactIoU\tDamagedIoU\tDestroyedIoU\t"
        "eventMean\teventWorst\tbirGain"
    )

    # Overall first in the table as well.
    for name in BUILDING_VARIANTS:
        path = Path(args.new_root) / f"{name}_{args.tag}_seed{args.seed}" / "training_summary.json"
        print_row("BUILDING", name, load_metrics(path))

    for name in OLD_VARIANTS:
        path = Path(args.old_root) / f"{name}_{args.tag}_seed{args.seed}" / "training_summary.json"
        print_row("REGION", name, load_metrics(path))


if __name__ == "__main__":
    main()
