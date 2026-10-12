#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = {
    "P0_grid": "pgrid_partition.json",
    "P1_slico": "pslico_partition.json",
    "P2_rgb_structure": "p2_rgb_structure_partition.json",
    "P5_anchored_change": "p5_anchored_change_partition.json",
    "P6_shuffled_sar": "p6_shuffled_sar_partition.json",
    "P7_no_confidence": "p7_no_confidence_partition.json",
    "P8_no_damage_boundary": "p8_no_damage_boundary_partition.json",
}


def main() -> None:
    p = argparse.ArgumentParser("Summarize V3 partition audits")
    p.add_argument(
        "--audit_root",
        default="/root/data/BRIGHT-eval-results/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3",
    )
    p.add_argument("--output_json", default="")
    args = p.parse_args()
    root = Path(args.audit_root)
    rows = {}
    for name, filename in ROWS.items():
        path = root / filename
        if path.is_file():
            rows[name] = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    comparisons = {}
    for left, right, label in (
        ("P2_rgb_structure", "P1_slico", "P2_minus_P1"),
        ("P5_anchored_change", "P2_rgb_structure", "P5_minus_P2"),
        ("P5_anchored_change", "P6_shuffled_sar", "P5_minus_P6"),
        ("P5_anchored_change", "P7_no_confidence", "P5_minus_P7"),
        ("P5_anchored_change", "P8_no_damage_boundary", "P5_minus_P8"),
    ):
        if left in rows and right in rows:
            comparisons[label] = {
                key: rows[left][key] - rows[right][key]
                for key in rows[left]
                if key in rows[right]
            }
    result = {
        "protocol": "DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3_SUMMARY",
        "rows": rows,
        "comparisons": comparisons,
        "admission": {
            "rgb_learning": "P2>P1 on macro_iou/purity/boundary_recall",
            "change_value": "P5>P2",
            "correct_correspondence": "P5>P6",
            "reliability_value": "P5>P7",
            "damage_supervision_value": "P5>P8",
        },
        "test_accessed": False,
    }
    output = Path(args.output_json) if args.output_json else root / "partition_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
