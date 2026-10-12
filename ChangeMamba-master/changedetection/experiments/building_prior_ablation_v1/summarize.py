#!/usr/bin/env python3
"""Summarise the paired CM0/CM1 Validation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm0", required=True)
    parser.add_argument("--cm1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = {}
    for name, path in (("cm0_original", args.cm0), ("cm1_building_prior", args.cm1)):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("test_accessed") is not False:
            raise RuntimeError(f"{name} is not a Validation-only result")
        rows[name] = payload["validation"]

    cm0 = rows["cm0_original"]
    cm1 = rows["cm1_building_prior"]
    comparison = {
        "miou_gain_pp": 100.0 * (cm1["miou"] - cm0["miou"]),
        "oa_gain_pp": 100.0 * (cm1["oa"] - cm0["oa"]),
        "loc_f1_gain_pp": 100.0 * (cm1["loc_f1"] - cm0["loc_f1"]),
        "clf_f1_gain_pp": 100.0 * (cm1["clf_f1"] - cm0["clf_f1"]),
        "class_iou_gain_pp": [
            100.0 * (right - left)
            for left, right in zip(cm0["sub_iou"], cm1["sub_iou"])
        ],
        "building_transition": cm1.get("building_transition"),
    }
    output = {
        "protocol": "CHANGEMAMBA_BUILDING_PRIOR_ABLATION_V1_SUMMARY",
        "rows": rows,
        "cm1_minus_cm0": comparison,
        "test_accessed": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
