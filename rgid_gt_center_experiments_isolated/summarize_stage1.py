#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser("Summarize the 2x2 Stage-1 GT experiment")
    parser.add_argument("--v1-binary", required=True)
    parser.add_argument("--v1-damage", required=True)
    parser.add_argument("--v2-binary", required=True)
    parser.add_argument("--v2-damage", required=True)
    args = parser.parse_args()
    paths = {
        "V1 model / binary GT": args.v1_binary,
        "V1 model / damage GT": args.v1_damage,
        "V2 model / binary GT": args.v2_binary,
        "V2 model / damage GT": args.v2_damage,
    }
    for label, path in paths.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        methods = data.get("methods", data)
        print(f"\n===== {label} =====")
        for mode in ("g0", "g1", "g2", "g3"):
            row = methods[mode]
            print(
                mode.upper(),
                f"F1={row['f1']:.6f}",
                f"P={row['precision']:.6f}",
                f"R={row['recall']:.6f}",
                f"merge={row['merge']}",
                f"split={row['split']}",
                f"mIoU={row['mean_matched_iou']:.6f}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
