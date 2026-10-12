from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for variant in ("C0", "C1", "C2", "C3"):
        path = Path(args.root) / variant / "summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        action = data["train_cal_action"]["0.02"]
        rows.append({
            "variant": variant,
            "cal_delta_valid_miou_pp": action["delta_valid_miou_pp"],
            "cal_delta_damaged_iou_pp": action["delta_damaged_iou_pp"],
            "cal_ap": data["train_cal_diagnostics"]["ap"],
            "scores": str(path.parent / "scores.pt"),
        })
    selected = max(rows, key=lambda row: (
        row["cal_delta_valid_miou_pp"], row["cal_ap"] if row["cal_ap"] is not None else -1.0
    ))
    output = {
        "protocol": "DAMAGEFORMER_DECISION_CHAIN_V2_SPECIALIST_SELECTION",
        "selection_source": "Train-Cal only",
        "coverage": 0.02,
        "rows": rows,
        "selected": selected,
        "region_increment_pp": {
            row["variant"]: row["cal_delta_valid_miou_pp"] - rows[0]["cal_delta_valid_miou_pp"]
            for row in rows
        },
        "test_accessed": False,
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

