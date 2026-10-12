#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_root", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    root = Path(args.audit_root)
    rows = {}
    for path in sorted(root.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = data.get("variant", path.parent.name)
        perturbation = data.get("perturbation", path.stem)
        rows.setdefault(variant, {})[perturbation] = data["metrics"]
    baseline = rows.get("c0_a1", {}).get("none", {})
    summary = {"protocol": "DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1_SUMMARY", "rows": {}}
    for variant, audits in rows.items():
        clean = audits.get("none", {})
        shifts = [m.get("valid_miou") for name, m in audits.items() if name != "none"]
        summary["rows"][variant] = {
            "valid_miou": clean.get("valid_miou"),
            "gain_vs_a1_pp": None if not baseline or not clean else 100 * (clean["valid_miou"] - baseline["valid_miou"]),
            "damaged_f1": clean.get("damaged_f1"),
            "destroyed_iou": clean.get("destroyed_iou"),
            "worst_event_miou4": clean.get("event_worst_miou4"),
            "mean_shift_valid_miou": sum(shifts) / len(shifts) if shifts else None,
        }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

