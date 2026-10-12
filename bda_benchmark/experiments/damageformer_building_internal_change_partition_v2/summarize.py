#!/usr/bin/env python3
"""Summarize V2 stage/variant training summaries under an output root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output_json", required=True)
    args = p.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob("training_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best = data.get("best_metrics", {}) or {}
        rows.append(
            {
                "path": str(path),
                "variant": (data.get("variant") or {}).get("name", ""),
                "stage": data.get("stage", ""),
                "completed_steps": data.get("completed_steps", 0),
                "valid_mIoU": best.get("valid_miou"),
                "mIoU4": best.get("miou4"),
                "damaged_iou": best.get("damaged_iou"),
                "damaged_f1": best.get("damaged_f1"),
                "destroyed_iou": best.get("destroyed_iou"),
                "weak_bag_accuracy": best.get("weak_bag_accuracy"),
                "change_alpha_abs_mean": best.get("change_alpha_abs_mean"),
                "feedback_gain": best.get("feedback_gain"),
                "test_accessed": data.get("test_accessed", None),
            }
        )
    output = {"count": len(rows), "rows": rows}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
