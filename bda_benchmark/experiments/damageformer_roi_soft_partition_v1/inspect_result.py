#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    root = Path(args.output_dir)
    summary = root / "training_summary.json"

    if not summary.exists():
        print(f"NOT COMPLETE: {summary} does not exist")
        latest = root / "latest.pth"
        print("latest checkpoint:", latest, "exists=", latest.exists())
        history = root / "history.jsonl"
        if history.exists():
            rows = [x for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]
            if rows:
                print("last train record:")
                print(rows[-1])
        return

    data = json.loads(summary.read_text(encoding="utf-8"))
    m = data.get("best_metrics", {})
    print("protocol:", data.get("protocol"))
    print("completed_steps:", data.get("completed_steps"))
    print("test_accessed:", data.get("test_accessed"))
    print("best valid_mIoU:", 100 * float(m.get("valid_miou", 0)))
    print("best mIoU4:", 100 * float(m.get("miou4", 0)))
    print("Damaged F1:", 100 * float(m.get("damaged_f1", 0)))
    print("Intact IoU:", 100 * float(m.get("intact_iou", 0)))
    print("Damaged IoU:", 100 * float(m.get("damaged_iou", 0)))
    print("Destroyed IoU:", 100 * float(m.get("destroyed_iou", 0)))
    print("Event mean:", 100 * float(m.get("event_mean_miou4", 0)))
    print("Event worst:", 100 * float(m.get("event_worst_miou4", 0)))
    print("Region residual abs mean:", m.get("region_residual_abs_mean"))
    print("Alignment confidence mean:", m.get("alignment_confidence_mean"))
    print("Alignment entropy mean:", m.get("alignment_entropy_mean"))


if __name__ == "__main__":
    main()
