#!/usr/bin/env python3
"""Create JSON/CSV/Markdown records for completed A0-A3 runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = ("a0", "a1", "a2", "a3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tag", default="probe5000")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    root = Path(args.output_root)
    records = []
    for variant in ORDER:
        directory = root / f"{variant}_{args.tag}_seed{args.seed}"
        path = directory / "training_summary.json"
        if not path.exists():
            records.append({"variant": variant, "state": "MISSING", "directory": str(directory)})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("best_metrics", {})
        records.append({
            "variant": variant, "state": "COMPLETE", "directory": str(directory),
            "best_step": payload.get("best_step"),
            "valid_miou": metrics.get("valid_miou"), "miou4": metrics.get("miou4"),
            "damaged_f1": metrics.get("damaged_f1"),
            "intact_iou": metrics.get("intact_iou"),
            "damaged_iou": metrics.get("damaged_iou"),
            "destroyed_iou": metrics.get("destroyed_iou"),
            "event_mean_miou4": metrics.get("event_mean_miou4"),
            "event_worst_miou4": metrics.get("event_worst_miou4"),
            "event_miou4_by_name": metrics.get("event_miou4_by_name", {}),
            "test_accessed": payload.get("test_accessed", False),
        })
    complete = {row["variant"]: row for row in records if row["state"] == "COMPLETE"}
    pairs = []
    for left, right, meaning in (
        ("a1", "a0", "soft ROI"),
        ("a2", "a1", "building-bag supervision"),
        ("a3", "a2", "hierarchical auxiliary supervision"),
    ):
        if left not in complete or right not in complete:
            continue
        candidate, reference = complete[left], complete[right]
        per_event_left = candidate["event_miou4_by_name"]
        per_event_right = reference["event_miou4_by_name"]
        common = sorted(set(per_event_left) & set(per_event_right))
        gains = {name: 100.0 * (per_event_left[name] - per_event_right[name]) for name in common}
        gain_pp = 100.0 * (candidate["valid_miou"] - reference["valid_miou"])
        damaged_gain_pp = 100.0 * (candidate["damaged_f1"] - reference["damaged_f1"])
        nonnegative = sum(value >= 0 for value in gains.values())
        pairs.append({
            "candidate": left, "reference": right, "meaning": meaning,
            "gain_pp": gain_pp,
            "damaged_f1_gain_pp": damaged_gain_pp,
            "event_nonnegative": nonnegative,
            "event_total": len(gains), "event_gain_pp": gains,
            "checks": {
                "miou_improved": gain_pp > 0.0,
                "damaged_f1_nonnegative": damaged_gain_pp >= 0.0,
                "at_least_10_of_14_events_nonnegative": len(gains) == 14 and nonnegative >= 10,
            },
        })
    result = {
        "protocol": "P3_DAMAGEFORMER_ABLATION_SUMMARY_V1", "tag": args.tag,
        "seed": args.seed, "variants": records, "pairwise": pairs,
        "test_accessed": False,
    }
    json_path = root / f"ablation_{args.tag}_seed{args.seed}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = root / f"ablation_{args.tag}_seed{args.seed}.csv"
    columns = ["variant", "state", "best_step", "valid_miou", "miou4", "damaged_f1", "intact_iou", "damaged_iou", "destroyed_iou", "event_mean_miou4", "event_worst_miou4"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in columns})
    md_path = root / f"ablation_{args.tag}_seed{args.seed}.md"
    lines = ["# DamageFormer → P3 controlled ablation", "", "| Variant | State | Best step | valid mIoU | D-F1 |", "|---|---:|---:|---:|---:|"]
    for row in records:
        value = "-" if row.get("valid_miou") is None else f"{100*row['valid_miou']:.4f}%"
        damaged = "-" if row.get("damaged_f1") is None else f"{100*row['damaged_f1']:.4f}%"
        lines.append(f"| {row['variant']} | {row['state']} | {row.get('best_step', '-')} | {value} | {damaged} |")
    lines += ["", "## Pairwise attribution", ""]
    for pair in pairs:
        lines.append(
            f"- {pair['candidate']} - {pair['reference']} ({pair['meaning']}): "
            f"mIoU {pair['gain_pp']:+.4f} pp, D-F1 {pair['damaged_f1_gain_pp']:+.4f} pp, "
            f"event nonnegative {pair['event_nonnegative']}/{pair['event_total']}."
        )
    lines += ["", "Test accessed: NEVER", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
