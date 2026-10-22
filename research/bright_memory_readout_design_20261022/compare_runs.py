"""Compare two complete new-format runs under a matched protocol; no inference.

Recompute dataset mIoU from summed confusion matrices. Per-image differences
are descriptive and are not averaged to obtain the dataset mIoU.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_results import analyze, metrics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(report_path):
    report_path = Path(report_path).resolve()
    report = read(report_path)
    root = report_path.parent.parent
    manifest = read(root / "manifest.json")
    summary = read(root / "summary.json")
    if summary.get("complete") is not True or summary.get("smoke") is True:
        raise ValueError("Run must be complete and non-smoke: " + str(root))
    if report.get("format") != "BRIGHT_MEMORY_READOUT_DESIGN_V1_EVALUATION":
        raise ValueError("Use analyze_results.py for legacy multi-arm reports")
    if report.get("complete") is not True or report.get("final_integrity_pending") is True or report.get("smoke") is True:
        raise ValueError("Evaluation report is incomplete/pending/smoke")
    rows = [json.loads(line) for line in (report_path.parent / "images.jsonl").read_text().splitlines() if line.strip()]
    names = [r["image"] for r in rows]
    if names != report.get("requested_names") or len(names) != len(set(names)) or len(names) != report.get("image_count"):
        raise ValueError("Per-image names/count do not match report")
    for arm, entry in report["metrics"].items():
        summed = [[sum(row["confusion"][arm][r][c] for row in rows) for c in range(4)] for r in range(4)]
        if summed != entry["confusion"]:
            raise ValueError("Per-image confusion does not reproduce aggregate")
    return report, manifest, rows


def compare(before_path, after_path):
    a, ma, ra = load(before_path)
    b, mb, rb = load(after_path)
    for key in ("split", "step", "seed", "class_names", "requested_names"):
        if a.get(key) != b.get(key):
            raise ValueError("Unmatched report field: " + key)
    for key in ("initial_head_sha256", "train_names", "val_names", "actual_training_names",
                "fixed_train_names", "protocol_info", "parameter_count", "support_arm"):
        if key not in ma or key not in mb or ma[key] != mb[key]:
            raise ValueError("Unmatched/missing manifest field: " + key)
    for key in ("source", "addon_sha256", "extension_files", "post500_lr_multiplier", "scheduler_horizon",
                "warmup_steps", "maximum_lr", "weight_decay", "grad_clip", "optimizer", "endpoint_step"):
        if key not in ma["plan"] or ma["plan"][key] != mb["plan"].get(key):
            raise ValueError("Unmatched plan field: " + key)
    ac, bc = dict(ma["runtime_config"]), dict(mb["runtime_config"])
    ac.pop("output", None); bc.pop("output", None)
    if ac != bc:
        raise ValueError("Runtime configuration differs beyond output")
    merged = dict(a, metrics={"before": a["metrics"][a["variant"]], "after": b["metrics"][b["variant"]]})
    result = analyze(merged, "before", "after")
    per_image, events = [], {}
    for left, right in zip(ra, rb):
        if left["event"] != right["event"]:
            raise ValueError("Per-image event differs")
        lc, rc = left["confusion"][a["variant"]], right["confusion"][b["variant"]]
        lm, rm = metrics(lc), metrics(rc)
        if lm["ground_truth_counts"] != rm["ground_truth_counts"]:
            raise ValueError("Per-image GT class counts differ")
        per_image.append(dict(image=left["image"], event=left["event"],
                              before_miou4_pct=lm["miou4_pct"], after_miou4_pct=rm["miou4_pct"],
                              miou4_difference_pp=rm["miou4_pct"]-lm["miou4_pct"]))
        pair = events.setdefault(left["event"], [[[0]*4 for _ in range(4)] for _ in range(2)])
        for k, cm in enumerate((lc, rc)):
            for r in range(4):
                for c in range(4): pair[k][r][c] += cm[r][c]
    per_event = {event: {"before": metrics(pair[0]), "after": metrics(pair[1])} for event, pair in events.items()}
    result.update(before_variant=a["variant"], after_variant=b["variant"], seed=a["seed"],
                  matched_manifest_checked=True, per_event=per_event,
                  grid_confusion_identical=a["metrics"]["grid_probq"]["confusion"] == b["metrics"]["grid_probq"]["confusion"],
                  source_reports={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (before_path, after_path)},
                  note="No significance claim; between-run paired pixel rescue/harm is unavailable without paired predictions.")
    if not result["grid_confusion_identical"]:
        result["grid_comparison_note"] = "Frozen Grid differed across executions; inspect before interpreting tiny differences."
    return result, per_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    args = p.parse_args()
    result, rows = compare(args.before, args.after)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    with (args.output / "paired_images.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result["differences"], ensure_ascii=False, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
