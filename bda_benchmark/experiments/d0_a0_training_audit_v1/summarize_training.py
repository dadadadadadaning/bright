#!/usr/bin/env python3
"""Summarize the complete D0/A0 training traces without touching test data."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "step" in value:
            rows.append(value)
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            by_step[int(row["step"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return [by_step[step] for step in sorted(by_step)]


def parse_validation_line(line: str) -> dict[str, float]:
    aliases = {
        "valid_mIoU": "valid_miou",
        "mIoU4": "miou4",
        "D-F1": "damaged_f1",
        "Affected R": "affected_recall",
        "Dmg gate-miss": "damaged_gate_miss",
        "Dmg->Int": "damaged_to_intact",
        "Des->Int": "destroyed_to_intact",
        "Int->Dmg": "intact_to_damaged",
    }
    metrics: dict[str, float] = {}
    for label, key in aliases.items():
        match = re.search(rf"{re.escape(label)}=({NUMBER})", line)
        if match:
            metrics[key] = float(match.group(1)) / 100.0
    iou = re.search(rf"IoU\(int/dmg/des\)=({NUMBER})/({NUMBER})/({NUMBER})", line)
    if iou:
        metrics.update(
            intact_iou=float(iou.group(1)) / 100.0,
            damaged_iou=float(iou.group(2)) / 100.0,
            destroyed_iou=float(iou.group(3)) / 100.0,
        )
    loc = re.search(rf"Loc P/R/F1=({NUMBER})/({NUMBER})/({NUMBER})", line)
    if loc:
        metrics.update(
            loc_precision=float(loc.group(1)) / 100.0,
            loc_recall=float(loc.group(2)) / 100.0,
            loc_f1=float(loc.group(3)) / 100.0,
        )
    event = re.search(rf"Event mean/worst=({NUMBER})/({NUMBER})", line)
    if event:
        metrics.update(
            event_mean_miou4=float(event.group(1)) / 100.0,
            event_worst_miou4=float(event.group(2)) / 100.0,
        )
    return metrics


def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    last_step: int | None = None
    if not path.exists():
        return train_rows, validation_rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                row = None
            if isinstance(row, dict) and "step" in row and "total_loss" in row:
                last_step = int(row["step"])
                train_rows.append(row)
        if "Validation |" in text or "Clean validation |" in text:
            metrics = parse_validation_line(text)
            if last_step is not None and metrics:
                validation_rows.append({"step": last_step, "metrics": metrics})
    return deduplicate(train_rows), deduplicate(validation_rows)


def read_history(path: Path) -> list[dict[str, Any]]:
    rows = read_json_lines(path)
    return deduplicate([row for row in rows if isinstance(row.get("metrics"), dict)])


def flatten_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"step": int(row["step"]), **row["metrics"]} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["step"]
    columns += sorted({key for row in rows for key in row if key != "step"})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_validation(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [row for row in rows if row.get("metrics", {}).get("valid_miou") is not None]
    return max(usable, key=lambda row: float(row["metrics"]["valid_miou"])) if usable else None


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    result: dict[str, Any] = {"exists": True, "path": str(path), "bytes": path.stat().st_size}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        result["step"] = payload.get("step")
        result["variant"] = payload.get("variant", payload.get("args", {}).get("variant"))
        result["best_metrics"] = payload.get("best_metrics", payload.get("metrics", {}))
        result["parent_checkpoint"] = payload.get("parent_checkpoint")
        del payload
    except Exception as error:  # metadata remains useful if torch is unavailable
        result["read_error"] = repr(error)
    return result


def load_eval(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return metrics if isinstance(metrics, dict) else None


def maybe_plot(output: Path, d0_train, a0_train, d0_val, a0_val) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, rows in (("D0", d0_train), ("A0", a0_train)):
        points = [(r["step"], r.get("total_loss")) for r in rows if r.get("total_loss") is not None]
        if points:
            axes[0].plot([p[0] for p in points], [p[1] for p in points], label=label, linewidth=1)
    for label, rows in (("D0", d0_val), ("A0", a0_val)):
        points = [(r["step"], r["metrics"].get("valid_miou")) for r in rows if r["metrics"].get("valid_miou") is not None]
        if points:
            axes[1].plot([p[0] for p in points], [100 * p[1] for p in points], marker="o", label=label)
    axes[0].set(title="Training loss", xlabel="Optimizer step", ylabel="Loss")
    axes[1].set(title="Validation mIoU", xlabel="Optimizer step", ylabel="mIoU (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    path = output / "d0_a0_training_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-log", type=Path, required=True)
    parser.add_argument("--d0-dir", type=Path, required=True)
    parser.add_argument("--a0-log", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d0-eval-json", type=Path)
    parser.add_argument("--a0-eval-json", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d0_train, d0_val = parse_log(args.d0_log)
    a0_train, a0_val_from_log = parse_log(args.a0_log)
    a0_history = read_history(args.a0_dir / "history.jsonl")
    a0_val = a0_history or a0_val_from_log

    write_csv(args.output_dir / "d0_train_curve.csv", d0_train)
    write_csv(args.output_dir / "d0_validation_curve.csv", flatten_validation(d0_val))
    write_csv(args.output_dir / "a0_train_curve.csv", a0_train)
    write_csv(args.output_dir / "a0_validation_curve.csv", flatten_validation(a0_val))

    d0_best = best_validation(d0_val)
    a0_best = best_validation(a0_val)
    d0_eval, a0_eval = load_eval(args.d0_eval_json), load_eval(args.a0_eval_json)
    comparison = None
    if d0_eval and a0_eval:
        keys = ("valid_miou", "damaged_f1", "intact_iou", "damaged_iou", "destroyed_iou", "affected_recall")
        comparison = {
            key: {
                "d0": d0_eval.get(key),
                "a0": a0_eval.get(key),
                "gain_pp": 100.0 * (float(a0_eval[key]) - float(d0_eval[key]))
                if d0_eval.get(key) is not None and a0_eval.get(key) is not None else None,
            }
            for key in keys
        }

    result = {
        "protocol": "D0_A0_TRAINING_AUDIT_V1",
        "d0": {
            "train_points": len(d0_train), "validation_points": len(d0_val),
            "last_logged_step": d0_train[-1]["step"] if d0_train else None,
            "best_from_curve": d0_best,
            "best_checkpoint": checkpoint_metadata(args.d0_dir / "best_val_miou.pth"),
            "latest_checkpoint": checkpoint_metadata(args.d0_dir / "latest_resume.pth"),
            "training_summary_exists": (args.d0_dir / "training_summary.json").exists(),
        },
        "a0": {
            "train_points": len(a0_train), "validation_points": len(a0_val),
            "last_logged_step": a0_train[-1]["step"] if a0_train else None,
            "best_from_curve": a0_best,
            "best_checkpoint": checkpoint_metadata(args.a0_dir / "best_val_miou.pth"),
            "latest_checkpoint": checkpoint_metadata(args.a0_dir / "latest_resume.pth"),
            "training_summary_exists": (args.a0_dir / "training_summary.json").exists(),
        },
        "fresh_validation_comparison": comparison,
        "test_accessed": False,
    }
    result["curve_png"] = maybe_plot(args.output_dir, d0_train, a0_train, d0_val, a0_val)
    json_path = args.output_dir / "d0_a0_training_audit.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def best_text(row: dict[str, Any] | None) -> str:
        if not row:
            return "-"
        return f"step {row['step']}, {100*row['metrics']['valid_miou']:.4f}%"

    lines = [
        "# D0 vs A0 training audit", "",
        "| Model | Last logged step | Validation points | Best curve mIoU | Complete summary |",
        "|---|---:|---:|---:|---:|",
        f"| D0 | {result['d0']['last_logged_step']} | {len(d0_val)} | {best_text(d0_best)} | {result['d0']['training_summary_exists']} |",
        f"| A0 | {result['a0']['last_logged_step']} | {len(a0_val)} | {best_text(a0_best)} | {result['a0']['training_summary_exists']} |",
        "", "Test accessed: NEVER", "",
    ]
    if comparison:
        lines += ["## Fresh validation comparison", "", "| Metric | D0 | A0 | Gain (pp) |", "|---|---:|---:|---:|"]
        for key, row in comparison.items():
            if row["gain_pp"] is not None:
                lines.append(f"| {key} | {100*row['d0']:.4f}% | {100*row['a0']:.4f}% | {row['gain_pp']:+.4f} |")
        lines.append("")
    md_path = args.output_dir / "d0_a0_training_audit.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
