#!/usr/bin/env python3
"""Read-only checkpoint and metric status for the controlled E0--E4 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def checkpoint_meta(path: Path) -> Dict[str, Any]:
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # status must continue when one archive is incomplete
        return {"read_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"format": type(payload).__name__}
    metrics = payload.get("best_metrics", payload.get("metrics", {}))
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "step": payload.get("step", -1),
        "variant": payload.get("variant", ""),
        "early_stopped": payload.get("early_stopped", False),
        "best_valid_miou": metrics.get("valid_miou"),
        "damaged_f1": metrics.get("damaged_f1"),
        "destroyed_iou": metrics.get("destroyed_iou"),
        "affected_recall": metrics.get("affected_recall"),
    }


def fmt_percent(value: Any) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) <= 1.5:
        number *= 100.0
    return f"{number:.4f}%"


def newest_checkpoint(directory: Path) -> Optional[Path]:
    candidates: Iterable[Path] = (
        directory / "final.pth",
        directory / "latest_resume.pth",
        directory / "best_val_miou.pth",
    )
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def main() -> None:
    parser = argparse.ArgumentParser("Inspect P5 ablation status without changing it")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--log_root", default="")
    args = parser.parse_args()

    root = Path(args.output_root)
    print(f"Output root: {root}")
    if not root.is_dir():
        print("Output root does not exist.")
        return

    experiment_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not experiment_dirs:
        print("No experiment directories found.")
        return

    for directory in experiment_dirs:
        rows = read_jsonl(directory / "metrics.jsonl")
        validations = [row for row in rows if "valid_miou" in row]
        latest_validation = validations[-1] if validations else None
        best_validation = max(validations, key=lambda row: float(row["valid_miou"])) \
            if validations else None
        checkpoint = newest_checkpoint(directory)
        metadata = checkpoint_meta(checkpoint) if checkpoint else {}
        completed = (directory / "final.pth").is_file()

        print("\n" + "=" * 78)
        print(directory.name)
        print(f"State       : {'COMPLETE' if completed else 'RUNNING/INCOMPLETE'}")
        print(f"Checkpoint  : {checkpoint.name if checkpoint else '--'}")
        print(f"Saved step  : {metadata.get('step', '--')}")
        if "read_error" in metadata:
            print(f"Read error  : {metadata['read_error']}")
        print(f"Best mIoU   : {fmt_percent(metadata.get('best_valid_miou'))}")
        if latest_validation:
            print(
                "Last val    : step={} | mIoU={} | D-F1={} | X-IoU={} | "
                "Affected-R={}".format(
                    latest_validation.get("step", "--"),
                    fmt_percent(latest_validation.get("valid_miou")),
                    fmt_percent(latest_validation.get("damaged_f1")),
                    fmt_percent(latest_validation.get("destroyed_iou")),
                    fmt_percent(latest_validation.get("affected_recall")),
                )
            )
        if best_validation:
            print(
                f"Best val row: step={best_validation.get('step', '--')} | "
                f"mIoU={fmt_percent(best_validation.get('valid_miou'))}"
            )
        print(f"Early stop  : {metadata.get('early_stopped', False)}")

    if args.log_root:
        log_root = Path(args.log_root)
        logs = sorted(log_root.glob("*.log"), key=lambda path: path.stat().st_mtime,
                      reverse=True) if log_root.is_dir() else []
        print("\n" + "=" * 78)
        print("Newest logs:")
        for path in logs[:8]:
            print(f"- {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()

