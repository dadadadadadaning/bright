#!/usr/bin/env python3
"""Create a reproducible inventory of local BRIGHT logs, JSON results and tests."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


MIOU_KEYS = (
    "valid_miou",
    "best_valid_miou",
    "foreground_miou",
    "target_3class_miou",
    "official_style_4class_miou",
    "miou4",
    "mIoU4",
    "miou",
    "mIoU",
)
METRIC_KEYS = set(MIOU_KEYS) | {
    "damaged_iou",
    "destroyed_iou",
    "intact_iou",
    "damaged_f1",
    "destroyed_f1",
    "loc_f1",
    "damage_macro_f1",
    "affected_recall",
    "event_mean_official_style_4class_miou",
    "event_worst_official_style_4class_miou",
    "gain_pp",
    "damaged_f1_gain_pp",
    "ci95_low_pp",
    "ci95_high_pp",
}

VALID_RE = re.compile(r"valid_mIoU\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.I)
BEST_RE = re.compile(r"best(?:\s+clean)?\s+valid_mIoU\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.I)
STEP_RE = re.compile(r"(?:Training complete:\s*)?step\s*=\s*(\d+)", re.I)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any, key: str) -> float | str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    if key.endswith("_pp"):
        return number
    return 100.0 * number if abs(number) <= 1.5 else number


def infer_stage(path: Path, payload: Mapping[str, Any] | None = None) -> str:
    text = str(path).lower()
    split = str((payload or {}).get("split", "")).lower()
    if split == "test" or "independent_test" in text or "\\test\\" in text:
        return "test"
    if "oracle" in text:
        return "oracle"
    if "smoke" in text:
        return "smoke"
    if "calibration" in text or "meta_train" in text or "train_only" in text:
        return "internal"
    if "val450" in text or "validation_full" in text or "\\val\\" in text:
        return "validation"
    return "unspecified"


def recursive_flag(value: Any, names: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in names and item is True:
                return True
            if recursive_flag(item, names):
                return True
    elif isinstance(value, list):
        return any(recursive_flag(item, names) for item in value)
    return False


def walk_metric_dicts(value: Any, location: str = "", depth: int = 0):
    if depth > 12:
        return
    if isinstance(value, Mapping):
        present = [key for key in METRIC_KEYS if key in value and isinstance(value[key], (int, float))]
        if present:
            yield location or "/", value, present
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                yield from walk_metric_dicts(item, f"{location}/{key}", depth + 1)
    elif isinstance(value, list) and len(value) <= 5000:
        for index, item in enumerate(value):
            if isinstance(item, (Mapping, list)):
                yield from walk_metric_dicts(item, f"{location}/{index}", depth + 1)


def scan_logs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        valid_values = [float(item) for item in VALID_RE.findall(text)]
        best_values = [float(item) for item in BEST_RE.findall(text)]
        steps = [int(item) for item in STEP_RE.findall(text)]
        if not valid_values and not best_values and "error" not in text.lower():
            continue
        lower = text.lower()
        status = (
            "error"
            if any(token in lower for token in ("traceback", "filenotfounderror", "runtimeerror:"))
            else "complete"
            if "training complete" in lower or "independent test" in lower
            else "partial_or_audit"
        )
        rows.append(
            {
                "experiment_group": path.relative_to(root).parts[0],
                "file": str(path.resolve()),
                "stage": infer_stage(path),
                "status": status,
                "reported_values": len(valid_values),
                "last_valid_miou_pct": valid_values[-1] if valid_values else "",
                "max_reported_valid_miou_pct": max(valid_values) if valid_values else "",
                "best_valid_miou_pct": best_values[-1] if best_values else "",
                "last_step": steps[-1] if steps else "",
                "mentions_early_stop": "early stop" in lower or "early_stopped=true" in lower,
                "mentions_test_access_true": bool(
                    re.search(r"test_accessed[\"']?\s*[:=]\s*true", text, re.I)
                ),
            }
        )
    return rows


def scan_json(root: Path):
    files, metrics = [], []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        # A copied nested archive exists; retain the primary copy only.
        if len(relative.parts) > 1 and relative.parts[0] == "BRIGHT-eval-results":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as error:
            files.append(
                {
                    "experiment_group": relative.parts[0],
                    "file": str(path.resolve()),
                    "parse_status": f"error: {error}",
                }
            )
            continue
        top = payload if isinstance(payload, Mapping) else {}
        test_true = recursive_flag(
            payload,
            {"test_accessed", "official_test_accessed", "official_instances_accessed"},
        )
        files.append(
            {
                "experiment_group": relative.parts[0],
                "file": str(path.resolve()),
                "parse_status": "ok",
                "stage": infer_stage(path, top),
                "protocol": top.get("protocol", top.get("eval_protocol", "")),
                "variant": top.get("variant", ""),
                "split": top.get("split", ""),
                "pass": top.get("pass", ""),
                "decision": top.get("decision", ""),
                "test_access_true_anywhere": test_true,
            }
        )
        for location, record, keys in walk_metric_dicts(payload):
            protocol = record.get("protocol", top.get("protocol", top.get("eval_protocol", "")))
            variant = record.get("variant", top.get("variant", ""))
            for key in sorted(keys):
                metrics.append(
                    {
                        "experiment_group": relative.parts[0],
                        "file": str(path.resolve()),
                        "json_path": location,
                        "stage": infer_stage(path, top),
                        "protocol": protocol,
                        "variant": variant,
                        "metric": key,
                        "raw_value": record[key],
                        "percent_or_pp": pct(record[key], key),
                        "test_access_true_anywhere": test_true,
                    }
                )
    return files, metrics


def test_rows(log_rows: list[dict[str, Any]], eval_root: Path) -> list[dict[str, Any]]:
    output = []
    for row in log_rows:
        if row["stage"] == "test" or row["mentions_test_access_true"]:
            output.append(
                {
                    "model": Path(row["file"]).stem,
                    "source": row["file"],
                    "valid_miou_pct": row["max_reported_valid_miou_pct"],
                    "test_access_explicit": row["mentions_test_access_true"],
                    "source_kind": "test_log",
                }
            )
    for path in sorted((eval_root / "S8_pair" / "test").rglob("summary_official.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        output.append(
            {
                "model": path.parent.name,
                "source": str(path.resolve()),
                "valid_miou_pct": pct(payload.get("target_3class_miou"), "target_3class_miou"),
                "miou4_pct": pct(payload.get("official_style_4class_miou"), "official_style_4class_miou"),
                "damaged_iou_pct": pct(payload.get("damaged_iou"), "damaged_iou"),
                "destroyed_iou_pct": pct(payload.get("destroyed_iou"), "destroyed_iou"),
                "damaged_f1_pct": pct(payload.get("damaged_f1"), "damaged_f1"),
                "event_mean_miou4_pct": pct(
                    payload.get("event_mean_official_style_4class_miou"),
                    "event_mean_official_style_4class_miou",
                ),
                "test_access_explicit": True,
                "source_kind": "official_summary",
            }
        )
    return output


def scan_code(project_root: Path, train_root: Path, eval_root: Path) -> list[dict[str, Any]]:
    output = []
    experiments = project_root / "bda_benchmark" / "experiments"
    for directory in sorted(path for path in experiments.iterdir() if path.is_dir()):
        key = re.sub(r"[^a-z0-9]", "", directory.name.lower())
        train_hits = [path for path in train_root.iterdir() if key in re.sub(r"[^a-z0-9]", "", path.name.lower())]
        eval_hits = [path for path in eval_root.iterdir() if key in re.sub(r"[^a-z0-9]", "", path.name.lower())]
        output.append(
            {
                "experiment": directory.name,
                "code_directory": str(directory.resolve()),
                "has_run_sh": (directory / "run.sh").is_file(),
                "python_files": len(list(directory.rglob("*.py"))),
                "matching_train_directories": "; ".join(str(path.resolve()) for path in train_hits),
                "matching_eval_directories": "; ".join(str(path.resolve()) for path in eval_hits),
                "result_match_status": "matched" if train_hits or eval_hits else "no_name_match",
            }
        )
    return output


def file_inventory(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                rows.append(
                    {
                        "root": str(root.resolve()),
                        "relative_path": str(path.relative_to(root)),
                        "absolute_path": str(path.resolve()),
                        "extension": path.suffix.lower(),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    logs = scan_logs(args.train_root)
    json_files, json_metrics = scan_json(args.eval_root)
    tests = test_rows(logs, args.eval_root)
    code = scan_code(args.project_root, args.train_root, args.eval_root)
    inventory = file_inventory((args.train_root, args.eval_root))

    write_csv(args.output_root / "all_files.csv", inventory, list(inventory[0]))
    write_csv(args.output_root / "training_log_results.csv", logs, list(logs[0]))
    write_csv(args.output_root / "json_files.csv", json_files, list(json_files[0]))
    write_csv(args.output_root / "json_metric_records.csv", json_metrics, list(json_metrics[0]))
    write_csv(args.output_root / "test_results.csv", tests, [
        "model", "source", "valid_miou_pct", "miou4_pct", "damaged_iou_pct",
        "destroyed_iou_pct", "damaged_f1_pct", "event_mean_miou4_pct",
        "test_access_explicit", "source_kind",
    ])
    write_csv(args.output_root / "code_experiment_inventory.csv", code, list(code[0]))

    summary = {
        "train_root": str(args.train_root.resolve()),
        "eval_root": str(args.eval_root.resolve()),
        "files": len(inventory),
        "log_result_rows": len(logs),
        "json_files": len(json_files),
        "json_metric_records": len(json_metrics),
        "test_rows": len(tests),
        "code_experiments": len(code),
        "code_experiments_without_name_matched_results": sum(
            row["result_match_status"] == "no_name_match" for row in code
        ),
        "json_files_with_explicit_test_access": sum(
            bool(row.get("test_access_true_anywhere")) for row in json_files
        ),
    }
    (args.output_root / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
