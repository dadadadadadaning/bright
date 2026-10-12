#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze E1 priority-5 ablation, postprocess, and prototype results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path("/root/shared-nvme/BRIGHT-master")
E1_REF = {
    "experiment": "E1_Sampler_A11SHU_C640_s3407",
    "mIoU4": 0.635532,
    "target3": 0.525102,
    "intact_iou": 0.727220,
    "damaged_iou": 0.388059,
    "destroyed_iou": 0.460026,
    "damaged_to_intact": 0.269005,
}

FIELDS = [
    "source",
    "experiment",
    "checkpoint_or_config",
    "path",
    "mIoU4",
    "target3",
    "background_iou",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "damage_pair_iou",
    "damaged_to_intact",
    "destroyed_to_intact",
    "damaged_to_destroyed",
    "destroyed_to_damaged",
    "damaged_to_background",
    "destroyed_to_background",
    "under_rate",
    "severity_conf_rate",
    "load_missing_keys_count",
    "load_unexpected_keys_count",
]


ALIASES = {
    "mIoU4": ["mIoU4", "official_style_4class_miou", "official4"],
    "target3": ["target3", "target_3class_miou"],
    "background_iou": ["background_iou"],
    "intact_iou": ["intact_iou"],
    "damaged_iou": ["damaged_iou"],
    "destroyed_iou": ["destroyed_iou"],
    "damaged_to_intact": ["damaged_to_intact"],
    "destroyed_to_intact": ["destroyed_to_intact"],
    "damaged_to_destroyed": ["damaged_to_destroyed"],
    "destroyed_to_damaged": ["destroyed_to_damaged"],
    "damaged_to_background": ["damaged_to_background"],
    "destroyed_to_background": ["destroyed_to_background"],
    "load_missing_keys_count": ["load_missing_keys_count"],
    "load_unexpected_keys_count": ["load_unexpected_keys_count"],
}


def to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def fmt(value: Any) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.6f}"


def first_metric(data: Dict[str, Any], key: str) -> Any:
    for alias in ALIASES[key]:
        if alias in data:
            return data.get(alias)
    return ""


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_metrics_txt(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)", line)
        if match:
            data[match.group(1)] = match.group(2)
    return data


def read_confusion_csv(path: Path) -> Optional[List[List[float]]]:
    if not path.exists():
        return None
    rows: List[List[float]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 5:
                    continue
                rows.append([float(x) for x in row[1:5]])
    except Exception:
        return None
    return rows if len(rows) == 4 and all(len(row) == 4 for row in rows) else None


def metrics_from_confusion(conf: Optional[List[List[float]]]) -> Dict[str, Any]:
    if conf is None:
        return {}
    eps = 1e-12
    per_iou: List[float] = []
    out: Dict[str, Any] = {}
    names = ["background_iou", "intact_iou", "damaged_iou", "destroyed_iou"]
    for cls in range(4):
        tp = conf[cls][cls]
        fp = sum(conf[r][cls] for r in range(4) if r != cls)
        fn = sum(conf[cls][c] for c in range(4) if c != cls)
        denom = tp + fp + fn
        iou = tp / (denom + eps) if denom > 0 else 0.0
        per_iou.append(iou)
        out[names[cls]] = iou
    out["mIoU4"] = sum(per_iou) / 4.0
    out["target3"] = sum(per_iou[1:4]) / 3.0

    gt_damaged = sum(conf[2])
    gt_destroyed = sum(conf[3])
    out["damaged_to_intact"] = conf[2][1] / (gt_damaged + eps) if gt_damaged > 0 else 0.0
    out["damaged_to_destroyed"] = conf[2][3] / (gt_damaged + eps) if gt_damaged > 0 else 0.0
    out["damaged_to_background"] = conf[2][0] / (gt_damaged + eps) if gt_damaged > 0 else 0.0
    out["destroyed_to_intact"] = conf[3][1] / (gt_destroyed + eps) if gt_destroyed > 0 else 0.0
    out["destroyed_to_damaged"] = conf[3][2] / (gt_destroyed + eps) if gt_destroyed > 0 else 0.0
    out["destroyed_to_background"] = conf[3][0] / (gt_destroyed + eps) if gt_destroyed > 0 else 0.0
    return out


def read_load_status(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, Any] = {}
    missing = re.findall(r"Missing keys:\s*([0-9]+)", text)
    unexpected = re.findall(r"Unexpected keys:\s*([0-9]+)", text)
    if missing:
        out["load_missing_keys_count"] = missing[-1]
    if unexpected:
        out["load_unexpected_keys_count"] = unexpected[-1]
    return out


def merge_metrics(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(primary)
    for key, value in fallback.items():
        if key not in out or out.get(key) in ("", None):
            out[key] = value
    return out


def row_from_data(source: str, experiment: str, ckpt: str, path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source": source,
        "experiment": experiment,
        "checkpoint_or_config": ckpt,
        "path": str(path),
    }
    for key in ALIASES:
        row[key] = first_metric(data, key)
    dmg = to_float(row.get("damaged_iou"))
    des = to_float(row.get("destroyed_iou"))
    d2i = to_float(row.get("damaged_to_intact"))
    x2i = to_float(row.get("destroyed_to_intact"))
    d2x = to_float(row.get("damaged_to_destroyed"))
    x2d = to_float(row.get("destroyed_to_damaged"))
    row["damage_pair_iou"] = (dmg + des) / 2.0 if dmg is not None and des is not None else ""
    row["under_rate"] = (d2i + x2i) / 2.0 if d2i is not None and x2i is not None else ""
    row["severity_conf_rate"] = (d2x + x2d) / 2.0 if d2x is not None and x2d is not None else ""
    return row


def collect_eval_tree(root: Path, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("*/*/summary_official.json")):
        data = read_json(summary_path)
        if not data:
            data = read_metrics_txt(summary_path.with_name("official_metrics.txt"))
        data = merge_metrics(data, metrics_from_confusion(read_confusion_csv(summary_path.with_name("confusion_4class_official.csv"))))
        data = merge_metrics(data, read_load_status(summary_path.with_name("eval.log")))
        rows.append(row_from_data(source, summary_path.parent.parent.name, summary_path.parent.name, summary_path, data))
    for summary_path in sorted(root.glob("*/summary_official.json")):
        data = read_json(summary_path)
        if not data:
            data = read_metrics_txt(summary_path.with_name("official_metrics.txt"))
        data = merge_metrics(data, metrics_from_confusion(read_confusion_csv(summary_path.with_name("confusion_4class_official.csv"))))
        data = merge_metrics(data, read_load_status(summary_path.with_name("eval.log")))
        rows.append(row_from_data(source, summary_path.parent.name, summary_path.parent.name, summary_path, data))
    return rows


def rank_key(row: Dict[str, Any]) -> tuple:
    return (
        to_float(row.get("target3")) or -1.0,
        to_float(row.get("damaged_iou")) or -1.0,
        to_float(row.get("destroyed_iou")) or -1.0,
        -(to_float(row.get("under_rate")) or 99.0),
        -(to_float(row.get("severity_conf_rate")) or 99.0),
    )


def best_by_experiment(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=rank_key, reverse=True):
        best.setdefault(str(row["experiment"]), row)
    return sorted(best.values(), key=rank_key, reverse=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) if key not in {"source", "experiment", "checkpoint_or_config", "path"} else row.get(key, "") for key in FIELDS})


def delta(row: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if row is None or key not in E1_REF:
        return None
    val = to_float(row.get(key))
    return None if val is None else val - float(E1_REF[key])


def find_best(rows: List[Dict[str, Any]], exp_name: str) -> Optional[Dict[str, Any]]:
    matches = [row for row in rows if row.get("experiment") == exp_name]
    return sorted(matches, key=rank_key, reverse=True)[0] if matches else None


def judgment(row: Optional[Dict[str, Any]], name: str) -> str:
    if row is None:
        return f"{name}: no usable result found."
    return (
        f"{name}: best={row.get('checkpoint_or_config')} target3={fmt(row.get('target3'))} "
        f"(delta {fmt(delta(row, 'target3'))}), damaged={fmt(row.get('damaged_iou'))} "
        f"(delta {fmt(delta(row, 'damaged_iou'))}), destroyed={fmt(row.get('destroyed_iou'))} "
        f"(delta {fmt(delta(row, 'destroyed_iou'))}), Dmg->Int={fmt(row.get('damaged_to_intact'))} "
        f"(delta {fmt(delta(row, 'damaged_to_intact'))})."
    )


def compare_rows(candidate: Optional[Dict[str, Any]], baseline: Optional[Dict[str, Any]], name: str, baseline_name: str) -> str:
    if candidate is None:
        return f"{name}: no usable result found."
    if baseline is None:
        return f"{name}: baseline {baseline_name} not found; {judgment(candidate, name)}"

    def diff(key: str) -> str:
        cand = to_float(candidate.get(key))
        base = to_float(baseline.get(key))
        if cand is None or base is None:
            return ""
        return f"{cand - base:+.6f}"

    return (
        f"{name} vs {baseline_name}: target3 delta={diff('target3')}, "
        f"damaged IoU delta={diff('damaged_iou')}, destroyed IoU delta={diff('destroyed_iou')}, "
        f"UnderRate delta={diff('under_rate')}, SeverityConfRate delta={diff('severity_conf_rate')}."
    )


def write_report(path: Path, all_rows: List[Dict[str, Any]], best_rows: List[Dict[str, Any]]) -> None:
    e1 = find_best(all_rows, "E1_Sampler_A11SHU_C640_s3407")
    no_cons = find_best(all_rows, "E1_noCons_Sampler_A11SHU_C640_s3407")
    stronger = find_best(all_rows, "E1_strongerSampler_A11SHU_C640_s3407")
    no_cons_stronger = find_best(all_rows, "E1_noCons_strongerSampler_A11SHU_C640_s3407")
    best_post = next((row for row in best_rows if row["source"] == "postprocess"), None)
    best_proto = next((row for row in best_rows if row["source"] == "prototype"), None)
    lines = [
        "# E1 Priority5 Analysis Report",
        "",
        f"Total results: {len(all_rows)}",
        "",
        "## E1 Reference",
        "",
        f"- target3: {E1_REF['target3']:.6f}",
        f"- damaged_iou: {E1_REF['damaged_iou']:.6f}",
        f"- destroyed_iou: {E1_REF['destroyed_iou']:.6f}",
        f"- damaged_to_intact: {E1_REF['damaged_to_intact']:.6f}",
        "",
        "## Best Overall",
        "",
    ]
    for row in best_rows[:10]:
        lines.append(
            f"- {row['experiment']} / {row['checkpoint_or_config']}: "
            f"target3={fmt(row.get('target3'))}, damaged={fmt(row.get('damaged_iou'))}, "
            f"destroyed={fmt(row.get('destroyed_iou'))}, under_rate={fmt(row.get('under_rate'))}"
        )
    lines += [
        "",
        "## Auto Conclusions",
        "",
        "1. E1_noCons vs E1_Sampler",
        "",
        judgment(no_cons, "E1_noCons"),
        compare_rows(no_cons, e1, "E1_noCons", "E1_Sampler"),
        "",
        "2. E1_strongerSampler vs E1_Sampler",
        "",
        judgment(stronger, "E1_strongerSampler"),
        compare_rows(stronger, e1, "E1_strongerSampler", "E1_Sampler"),
        "",
        "3. E1_noCons_strongerSampler vs single changes",
        "",
        judgment(no_cons_stronger, "E1_noCons_strongerSampler"),
        compare_rows(no_cons_stronger, no_cons, "E1_noCons_strongerSampler", "E1_noCons"),
        compare_rows(no_cons_stronger, stronger, "E1_noCons_strongerSampler", "E1_strongerSampler"),
        "",
        "4. E1_postprocess_consistency vs E1_Sampler",
        "",
        judgment(best_post, "E1_postprocess_consistency"),
        compare_rows(best_post, e1, "E1_postprocess_consistency", "E1_Sampler"),
        "",
        "5. E1_prototype_memory vs E1_Sampler",
        "",
        judgment(best_proto, "E1_prototype_memory"),
        compare_rows(best_proto, e1, "E1_prototype_memory", "E1_Sampler"),
        "",
        "## Ranking Rule",
        "",
        "Priority: target3, damaged_iou, destroyed_iou, lower UnderRate, lower SeverityConfRate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    out_root = root / "eval_results_official" / "E1_priority5_analysis"
    rows: List[Dict[str, Any]] = []
    rows.extend(collect_eval_tree(root / "eval_results_official" / "E1_priority5_fixed_test", "fixed_eval"))
    rows.extend(collect_eval_tree(root / "eval_results_official" / "E1_postprocess_consistency", "postprocess"))
    rows.extend(collect_eval_tree(root / "eval_results_official" / "E1_prototype_memory", "prototype"))
    rows = sorted(rows, key=rank_key, reverse=True)
    best = best_by_experiment(rows)
    write_csv(out_root / "E1_priority5_all_results.csv", rows)
    write_csv(out_root / "E1_priority5_best_by_experiment.csv", best)
    write_report(out_root / "E1_priority5_report.md", rows, best)
    print(f"Wrote E1 priority5 analysis to: {out_root}")


if __name__ == "__main__":
    main()
