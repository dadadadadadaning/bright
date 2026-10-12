"""Read-only Chinese summary of a completed formal Val450 region-path run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ARMS, FORMAT, PAIRS, metric_deltas, source_hash


IMMUTABLE_KEYS = ("files", "inherited_code", "new_code", "frozen_models",
                  "runtime", "protocol", "probe_code")


def checked_result(output):
    root = Path(output)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    for name, record in (("summary", summary), ("result", result)):
        if record.get("format") != FORMAT:
            raise ValueError(name + " is not this Val450 experiment")
        if record.get("complete") is not True or record.get("full_validation_evidence") is not True:
            raise ValueError(name + " has not passed full Val450 checks")
        if record.get("image_count") != 450 or record.get("test_accessed") is not False:
            raise ValueError(name + " image count or split provenance is invalid")
        immutable = record.get("immutable")
        if (not isinstance(immutable, dict) or
                any(immutable.get(key) is not True for key in IMMUTABLE_KEYS)):
            raise ValueError(name + " immutable checks failed or are absent")
        if record.get("native_a1_replay_ok") is not True:
            raise ValueError(name + " native A1 replay check failed")
    reproduction = result.get("reproduction", {})
    if (summary.get("reproduction_passed") is not True or
            reproduction.get("passed") is not True or
            reproduction.get("image_count") != 450 or
            reproduction.get("tolerance") != 1e-5 or
            not reproduction.get("checks") or
            any(row.get("passed") is not True for row in reproduction["checks"])):
        raise ValueError("Original five-arm 1e-5 reproduction did not pass")
    if (summary.get("immutable") != result.get("immutable") or
            summary.get("source_sha256") != result.get("source_sha256") or
            summary.get("result_file") != "result.json"):
        raise ValueError("Summary and result identify different evidence")
    if source_hash() != result["source_sha256"]:
        raise ValueError("The current probe/dependency code differs from the saved run")
    if result.get("expected_image_count") != 450 or result.get("training") is not False:
        raise ValueError("Result is not the frozen Val450 evaluation")
    guard = result.get("pool_guard", {})
    if guard.get("backend") != "native" or guard.get("restored") is not True or guard.get("failures"):
        raise ValueError("Native pooling guard did not finish cleanly")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(ARMS):
        raise ValueError("Nine-arm metric table is missing or incomplete")
    for arm in ARMS:
        row = metrics[arm]
        if row.get("valid_miou") is None or not isinstance(row.get("iou"), list) or len(row["iou"]) != 4:
            raise ValueError("Incomplete class metrics: " + arm)
    if set(result.get("transitions", {})) != set(PAIRS):
        raise ValueError("Paired transition table is missing or incomplete")
    return result


def format_report(result):
    metrics = result["metrics"]
    deltas = metric_deltas(metrics)

    def pct(value):
        return "NA" if value is None else f"{100.0 * value:.4f}"

    lines = [
        "Val450 正式对照：复现、权重/代码/运行时完整性、原生 A1 回放均通过。",
        "数值单位为百分比；顺序：Background / Intact / Damaged / Destroyed。",
        "路径             前景mIoU   Background   Intact   Damaged   Destroyed",
    ]
    for arm in ARMS:
        row = metrics[arm]
        iou = row["iou"]
        lines.append(f"{arm:<16} {pct(row['valid_miou']):>8}   " +
                     "   ".join(f"{pct(value):>8}" for value in iou))
    lines.append("")
    for pair, label in (
        ("pixel_grid__grid_probq", "Q 概率共享 pixel_grid→grid_probq"),
        ("a1_pixel__grid_probq", "原生 A1→grid_probq（计算顺序也不同）"),
        ("j1__j3", "J1→J3（分别训练，非纯 Memory 因果增益）"),
    ):
        lines.append(f"{label}: {deltas[pair]:+.4f} 个百分点")

    events = []
    for event, rows in result.get("per_event", {}).items():
        before, after = rows.get("a1_pixel"), rows.get("grid_probq")
        if not before or not after:
            continue
        x, y = before["iou"][1], after["iou"][1]
        if x is not None and y is not None:
            events.append((100.0 * (y - x), event, before.get("support", [None] * 4)[1]))
    events.sort(key=lambda item: (item[0], item[1]))
    if events:
        lines.append("")
        lines.append("Intact IoU 相对 A1 下降最多的事件（百分点；括号内为 GT Intact 像素数）：")
        for delta, event, pixels in events[:5]:
            lines.append(f"  {event}: {delta:+.4f}（{pixels}）")
    lines.append("统计单位是像素，不是建筑；此对照没有训练 grid_probq+Memory。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Completed formal Val450 output directory")
    args = parser.parse_args()
    print(format_report(checked_result(args.output)))


if __name__ == "__main__":
    main()
