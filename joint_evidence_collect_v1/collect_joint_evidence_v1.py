#!/usr/bin/env python3
"""Copy allowlisted BRIGHT reports and summarize observed Train/Val evidence.

Standard library only. Never imports the experiment, opens checkpoints/images,
evaluates a model, resumes a job, scans a source tree, or uses the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from datetime import datetime, timezone
import zipfile

DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D"
DEFAULT_DIAGNOSTIC = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_GENERALIZATION_DIAGNOSTIC_V1/full_20261016_090249_389262"
DEFAULT_OUTPUT = "/root/shared-nvme/BRIGHT-master/New"
PATHS = ("grid_probq", "a", "b", "c", "d")
ARMS = PATHS[1:]
WINDOWS = ((1, 500), (501, 1000), (1001, 1500), (4501, 5000))
VAL_DIRS = ((0, "initial_val"), (500, "val_step_000500"),
            (1000, "val_step_001000"), (1500, "val_step_001500"), (5000, "final_val"))
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_TAIL_BYTES = 128 * 1024
MAX_JOURNAL_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def number(value, percent=False):
    if not finite(value):
        return "NA"
    value = value * 100 if percent else value
    return "%.4f" % value if finite(value) else "NA"


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def parse_json(data):
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number: " + value)
        return parsed
    result = json.loads(data, parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def safe_open(path):
    """Refuse source links, special files, Test paths and non-allowlisted redirects."""
    if any(part.lower() == "test" for part in path.parts):
        raise ValueError("refusing a Test path")
    if path.resolve(strict=False) != path:
        raise ValueError("refusing symlink source path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("source is not a regular file")
    return os.fdopen(fd, "rb")


def metadata(info):
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "device": info.st_dev, "inode": info.st_ino}


def capture(path, output, relative, *, json_report=True, tail=False):
    record = {"source": str(path), "kind": "log_tail" if tail else "report",
              "status": "unknown", "snapshot": None}
    parsed = None
    try:
        with safe_open(path) as stream:
            before = metadata(os.fstat(stream.fileno()))
            record["source_before"] = before
            if not tail and before["size"] > MAX_REPORT_BYTES:
                record.update(status="oversize", limit_bytes=MAX_REPORT_BYTES)
                return record, None
            offset = max(0, before["size"] - MAX_TAIL_BYTES) if tail else 0
            stream.seek(offset)
            data = stream.read(before["size"] - offset if tail else MAX_REPORT_BYTES + 1)
            after = metadata(os.fstat(stream.fileno()))
        record.update(source_after=after, source_changed_during_read=before != after)
        if not tail and len(data) > MAX_REPORT_BYTES:
            record.update(status="oversize", limit_bytes=MAX_REPORT_BYTES)
            return record, None
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        record.update(status="copied", snapshot=relative, snapshot_bytes=len(data),
                      snapshot_sha256=hashlib.sha256(data).hexdigest())
        if tail:
            record.update(offset_bytes=offset, is_tail=offset > 0,
                          note="Bounded raw bytes; first/last line may be partial.")
        if json_report:
            try:
                parsed = parse_json(data)
                record["json_status"] = "parsed_object"
            except (ValueError, UnicodeError, RecursionError) as error:
                record.update(status="parse_error", json_status="parse_error", error=str(error))
    except FileNotFoundError:
        record["status"] = "missing"
    except (OSError, ValueError) as error:
        record.update(status="read_error", error=str(error))
    return record, parsed


def statistics(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    count = len(values)
    scale = max(abs(ordered[0]), abs(ordered[-1]))
    # Scaling keeps valid huge finite observations from overflowing the sum.
    mean = (scale * max(-1.0, min(1.0, math.fsum(x / scale for x in values) / count))
            if scale else 0.0)
    middle = count // 2
    median = (ordered[middle] if count % 2 else
              ordered[middle - 1] / 2 + ordered[middle] / 2)
    return {"n": count, "mean": mean, "median": median,
            "min": ordered[0], "max": ordered[-1]}


def training_windows(path):
    record = {"source": str(path), "kind": "streamed_journal", "status": "unknown",
              "snapshot": None, "note": "Statistics only; raw training journal is not copied."}
    rows = {}
    counts = {"lines": 0, "blank_lines": 0, "malformed_lines": 0,
              "oversize_lines": 0, "duplicate_steps": 0, "invalid_steps": 0,
              "unterminated_lines": 0, "truncated_json_lines": 0}
    seen = set()
    digest = hashlib.sha256()
    read_bytes = 0
    try:
        with safe_open(path) as stream:
            before = metadata(os.fstat(stream.fileno()))
            record["source_before"] = before
            limit = min(before["size"], MAX_JOURNAL_BYTES)
            record["byte_limit_reached"] = before["size"] > MAX_JOURNAL_BYTES
            while read_bytes < limit:
                line = stream.readline(min(MAX_LINE_BYTES + 1, limit - read_bytes))
                if not line:
                    break
                digest.update(line)
                read_bytes += len(line)
                counts["lines"] += 1
                if len(line) > MAX_LINE_BYTES:
                    counts["oversize_lines"] += 1
                    while not line.endswith(b"\n") and read_bytes < limit:
                        line = stream.readline(min(MAX_LINE_BYTES, limit - read_bytes))
                        if not line:
                            break
                        digest.update(line)
                        read_bytes += len(line)
                    if not line.endswith(b"\n"):
                        counts["unterminated_lines"] += 1
                    continue
                unterminated = not line.endswith(b"\n")
                counts["unterminated_lines"] += int(unterminated)
                if not line.strip():
                    counts["blank_lines"] += 1
                    continue
                try:
                    # Preserve non-finite fields so each field can be excluded separately.
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                except (ValueError, UnicodeError, RecursionError):
                    counts["malformed_lines"] += 1
                    counts["truncated_json_lines"] += int(unterminated)
                    continue
                step = row.get("step")
                if type(step) is not int or step < 1:
                    counts["invalid_steps"] += 1
                    continue
                if step in seen:
                    counts["duplicate_steps"] += 1
                    continue
                seen.add(step)
                if any(low <= step <= high for low, high in WINDOWS):
                    rows[step] = row
            after = metadata(os.fstat(stream.fileno()))
        record.update(status="streamed", source_after=after,
                      source_changed_during_read=before != after,
                      observed_bytes=read_bytes, observed_bytes_sha256=digest.hexdigest())
    except FileNotFoundError:
        record["status"] = "missing"
    except (OSError, ValueError) as error:
        record.update(status="read_error", error=str(error))
    record.update(counts=counts, unique_steps=len(seen), min_step=min(seen) if seen else None,
                  max_step=max(seen) if seen else None,
                  observed_all_steps_1_to_5000=all(step in seen for step in range(1, 5001)),
                  duplicate_policy="first parseable row for each step retained; duplicates excluded")
    windows = {}
    for low, high in WINDOWS:
        selected = [row for step, row in rows.items() if low <= step <= high]
        expected = high - low + 1
        item = {"observed_rows": len(selected), "expected_rows": expected,
                "missing_steps": expected - len(selected), "heads": {}}
        events = {}
        image_names = set()
        event_available = image_available = 0
        for row in selected:
            if isinstance(row.get("event"), str) and row["event"]:
                event_available += 1
                events[row["event"]] = events.get(row["event"], 0) + 1
            if isinstance(row.get("image"), str) and row["image"]:
                image_available += 1
                image_names.add(row["image"])
        item.update(event_counts=events, rows_with_event=event_available,
                    rows_without_event=len(selected) - event_available,
                    unique_images=len(image_names) if image_available else None,
                    rows_with_image=image_available,
                    rows_without_image=len(selected) - image_available)
        for arm in ARMS:
            result = {}
            for field in ("loss", "lr", "grad_norm"):
                values, absent, invalid = [], 0, 0
                for row in selected:
                    heads = row.get("heads")
                    head = heads.get(arm) if isinstance(heads, dict) else None
                    head = head if isinstance(head, dict) else {}
                    # Existing training schema stores learning rate at row level.
                    value = row.get("lr") if field == "lr" else head.get(field)
                    if value is None:
                        absent += 1
                    elif finite(value):
                        values.append(value)
                    else:
                        invalid += 1
                result[field] = dict(statistics(values), missing_or_null=absent,
                                     invalid_or_nonfinite=invalid)
            item["heads"][arm] = result
        windows[f"{low}-{high}"] = item
    return record, windows


def diagnostic_status(summary, inventory_status):
    if summary is None:
        return {"state": "unavailable", "summary_status": inventory_status,
                "note": "缺少可读 summary 不等于从未运行；参见日志尾部和 traceback。"}
    results = summary.get("results")
    results = results if isinstance(results, dict) else {}
    steps = summary.get("steps")
    steps = steps if isinstance(steps, list) and steps else [500, 1500, 5000]
    missing = []
    for step in steps:
        result = results.get(str(step))
        for split in ("fixed_train", "val"):
            report = result.get(split) if isinstance(result, dict) else None
            metrics = report.get("metrics") if isinstance(report, dict) else None
            for arm in PATHS:
                row = metrics.get(arm) if isinstance(metrics, dict) else None
                nested = row.get("metrics") if isinstance(row, dict) else None
                if not (isinstance(nested, dict) and all(finite(row.get(name)) for name in
                        ("mean_image_loss", "mean_image_nll", "mean_image_lovasz"))
                        and isinstance(nested.get("iou"), list)
                        and isinstance(nested.get("confusion"), list)):
                    missing.append(f"{step}/{split}/{arm}")
    state = ("reported_incomplete" if summary.get("complete") is not True else
             "reported_complete_but_missing_fields" if missing else "reported_complete")
    return {"state": state, "reported_complete": summary.get("complete"),
            "reported_execution_complete": summary.get("execution_complete"),
            "mode": summary.get("mode"), "available_steps": sorted(results),
            "missing_metric_groups": missing, "error": summary.get("error"),
            "note": "仅转述已保存报告，不验证执行、权重、数据或完整性。"}


def render(root, diagnostic, inventory, parsed, windows, journal, diagnosis):
    lines = ["BRIGHT 联合证据收集（只读，观察性汇总）", f"原训练目录：{root}",
             f"诊断目录：{diagnostic}", "", "文件状态（missing ≠ 从未运行）："]
    lines.extend(f"  {key}: {value['status']}" for key, value in inventory.items())
    lines.extend(["", "保存的 Val 指标（%，不重跑；NA=未提供/不可读；未验证完成性）",
                  "step | path | fg-mIoU | IoU BG/Intact/Damaged/Destroyed | image_count | execution_complete"])
    for step, folder in VAL_DIRS:
        report = parsed.get("val_" + str(step))
        if not isinstance(report, dict):
            lines.append(f"{step} | NA（{inventory['val_' + str(step)]['status']}）")
            continue
        metrics = report.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        for arm in PATHS:
            row = metrics.get(arm)
            row = row if isinstance(row, dict) else {}
            iou = row.get("iou")
            iou = iou if isinstance(iou, list) and len(iou) == 4 else [None] * 4
            lines.append(f"{step} | {arm} | {number(row.get('foreground_miou'), True)} | "
                         + "/".join(number(x, True) for x in iou)
                         + f" | {report.get('image_count', 'NA')} | {report.get('execution_complete', 'NA')}")
    lines.extend(["", "Train journal 窗口（首个可解析 step 保留；重复行不计入）：",
                  "window | rows/500 | head | loss mean/median[n] | lr mean[n] | grad_norm mean[n]"])
    for window, item in windows.items():
        for arm in ARMS:
            fields = item["heads"][arm]
            loss, lr, grad = (fields[name] for name in ("loss", "lr", "grad_norm"))
            lr_text = "%.3e" % lr["mean"] if finite(lr["mean"]) else "NA"
            lines.append(f"{window} | {item['observed_rows']}/{item['expected_rows']} | {arm} | "
                         f"{number(loss['mean'])}/{number(loss['median'])}[{loss['n']}] | "
                         f"{lr_text}[{lr['n']}] | {number(grad['mean'])}[{grad['n']}]")
        lines.append(f"  event 记录 {item['rows_with_event']}/{item['observed_rows']}；"
                     f"image 记录 {item['rows_with_image']}/{item['observed_rows']}；"
                     f"可见唯一 image={item['unique_images'] if item['unique_images'] is not None else 'NA'}；"
                     "event 分布见 inventory.json（只用 event 字段）。")
    lines.append("日志状态：" + json.dumps({key: journal.get(key) for key in
                 ("status", "unique_steps", "min_step", "max_step", "counts",
                  "source_changed_during_read", "byte_limit_reached")}, ensure_ascii=False))
    lines.extend(["不同窗口采样图像可能不同；训练窗口 loss 变化不能单独诊断过拟合。",
                  "缺失、null、非有限数不记作 0；n 是各字段有效样本数，缺失/无效计数见 inventory.json。",
                  "", "固定 Train / Val 诊断：" + diagnosis["state"]
                  + "；mode=" + str(diagnosis.get("mode", "NA")),
                  "诊断状态：" + json.dumps(diagnosis, ensure_ascii=False),
                  "step | split | path | mean loss (NLL+Lovasz) | fg-mIoU% | IoU BG/Intact/Damaged/Destroyed%"])
    summary = parsed.get("diagnostic_summary")
    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    results = results if isinstance(results, dict) else {}
    for step in sorted(results, key=lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item))):
        entry = results[step]
        if not isinstance(entry, dict):
            continue
        for split in ("fixed_train", "val"):
            report = entry.get(split)
            if isinstance(report, dict):
                identity = {key: report.get(key) for key in
                            ("image_count", "names_sha256", "order_sha256", "loss_definition",
                             "same_image_event_exclusion_passed")}
                lines.append(f"{step}/{split} 已保存元数据：" + json.dumps(identity, ensure_ascii=False))
            metrics = report.get("metrics") if isinstance(report, dict) else None
            metrics = metrics if isinstance(metrics, dict) else {}
            for arm in PATHS:
                row = metrics.get(arm)
                if not isinstance(row, dict):
                    lines.append(f"{step} | {split} | {arm} | NA")
                    continue
                metric = row.get("metrics")
                metric = metric if isinstance(metric, dict) else {}
                iou = metric.get("iou")
                iou = iou if isinstance(iou, list) and len(iou) == 4 else [None] * 4
                lines.append(f"{step} | {split} | {arm} | {number(row.get('mean_image_loss'))} "
                             f"({number(row.get('mean_image_nll'))}+{number(row.get('mean_image_lovasz'))}) | "
                             f"{number(metric.get('foreground_miou'), True)} | "
                             + "/".join(number(x, True) for x in iou))
                lines.append("  confusion GT行/预测列=BG,Intact,Damaged,Destroyed: "
                             + json.dumps(metric.get("confusion"), ensure_ascii=False))
        replay = entry.get("val_reproduction")
        lines.append(f"{step} Val reproduction reported passed="
                     + str(replay.get("passed") if isinstance(replay, dict) else "NA"))
    lines.extend(["", "原始 metrics/per_event 等字段保留在 snapshots 内原 JSON；未读取模型或原始数据。",
                  "日志尾部可能包含局部进度；summary 未更新、缺失或不完整不等于任务从未运行。",
                  "本次仅收集文件快照，不验证权重或流水线完成性；没有自动重启、训练或评估。",
                  "单次种子、事后固定 Train 样本及现有 Val 只能提供诊断线索，不能证明稳定改进。"])
    return "\n".join(lines) + "\n"


def collect(root, diagnostic, output_parent):
    # Canonicalize user-selected roots once so mounted/symlinked directory aliases work.
    # Individual report paths are still refused if they redirect through a link.
    root = Path(root).expanduser().resolve()
    diagnostic = Path(diagnostic).expanduser().resolve()
    source_roots = (root, diagnostic)
    output_parent = Path(output_parent).expanduser().resolve()
    if any(output_parent == source or source in output_parent.parents for source in source_roots):
        raise ValueError("output-parent must be outside both source roots")
    if any(part.lower() == "test" for path in (root, diagnostic) for part in path.parts):
        raise ValueError("source roots cannot be Test paths")
    output_parent.mkdir(parents=True, exist_ok=True)
    prefix = "evidence_collect_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_")
    output = Path(tempfile.mkdtemp(prefix=prefix, dir=output_parent))
    inventory, parsed = {}, {}
    files = [("workflow_state", root / "workflow_state.json", "original/workflow_state.json"),
             ("source", root / "source.json", "original/source.json"),
             ("external_source", Path(str(root) + ".source.json"), "original/external.source.json"),
             ("train_summary", root / "train/summary.json", "original/train/summary.json"),
             ("train_manifest", root / "train/manifest.json", "original/train/manifest.json")]
    files += [("val_" + str(step), root / "train" / folder / "result.json",
               "original/train/" + folder + "/result.json") for step, folder in VAL_DIRS]
    files.append(("diagnostic_summary", diagnostic / "summary.json", "diagnostic/summary.json"))
    for key, path, relative in files:
        inventory[key], parsed[key] = capture(path, output, "snapshots/" + relative)
    inventory["diagnostic_traceback"], _ = capture(
        diagnostic / "traceback.txt", output, "snapshots/diagnostic/traceback.txt", json_report=False)
    for key, path, relative in (
            ("root_log_tail", Path(str(root) + ".log"), "original/root.log.tail.txt"),
            ("train_log_tail", root / "logs/train.log", "original/train.log.tail.txt"),
            ("diagnostic_log_tail", Path(str(diagnostic) + ".log"), "diagnostic/run.log.tail.txt")):
        inventory[key], _ = capture(path, output, "snapshots/" + relative, json_report=False, tail=True)
    journal, windows = training_windows(root / "train/train.jsonl")
    inventory["train_journal"] = journal
    diagnosis = diagnostic_status(parsed.get("diagnostic_summary"), inventory["diagnostic_summary"]["status"])
    record = {"format": "BRIGHT_JOINT_EVIDENCE_COLLECT_V1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "root": str(root), "diagnostic": str(diagnostic), "output": str(output),
              "read_only_sources": True, "observational_only": True,
              "limits_bytes": {"report": MAX_REPORT_BYTES, "log_tail": MAX_TAIL_BYTES,
                               "journal": MAX_JOURNAL_BYTES, "journal_line": MAX_LINE_BYTES},
              "files": inventory, "training_windows": windows, "diagnostic_status": diagnosis,
              "interpretation": "Missing does not mean never run. No verification of weights or pipeline completion."}
    report_path = output / "REPORT.txt"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(render(root, diagnostic, inventory, parsed, windows, journal, diagnosis))
    inventory_path = output / "inventory.json"
    with inventory_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    zip_path = output.with_suffix(".zip")
    # Only the explicit generated file list is archived; no source-directory scan.
    generated = [report_path, inventory_path]
    generated += [output / row["snapshot"] for row in inventory.values() if row.get("snapshot")]
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in generated:
            archive.write(path, arcname=output.name + "/" + path.relative_to(output).as_posix())
    return report_path, zip_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Original A/B/C/D run root")
    parser.add_argument("--diagnostic", default=DEFAULT_DIAGNOSTIC, help="Existing fixed-Train/Val diagnostic run")
    parser.add_argument("--output-parent", default=DEFAULT_OUTPUT, help="Parent of a new unique evidence directory")
    args = parser.parse_args(argv)
    report, archive = collect(args.root, args.diagnostic, args.output_parent)
    print("REPORT:", report)
    print("ZIP:", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
