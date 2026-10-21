"""Read existing BRIGHT evidence; never load checkpoints, import Torch or train.

All computed metrics are percentages. Confusion axes: GT row, prediction column.
Missing/invalid evidence remains explicit. This tool does NOT certify old runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

CLASSES = ["background", "intact", "damaged", "destroyed"]
DEFAULT_ROOT = "/root/shared-nvme/BRIGHT-checkpoints"
FOCUS = ("SRAC_JOINT_ANCHOR_EXTENDED_V2", "SRAC_JOINT_ANCHOR_FRESH_V3",
         "SRAC_SLOT_MEMORY_ABLATION_V1", "SRAC_SLOT_MEMORY_ABLATION_RESUME_V1",
         "SRAC_SOFT_MEMORY_Q_REBASED_V1", "SRAC_SOFT_MEMORY_Q_REBASED_TEST_V2")
STATUS_KEYS = ("format", "phase", "complete", "execution_complete", "training_complete",
               "full_validation_evidence", "full_test_evidence", "final_integrity_pending",
               "smoke", "test_accessed", "source_mode", "branch_step", "endpoint_step",
               "last_step", "last_complete_step", "restored_step", "schedule_horizon",
               "seed", "arm", "arms", "new_updates", "updates_per_arm", "error",
               "failure_phase", "release_sha256", "last_checkpoint", "evaluations")


def read_object(path):
    raw = path.read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("JSON exceeds 32 MiB; inspect separately")
    d = json.loads(raw)
    if not isinstance(d, dict):
        raise ValueError("Expected a JSON object")
    return d, hashlib.sha256(raw).hexdigest()


def percent_ratio(a, b):
    return 100.0 * a / b if b else None


def confusion_metrics(cm):
    if not isinstance(cm, list) or len(cm) != 4 or any(
        not isinstance(row, list) or len(row) != 4 for row in cm
    ):
        raise ValueError("confusion must be 4x4")
    if any(type(x) is not int or x < 0 for row in cm for x in row):
        raise ValueError("confusion counts must be nonnegative JSON integers")
    gt = [sum(row) for row in cm]
    pred = [sum(cm[i][j] for i in range(4)) for j in range(4)]
    n = sum(gt)
    if n == 0:
        raise ValueError("empty confusion matrix")
    tp = [cm[c][c] for c in range(4)]
    iou = [percent_ratio(tp[c], gt[c] + pred[c] - tp[c]) for c in range(4)]
    precision = [percent_ratio(tp[c], pred[c]) for c in range(4)]
    recall = [percent_ratio(tp[c], gt[c]) for c in range(4)]
    f1 = [percent_ratio(2 * tp[c], gt[c] + pred[c]) for c in range(4)]
    binary_tp = sum(cm[i][j] for i in (1, 2, 3) for j in (1, 2, 3))
    binary_fp, binary_fn = sum(cm[0][1:]), sum(cm[i][0] for i in (1, 2, 3))
    result = dict(
        iou_pct=iou, precision_pct=precision, recall_pct=recall, class_f1_pct=f1,
        miou4_pct=sum(x or 0 for x in iou) / 4,
        foreground_miou3_pct=sum(x or 0 for x in iou[1:]) / 3,
        pixel_oa_pct=percent_ratio(sum(tp), n),
        pixel_building_binary_f1_pct=percent_ratio(2 * binary_tp, 2 * binary_tp + binary_fp + binary_fn),
        target_pixels=gt, predicted_pixels=pred, valid_pixels=n,
        absent_class_policy="fixed 4/3 class means; absent-union IoU contributes zero",
        confusion=cm,
    )
    for src, dst in ((2, 0), (2, 1), (2, 3), (1, 2), (3, 2)):
        key = CLASSES[src] + "_to_" + CLASSES[dst]
        result[key + "_pixels"] = cm[src][dst]
        result[key + "_pct_of_gt_class"] = percent_ratio(cm[src][dst], gt[src])
    return result


def metric_values(m, classes, issues):
    if classes is not None and [str(x).lower() for x in classes] != CLASSES:
        raise ValueError("Unsupported class order; refusing to relabel columns")
    if classes is None:
        issues.append("class_names absent: legacy BG/Intact/Damaged/Destroyed order assumed; confirm before reporting")
    if "confusion" in m:
        out = confusion_metrics(m["confusion"])
        if "iou" in m:
            supplied = m["iou"]
            if not isinstance(supplied, list) or len(supplied) != 4:
                raise ValueError("saved IoU must have four entries")
            for a, b in zip(supplied, out["iou_pct"]):
                if a is None and b is None:
                    continue
                if a is None or b is None or not isinstance(a, (int, float)) or not math.isfinite(a) or abs(100 * a - b) > 1e-4:
                    issues.append("saved IoU differs from confusion-derived IoU; inspect protocol/units")
                    break
        out["metric_source"] = "recomputed from saved confusion (GT rows)"
        return out
    iou = m.get("iou")
    if not isinstance(iou, list) or len(iou) != 4 or any(
        x is not None and (type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1)
        for x in iou
    ):
        raise ValueError("No usable confusion or four-entry fractional IoU; no automatic unit guessing")
    issues.append("confusion missing: cannot recover OA/F1 or error destinations from IoU alone")
    return dict(iou_pct=[None if x is None else 100 * x for x in iou],
                miou4_pct=25 * sum(x or 0 for x in iou),
                foreground_miou3_pct=100 / 3 * sum(x or 0 for x in iou[1:]),
                metric_source="saved fractional IoU; not independently verified",
                absent_class_policy="fixed 4/3 class means; absent IoU contributes zero")


def split_identity(d, path):
    if d.get("split") is not None:
        return str(d["split"]), "declared"
    name = path.parent.name.lower()
    if name.startswith("val_") or name in ("initial_val", "val_baseline"):
        return "val", "inferred from folder; confirm"
    if name.startswith("fixed_train"):
        return "fixed_train", "inferred from folder; confirm"
    # In particular do not infer Test from test_accessed alone.
    return "unknown", "missing"


def metric_scopes(d):
    """Known schemas only; never turn GT-filtered audit subsets into global scores."""
    metrics = d.get("metrics")
    if isinstance(metrics, dict):
        scopes = [("all", metrics)]
        if isinstance(d.get("per_event"), dict):
            scopes += list(d["per_event"].items())
        return "evaluation", "metrics", scopes
    pixels = d.get("pixels")
    overall = pixels.get("all_labeled") if isinstance(pixels, dict) else None
    metrics = overall.get("metrics") if isinstance(overall, dict) else None
    if isinstance(metrics, dict):
        scopes = [("all", metrics)]
        events = d.get("per_event", {})
        if not isinstance(events, dict):
            raise ValueError("audit per_event must be an object")
        for event, record in events.items():
            try:
                event_metrics = record["pixels"]["all_labeled"]["metrics"]
            except (KeyError, TypeError):
                raise ValueError("audit event lacks all_labeled metrics: " + str(event))
            if not isinstance(event_metrics, dict):
                raise ValueError("audit event metrics must be an object: " + str(event))
            scopes.append((event, event_metrics))
        return "audit_replay", "pixels.all_labeled.metrics", scopes
    raise ValueError("result.json has no supported metrics object")


def scan_files(root, max_depth):
    skipped = {".git", "__pycache__", "logs", "checkpoints", "predictions", "pool_debug"}
    for directory, dirs, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        depth = len(directory.relative_to(root).parts)
        dirs[:] = sorted(x for x in dirs if x not in skipped and not (directory / x).is_symlink()) if depth < max_depth else []
        for name in ("summary.json", "result.json"):
            p = directory / name
            if name in names and not p.is_symlink():
                yield p


def nearby_metadata(path, boundary, cache):
    documents = []
    seen = set()
    for parent in [path.parent, *path.parent.parents]:
        if not parent.is_relative_to(boundary):
            break
        for rel in ("manifest.json", "train/manifest.json", "config.json", "source.json"):
            p = parent / rel
            if p in seen or not p.is_file() or p.is_symlink():
                continue
            seen.add(p)
            if p not in cache:
                try:
                    d, sha = read_object(p)
                    cache[p] = dict(path=str(p), sha256=sha, document=d)
                except (OSError, ValueError) as e:
                    cache[p] = dict(path=str(p), error=str(e))
            documents.append(cache[p])
    fields = {}
    for item in documents:
        d = item.get("document", {})
        for config in (d.get("runtime_config", {}), d.get("arguments", {}), d):
            if not isinstance(config, dict):
                continue
            for key in ("seed", "lr", "weight_decay", "steps", "topk", "key_name", "chunk_size",
                        "source_mode", "source_checkpoint", "source_sha256", "release_sha256",
                        "weight", "lr_multiplier", "repo", "protocol", "bank_root", "values"):
                if key in config and key not in fields:
                    fields[key] = config[key]
    return fields, [dict(path=x["path"], **{k: x[k] for k in ("sha256", "error") if k in x}) for x in documents]


def collect(roots, max_depth=7):
    rows, statuses, errors, metadata_cache = [], [], [], {}
    visited = set()
    for root in roots:
        if not root.is_dir():
            errors.append(dict(path=str(root), error="directory missing; not treated as an empty completed run"))
            continue
        for p in scan_files(root, max_depth):
            if p.resolve() in visited:
                continue
            visited.add(p.resolve())
            try:
                d, sha = read_object(p)
                meta, refs = nearby_metadata(p, root, metadata_cache)
                if p.name == "summary.json":
                    statuses.append(dict(path=str(p), sha256=sha,
                                         fields={k: d[k] for k in STATUS_KEYS if k in d}, metadata=meta))
                    continue
                report_kind, metrics_location, scopes = metric_scopes(d)
                split, split_source = split_identity(d, p)
                for scope, branches in scopes:
                    if not isinstance(branches, dict):
                        continue
                    for branch, m in branches.items():
                        issues = []
                        try:
                            if not isinstance(m, dict):
                                raise ValueError("metric branch is not an object")
                            values = metric_values(m, m.get("class_names", d.get("class_names")), issues)
                            if d.get("complete") is not True:
                                issues.append("report complete is not true")
                            if d.get("final_integrity_pending") is True:
                                issues.append("report integrity pending")
                            if d.get("smoke") is True:
                                issues.append("smoke report; exclude from formal tables")
                            if split_source != "declared":
                                issues.append("split " + split_source)
                            image_count = d.get("image_count") if scope == "all" else d.get("per_event_image_count", {}).get(scope)
                            if scope == "all" and split in ("val", "test") and image_count != {"val": 450, "test": 878}[split]:
                                issues.append("image_count differs from expected Val450/Test878; verify split protocol")
                            if scope == "all" and split == "val" and d.get("full_validation_evidence") is not True:
                                issues.append("full_validation_evidence is not true; inspect enclosing summary")
                            if report_kind == "audit_replay":
                                issues.append("audit replay: diagnostic evidence, not an additional trained ablation")
                            rows.append(dict(path=str(p), result_sha256=sha, format=d.get("format"),
                                report_kind=report_kind, metrics_location=metrics_location,
                                split=split, split_source=split_source, scope=scope, branch=branch,
                                step=d.get("step", d.get("checkpoint_step")), image_count=image_count,
                                complete=d.get("complete"), smoke=d.get("smoke"),
                                phase=d.get("phase"), final_integrity_pending=d.get("final_integrity_pending"),
                                names_sha256=d.get("names_sha256"), checkpoint_sha256=d.get("checkpoint_sha256"),
                                metadata=meta, metadata_files=refs, issues=issues, **values))
                        except (ValueError, TypeError, KeyError) as e:
                            errors.append(dict(path=str(p), scope=scope, branch=branch, error=str(e)))
            except (OSError, ValueError, TypeError) as e:
                errors.append(dict(path=str(p), error=str(e)))
    return dict(format="BRIGHT_READ_ONLY_EXPERIMENT_REVIEW_V1", roots=[str(r) for r in roots],
                training_started=False, checkpoints_loaded=False, original_files_modified=False,
                results_promoted=False, statuses=statuses, metrics=rows, read_errors=errors,
                notes=["This is an index of saved evidence, not a new evaluation or integrity certificate.",
                       "Do not compare different split/source/checkpoint identities merely because names match.",
                       "Class means use fixed classes, not mean of per-image or per-event IoUs.",
                       "F_ch/F_cls are not assigned without their benchmark definitions.",
                       "Missing metadata is unknown, never zero/false by default."])


def csv_text(rows):
    columns = ["path", "report_kind", "metrics_location", "split", "scope", "branch", "step", "seed", "image_count", "complete",
               "miou4_pct", "foreground_miou3_pct", *[c + "_iou_pct" for c in CLASSES],
               "pixel_oa_pct", "pixel_building_binary_f1_pct", "damaged_precision_pct", "damaged_recall_pct",
               "damaged_to_background_pct_of_gt_class", "damaged_to_intact_pct_of_gt_class",
               "damaged_to_destroyed_pct_of_gt_class", "intact_to_damaged_pct_of_gt_class", "issues"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for r in rows:
        row = {k: r.get(k) for k in columns}
        row["seed"] = r["metadata"].get("seed")
        row.update({c + "_iou_pct": v for c, v in zip(CLASSES, r["iou_pct"])})
        for key in ("precision", "recall"):
            row["damaged_" + key + "_pct"] = r.get(key + "_pct", [None] * 4)[2]
        row["issues"] = " | ".join(r["issues"])
        writer.writerow(row)
    return stream.getvalue()


def markdown(report):
    lines = ["# BRIGHT 已有实验只读清单", "", "本文件不表示重新验证成功，不会启动训练或修改原结果。",
             "数值单位为 %；差值应表述为百分点。详细结果和逐事件混淆见 metrics.csv / inventory.json。", "",
             "## 运行状态", "", "| summary 路径 | phase | complete | 最后步数 | error |", "|---|---|---|---|---|"]
    def clean(x):
        return str(x if x is not None else "unknown").replace("|", "/").replace("\n", " ")
    for s in report["statuses"]:
        d = s["fields"]
        lines.append("| " + " | ".join(clean(x) for x in (s["path"], d.get("phase"), d.get("complete"),
                     d.get("last_complete_step", d.get("last_step", d.get("endpoint_step"))), d.get("error"))) + " |")
    lines += ["", "## 全集指标（未自动选择最佳 Test）", "",
              "| result 路径 | split | step | 分支 | 四类 mIoU | 前景 mIoU | Damaged IoU | 待核对项数 |",
              "|---|---|---|---|---|---|---|---|"]
    def fmt(x):
        return "unknown" if x is None else f"{x:.4f}"
    for r in report["metrics"]:
        if r["scope"] == "all":
            lines.append("| " + " | ".join(clean(x) for x in (r["path"], r["split"], r["step"], r["branch"],
                fmt(r["miou4_pct"]), fmt(r["foreground_miou3_pct"]), fmt(r["iou_pct"][2]), len(r["issues"]))) + " |")
    lines += ["", "## 无法读取或无法识别的证据", ""]
    lines += ["- " + clean(e) for e in report["read_errors"]] or ["- 无读取错误。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", help="Specific existing run/family; repeatable. Default: known experiment families.")
    parser.add_argument("--output", help="NEW review directory outside input roots; existing directory refused.")
    parser.add_argument("--max-depth", type=int, default=7)
    args = parser.parse_args(argv)
    if args.max_depth < 0:
        parser.error("max-depth must be nonnegative")
    roots = [Path(p).expanduser().resolve() for p in args.root] if args.root else [Path(DEFAULT_ROOT) / p for p in FOCUS]
    if args.output:
        output = Path(args.output).expanduser().absolute()
        resolved = output.resolve()
        if any(resolved.is_relative_to(r) or r.is_relative_to(resolved) for r in roots):
            parser.error("output must be separate from all input roots")
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="bright_experiment_review_"))
    print("READ ONLY: indexing existing summaries and metrics; no Torch/GPU/training", flush=True)
    report = collect(roots, args.max_depth)
    (output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "metrics.csv").write_text(csv_text(report["metrics"]), encoding="utf-8-sig")
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(f"OUTPUT={output}\nSUMMARIES={len(report['statuses'])} METRIC_ROWS={len(report['metrics'])} READ_ERRORS={len(report['read_errors'])}", flush=True)
    print("Completed inventory only; unknown/incomplete runs have NOT been promoted.", flush=True)
    return 0 if report["metrics"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
