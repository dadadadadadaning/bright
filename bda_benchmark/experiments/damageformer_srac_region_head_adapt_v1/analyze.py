"""CPU-only breakdown of an existing, verified Val450 region-path journal.

No checkpoints, images, Q, training cache, GPU, or Test dataset are opened.
Counts come exclusively from already frozen predictions and saved Val labels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import (
    ARMS, checked_protocol, source_hash,
)
from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.report import checked_result


FORMAT = "SRAC_REGION_HEAD_ADAPT_ANALYSIS_V1"
LABELS = ("Background", "Intact", "Damaged", "Destroyed")
PAIR = "region_mean__region"
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def artifact_path(root, relative):
    """Accept only confined, canonical relative prediction NPZ names."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe prediction artifact path")
    rel = PurePosixPath(relative)
    if (rel.is_absolute() or ".." in rel.parts or rel.as_posix() != relative
            or len(rel.parts) != 2 or rel.parts[0] != "predictions"
            or rel.suffix != ".npz"):
        raise ValueError("Unsafe prediction artifact path")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if root not in result.parents or not result.is_file():
        raise ValueError("Missing or escaping prediction artifact: " + relative)
    return result


def validated_maps(target, reference, candidate):
    values = [np.asarray(v) for v in (target, reference, candidate)]
    if values[0].ndim == 3 and values[0].shape[0] == 1:
        values = [v[0] if v.ndim == 3 and v.shape[0] == 1 else v for v in values]
    y, a, b = values
    if y.ndim != 2 or not y.size or any(v.shape != y.shape for v in values):
        raise ValueError("Expected matching nonempty label maps")
    if any(v.dtype.kind not in "iu" for v in values):
        raise ValueError("Label maps must have integer dtype")
    if not np.all(((y >= 0) & (y < 4)) | (y == 255)):
        raise ValueError("GT must use classes 0..3 or ignore 255")
    if any(not np.all((v >= 0) & (v < 4)) for v in (a, b)):
        raise ValueError("Predicted class outside 0..3")
    return y.astype(np.int64), a.astype(np.int64), b.astype(np.int64)


def class_boundary(target):
    """Mark BOTH valid endpoints of a four-neighbor GT class change.

    One saved-output-grid-pixel adjacency, no dilation, no image-edge padding.
    Ignore 255 never creates a boundary and never enters either stratum.
    This is a four-class semantic boundary, not a building-instance boundary.
    """
    y = np.asarray(target)
    if y.ndim != 2 or y.dtype.kind not in "iu":
        raise ValueError("Boundary requires a 2-D integer GT map")
    valid = (y >= 0) & (y < 4)
    edge = np.zeros(y.shape, dtype=bool)
    vertical = valid[:-1] & valid[1:] & (y[:-1] != y[1:])
    horizontal = valid[:, :-1] & valid[:, 1:] & (y[:, :-1] != y[:, 1:])
    edge[:-1] |= vertical
    edge[1:] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    return edge


def joint_counts(target, reference, candidate, mask=None):
    y, a, b = validated_maps(target, reference, candidate)
    valid = y < 4
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != y.shape or mask.dtype.kind != "b":
            raise ValueError("Stratum mask must be bool with the GT shape")
        valid &= mask
    ids = y[valid] * 16 + a[valid] * 4 + b[valid]
    return np.bincount(ids, minlength=64).reshape(4, 4, 4)


def ratio(num, denominator):
    return float(num / denominator) if denominator else None


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp, rows, columns = cm.diagonal(), cm.sum(1), cm.sum(0)
    union = rows + columns - tp
    def ratios(num, den):
        return [ratio(int(n), int(d)) for n, d in zip(num, den)]
    iou = ratios(tp, union)
    present = [v for v in iou[1:] if v is not None]
    return dict(count=int(cm.sum()), confusion=cm.tolist(),
                valid_miou=float((tp / np.maximum(union, 1))[1:].mean()) if cm.sum() else None,
                present_foreground_miou=sum(present)/len(present) if present else None,
                iou=iou, precision=ratios(tp, columns), recall=ratios(tp, rows),
                f1=ratios(2*tp, rows+columns), support=rows.tolist(),
                background_to_building=int(cm[0, 1:].sum()),
                building_to_background=int(cm[1:, 0].sum()))


def transition_summary(joint):
    joint = np.asarray(joint)
    if joint.shape != (4, 4, 4) or joint.dtype.kind not in "iu" or np.any(joint < 0):
        raise ValueError("Need nonnegative integer [GT, reference, candidate] counts")
    classes = {}
    for c, name in enumerate(LABELS):
        n, tp0, tp1 = int(joint[c].sum()), int(joint[c, c].sum()), int(joint[c, :, c].sum())
        kept = int(joint[c, c, c])
        harm, rescue = tp0-kept, tp1-kept
        fp0, fp1 = int(joint[:, c].sum())-tp0, int(joint[:, :, c].sum())-tp1
        classes[name] = dict(gt_count=n, reference_tp=tp0, candidate_tp=tp1,
                             harm=harm, rescue=rescue, net_tp=rescue-harm,
                             harm_rate=ratio(harm, tp0), rescue_rate=ratio(rescue, n-tp0),
                             reference_fp=fp0, candidate_fp=fp1, delta_fp=fp1-fp0)
    return dict(joint_gt_reference_candidate=joint.tolist(), per_class=classes)


def breakdown(joint):
    row = transition_summary(joint)
    j = np.asarray(joint, dtype=np.int64)
    before, after = metrics(j.sum(2)), metrics(j.sum(1))
    n = int(j.sum())
    row.update(valid_pixels=n, metrics={"region_mean": before, "region": after},
               reference_errors=n-int(np.trace(j.sum(2))),
               candidate_errors=n-int(np.trace(j.sum(1))))
    row["reference_error_rate"] = ratio(row["reference_errors"], n)
    row["candidate_error_rate"] = ratio(row["candidate_errors"], n)
    d = row["per_class"]["Damaged"]
    row["damaged_harm_destination"] = {
        LABELS[c]: dict(count=int(j[2, 2, c]),
                        rate_per_reference_correct_damaged=ratio(int(j[2, 2, c]), d["reference_tp"]))
        for c in (0, 1, 3)}
    row["damaged_rescue_previous_prediction"] = {
        LABELS[c]: dict(count=int(j[2, c, 2]),
                        rate_per_reference_wrong_damaged=ratio(int(j[2, c, 2]), d["gt_count"]-d["reference_tp"]))
        for c in (0, 1, 3)}
    false_positive = {}
    for c in (0, 1, 3):
        added = int(j[c, :, 2].sum()-j[c, 2, 2])
        removed = int(j[c, 2, :].sum()-j[c, 2, 2])
        gt_count = int(j[c].sum())
        eligible_add = gt_count-int(j[c, 2, :].sum())
        eligible_remove = int(j[c, 2, :].sum())
        false_positive[LABELS[c]] = dict(
            gt_count=gt_count, added=added, removed=removed, net=added-removed,
            addition_eligible_reference_not_damaged=eligible_add,
            removal_eligible_reference_damaged=eligible_remove,
            addition_rate=ratio(added, eligible_add), removal_rate=ratio(removed, eligible_remove),
            net_rate_per_gt_class=ratio(added-removed, gt_count))
    row["damaged_false_positive_by_true_class"] = false_positive
    row["miou_delta_percentage_points"] = (
        100.0*(after["valid_miou"]-before["valid_miou"])
        if before["valid_miou"] is not None else None)
    return row


def assert_equal(actual, expected, location):
    """Integer counts must agree exactly; only float arithmetic permits 1e-12."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual)-set(expected):
            raise ValueError("Missing count/metric fields at " + location)
        for key, value in actual.items():
            assert_equal(value, expected[key], location + "/" + key)
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError("Array length mismatch at " + location)
        for index, value in enumerate(actual):
            assert_equal(value, expected[index], location + "/" + str(index))
    elif isinstance(actual, float):
        if (not isinstance(expected, (int, float)) or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Metric differs from frozen evidence at " + location)
    elif actual != expected:
        raise ValueError("Count differs from frozen evidence at " + location)


def check_protocol_reference(result, override=None):
    expected_digest = result.get("protocol_sha256")
    if not isinstance(expected_digest, str) or SHA.fullmatch(expected_digest) is None:
        raise ValueError("Missing/invalid source E0 SHA256")
    saved = result.get("protocol_path")
    if not isinstance(saved, str) or not saved:
        raise ValueError("Missing source E0 protocol path")
    path = Path(override if override else saved)
    path = path / "protocol.json" if path.is_dir() else path
    if not path.is_file():
        if override:
            raise ValueError("Explicit source E0 protocol does not exist")
        return None, dict(available=False, selected_images_order_verified=False,
                          expected_sha256=expected_digest, saved_path=saved,
                          reason="Saved protocol file unavailable; journal order/count and saved formal evidence checked")
    if digest(path) != expected_digest:
        raise ValueError("Source E0 protocol SHA256 differs")
    path, protocol = checked_protocol(path)
    return protocol["selected_images"], dict(
        available=True, selected_images_order_verified=True,
        expected_sha256=expected_digest, actual_path=str(path.resolve()))


def analyze(input_dir, output_dir, *, protocol=None, boundary=True):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("Analysis output must be outside the read-only source directory")
    if output.exists():
        raise ValueError("Analysis output must be a fresh directory")
    source_paths = {name: root / name for name in ("summary.json", "result.json", "images.jsonl")}
    initial_hashes = {name: digest(path) for name, path in source_paths.items()}
    result = checked_result(root)
    if result.get("profile") != "legacy" or result.get("pool_mode") != "native":
        raise ValueError("Only formal legacy/native Val450 evidence is accepted")
    names, protocol_report = check_protocol_reference(result, protocol)
    rows = [json.loads(line) for line in source_paths["images.jsonl"].read_text(encoding="utf-8").splitlines()]
    if len(rows) != 450 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Journal must contain exactly 450 image objects")
    images = [row.get("image") for row in rows]
    if (any(not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None for name in images)
            or len(set(images)) != 450
            or [row.get("index") for row in rows] != list(range(450))):
        raise ValueError("Journal image names/indices are invalid, duplicated or out of order")
    if names is not None and images != names:
        raise ValueError("Journal image order differs from the source E0 Val list")
    if any(row.get("event") != re.sub(r"_\d+$", "", row["image"]) for row in rows):
        raise ValueError("Journal event does not match its image identifier")
    cms = {arm: np.zeros((4, 4), dtype=np.int64) for arm in ARMS}
    per_event, event_joints = {}, {}
    joint = np.zeros((4, 4, 4), dtype=np.int64)
    strata = {key: np.zeros_like(joint) for key in ("boundary", "interior")} if boundary else {}
    event_strata = {}
    artifact_hashes = {}
    ignored_pixels = 0
    for index, row in enumerate(rows):
        path = artifact_path(root, row.get("prediction_file"))
        if path in artifact_hashes:
            raise ValueError("Journal reuses a prediction artifact")
        expected = row.get("prediction_sha256")
        if not isinstance(expected, str) or SHA.fullmatch(expected) is None:
            raise ValueError("Missing prediction artifact SHA256")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Prediction artifact checksum mismatch: " + str(path))
        artifact_hashes[path] = expected
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            needed = ["target", *("prediction_" + arm for arm in ARMS)]
            if any(name not in archive.files for name in needed):
                raise ValueError("Prediction artifact misses a target or prediction map")
            raw = {name: archive[name] for name in needed}
        if raw["target"].ndim != 3 or raw["target"].shape[0] != 1:
            raise ValueError("Saved target must be [1,H,W]")
        y, a, b = validated_maps(raw["target"], raw["prediction_region_mean"], raw["prediction_region"])
        valid = y < 4
        ignored_pixels += int((~valid).sum())
        event_name = row["event"]
        event_cms = per_event.setdefault(event_name, {arm: np.zeros((4, 4), dtype=np.int64) for arm in ARMS})
        if not isinstance(row.get("confusion"), dict) or set(row["confusion"]) != set(ARMS):
            raise ValueError("Journal image confusion table is incomplete")
        for arm in ARMS:
            prediction = raw["prediction_" + arm]
            if prediction.shape != raw["target"].shape:
                raise ValueError("Saved prediction shape differs from target: " + arm)
            if prediction.dtype.kind not in "iu" or not np.all((prediction >= 0) & (prediction < 4)):
                raise ValueError("Invalid predicted class/dtype: " + arm)
            pred = prediction[0].astype(np.int64)
            cm = np.bincount(y[valid]*4+pred[valid], minlength=16).reshape(4, 4)
            assert_equal(cm.tolist(), row["confusion"][arm], f"image/{index}/{arm}")
            cms[arm] += cm
            event_cms[arm] += cm
        image_joint = joint_counts(y, a, b)
        joint += image_joint
        event_joints.setdefault(event_name, np.zeros_like(joint))[:] += image_joint
        if boundary:
            edge = class_boundary(y)
            masks = {"boundary": edge, "interior": valid & ~edge}
            event_s = event_strata.setdefault(event_name, {key: np.zeros_like(joint) for key in strata})
            for key, mask in masks.items():
                counts = joint_counts(y, a, b, mask)
                strata[key] += counts
                event_s[key] += counts
        if (index+1) % 25 == 0:
            print(f"Read-only Val maps: {index+1}/450", flush=True)
    for arm in ARMS:
        assert_equal(metrics(cms[arm]), result["metrics"][arm], "metrics/" + arm)
    if set(per_event) != set(result.get("per_event", {})):
        raise ValueError("Event set differs from saved evidence")
    for event_name, event_cms in per_event.items():
        for arm in ARMS:
            assert_equal(metrics(event_cms[arm]), result["per_event"][event_name][arm],
                         "event/" + event_name + "/" + arm)
    assert_equal(transition_summary(joint), result["transitions"][PAIR], "transitions/" + PAIR)
    if boundary and not np.array_equal(sum(strata.values()), joint):
        raise ValueError("Boundary/interior do not partition all valid pixels")
    for name, path in source_paths.items():
        if digest(path) != initial_hashes[name]:
            raise ValueError("Source metadata changed while analyzing: " + name)
    for path, expected in artifact_hashes.items():
        if digest(path) != expected:
            raise ValueError("Prediction file changed while analyzing: " + str(path))
    if source_hash() != result["source_sha256"]:
        raise ValueError("Original evaluator code changed during analysis")
    if protocol_report["available"] and digest(protocol_report["actual_path"]) != result["protocol_sha256"]:
        raise ValueError("E0 protocol changed during analysis")
    report = dict(
        format=FORMAT, complete=True, full_validation_evidence=True, image_count=450,
        source=str(root), source_metadata_sha256=initial_hashes,
        source_evaluator_sha256=result["source_sha256"], analyzer_sha256=digest(__file__), protocol=protocol_report,
        verified_prediction_files=450, source_files_unchanged=True,
        original_reproduction_tolerance=1e-5, original_reproduction_passed=True,
        native_a1_replay_ok=True, profile="legacy", pool_mode="native",
        pair=PAIR, class_order=list(LABELS), units="pixel counts; rates are fractions; IoU changes are percentage points",
        training=False, inference=False, test_accessed=False, ignored_pixels=ignored_pixels,
        overall=breakdown(joint), per_event={name: breakdown(value) for name, value in sorted(event_joints.items())},
        boundary_definition=("Both valid endpoints of a GT four-class horizontal/vertical change; one saved-output-grid-pixel adjacency; no dilation; ignore255 excluded; image border not automatically an edge" if boundary else None),
        boundary_strata={name: breakdown(value) for name, value in strata.items()},
        per_event_boundary_strata={name: {key: breakdown(value) for key, value in strata_.items()}
                                   for name, strata_ in sorted(event_strata.items())},
        limitations=["Counts describe saved frozen predictions, not causal component effects.",
                     "No Q weights were saved here: this analysis cannot identify Q mixing or mixed-region ownership.",
                     "Four-class semantic edges do not identify building instances or separate adjacent same-class buildings.",
                     "Boundary/interior rates use each stratum's own explicit eligible pixel denominator; pixel counts are not building counts."],
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    markdown = format_report(report)
    (output / "report.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return report


def format_report(report):
    overall = report["overall"]
    d = overall["per_class"]["Damaged"]
    def percent(value):
        return "NA" if value is None else f"{value*100:.4f}%"
    lines = ["Val450：已逐图校验冻结预测；region_mean → region。", "",
             f"前景 mIoU 变化：{overall['miou_delta_percentage_points']:+.4f} 个百分点。",
             f"Damaged 救回 {d['rescue']}，误改 {d['harm']}，净正确变化 {d['net_tp']:+d}，假阳性净变化 {d['delta_fp']:+d}。", "",
             "| Damaged 变化 | 类别 | 像素数 | 固定原预测分母下的比例 |",
             "|---|---|---:|---:|"]
    for name, value in overall["damaged_harm_destination"].items():
        lines.append(f"| 原本正确，改错到 | {name} | {value['count']} | {percent(value['rate_per_reference_correct_damaged'])} |")
    for name, value in overall["damaged_rescue_previous_prediction"].items():
        lines.append(f"| 原本错分，从该类救回 | {name} | {value['count']} | {percent(value['rate_per_reference_wrong_damaged'])} |")
    lines += ["", "误改比例分母＝region_mean 判对的真实 Damaged；救回比例分母＝region_mean 判错的真实 Damaged。", "",
              "| Damaged 假阳性的真实类别 | 新增 | 消除 | 净变化 |", "|---|---:|---:|---:|"]
    for name, row in overall["damaged_false_positive_by_true_class"].items():
        lines.append(f"| {name} | {row['added']} | {row['removed']} | {row['net']:+d} |")
    lines += ["", "| 事件 | mIoU 变化（百分点） | D 救回 | D 误改 | D 假阳性净变化 |", "|---|---:|---:|---:|---:|"]
    for name, row in sorted(report["per_event"].items(), key=lambda item: (item[1]["miou_delta_percentage_points"] or 0, item[0])):
        cls = row["per_class"]["Damaged"]
        delta = row["miou_delta_percentage_points"]
        text_delta = "NA" if delta is None else f"{delta:+.4f}"
        lines.append(f"| {name} | {text_delta} | {cls['rescue']} | {cls['harm']} | {cls['delta_fp']:+d} |")
    if report["boundary_strata"]:
        lines += ["", "| GT 位置 | 有效像素 | 原错误率 | 新错误率 | D 救回率 | D 误改率 |", "|---|---:|---:|---:|---:|---:|"]
        for name, row in report["boundary_strata"].items():
            cls = row["per_class"]["Damaged"]
            lines.append(f"| {name} | {row['valid_pixels']} | {percent(row['reference_error_rate'])} | {percent(row['candidate_error_rate'])} | {percent(cls['rescue_rate'])} | {percent(cls['harm_rate'])} |")
        lines += ["", "边界＝保存标签网格上，水平／垂直相邻且有效的不同类别像素对的两端；不膨胀、不把图像外沿或 ignore255 当类别边界。",
                  "各层错误率分母为该层有效像素；D 救回／误改率分母分别为该层原先判错／判对的真实 D。分母与完整计数保存在 report.json。"]
    if not report["protocol"]["available"]:
        lines += ["", "原 E0 protocol.json 不在记录路径：已检查日志编号、唯一性、摘要和预测哈希，但无法独立对照 E0 图像顺序。可用 --protocol 提供 SHA256 相同的原文件。"]
    lines += ["", "统计单位是像素。该文件没有 Q 或建筑实例编号，不能据此判断混合超像素占比、跨建筑归属或某个模块的因果作用。没有运行训练、推理或读取 Test。"]
    return "\n".join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Completed SRAC_REGION_VAL_FORMAL_V1 Val450 directory")
    p.add_argument("--output", required=True, help="Fresh sibling/output directory; source evidence remains read-only")
    p.add_argument("--protocol", help="Optional relocated original E0 protocol.json; exact SHA256 must match")
    p.add_argument("--no-boundary", action="store_true", help="Skip semantic boundary/interior strata")
    return p


def main():
    args = parser().parse_args()
    analyze(args.input, args.output, protocol=args.protocol, boundary=not args.no_boundary)


if __name__ == "__main__":
    main()
