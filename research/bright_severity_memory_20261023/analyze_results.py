"""Read-only analysis of two arms in ONE completed existing result.json.

No Torch, training, Test inference, route changes, or original-report writes.
Differences are net aggregate counts, not paired per-pixel correction counts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

CLASSES = ["background", "intact", "damaged", "destroyed"]


def checked_matrix(value):
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise ValueError("Expected four-by-four confusion matrix")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for row in value for x in row):
        raise ValueError("Confusion counts must be nonnegative JSON integers")
    if sum(map(sum, value)) == 0:
        raise ValueError("Confusion matrix is empty")
    return value


def ratio(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def metrics(cm):
    cm = checked_matrix(cm)
    gt = [sum(row) for row in cm]
    predicted = [sum(cm[r][c] for r in range(4)) for c in range(4)]
    iou = [ratio(cm[c][c], gt[c] + predicted[c] - cm[c][c]) for c in range(4)]
    tp = sum(cm[r][c] for r in range(1, 4) for c in range(1, 4))
    fn = sum(cm[r][0] for r in range(1, 4))
    fp = sum(cm[0][c] for c in range(1, 4))
    return dict(
        iou_pct=iou,
        miou4_pct=sum(v if v is not None else 0.0 for v in iou) / 4,
        foreground_miou3_pct=sum(v if v is not None else 0.0 for v in iou[1:]) / 3,
        averaging="fixed 4/3 classes; absent-class IoU contributes zero, matching the original protocol",
        precision_pct=[ratio(cm[c][c], predicted[c]) for c in range(4)],
        recall_pct=[ratio(cm[c][c], gt[c]) for c in range(4)],
        oa_pct=ratio(sum(cm[c][c] for c in range(4)), sum(gt)),
        building_iou_pct=ratio(tp, tp + fn + fp),
        building_precision_pct=ratio(tp, tp + fp),
        building_recall_pct=ratio(tp, tp + fn),
        building_false_negative=fn, building_false_positive=fp,
        row_normalized_pct=[[ratio(x, gt[r]) for x in cm[r]] for r in range(4)],
        ground_truth_counts=gt,
        building_definition="collapse foreground labels after four-class argmax, not a separate threshold",
    )


def analyze(report, before, after):
    if report.get("complete") is not True or report.get("final_integrity_pending") is True:
        raise ValueError("Refuse incomplete or integrity-pending report")
    if report.get("smoke") is True or "smoke" in str(report.get("phase", "")).lower():
        raise ValueError("Smoke is not formal evaluation evidence")
    if any(x in str(report.get("phase", "")).lower() for x in ("failed", "pending")):
        raise ValueError("Report phase is failed/pending")
    if report.get("class_names") != CLASSES:
        raise ValueError("Require explicit class_names in background/intact/damaged/destroyed order")
    split = report.get("split")
    if split not in ("val", "fixed_train") or report.get("test_accessed") is not False:
        raise ValueError("This version accepts Val/fixed-Train only; Test is unavailable")
    expected = {"val": 450}.get(split)
    if expected and report.get("image_count") != expected:
        raise ValueError("Partial split cannot be used as full evaluation")
    if before == after:
        raise ValueError("Choose two distinct arms")
    cms = {a: checked_matrix(report["metrics"][a]["confusion"]) for a in (before, after)}
    ms = {a: metrics(cms[a]) for a in (before, after)}
    if ms[before]["ground_truth_counts"] != ms[after]["ground_truth_counts"]:
        raise ValueError("Arms disagree on ground-truth class counts")
    scalar_keys = ("miou4_pct", "foreground_miou3_pct", "oa_pct", "building_iou_pct",
                   "building_precision_pct", "building_recall_pct")
    delta = {k.replace("_pct", "_pp"): (ms[after][k] - ms[before][k]
             if ms[after][k] is not None and ms[before][k] is not None else None) for k in scalar_keys}
    delta["iou_pp"] = [b - a if a is not None and b is not None else None
                       for a, b in zip(ms[before]["iou_pct"], ms[after]["iou_pct"])]
    return dict(
        format="BRIGHT_SEVERITY_CONFUSION_COMPARISON_V1", split=split,
        step=report.get("step"), image_count=report.get("image_count"),
        before=before, after=after, class_names=CLASSES, metrics=ms, differences=delta,
        net_confusion_count_changes=[[cms[after][r][c] - cms[before][r][c] for c in range(4)] for r in range(4)],
        caveats=["Report status checked; input images/checkpoints not independently reread here.",
                 "Net confusion differences are not paired pixel correction/harm counts.",
                 "Single-run differences do not establish multi-seed stability or significance.",
                 "OA is pixel accuracy; no assumed F_ch/F_cls definition."],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--before", default="grid_probq")
    ap.add_argument("--after", default="severity_memory")
    ap.add_argument("--output", type=Path, help="Optional NEW JSON file; never overwrite")
    args = ap.parse_args()
    raw = args.result.read_bytes()
    out = analyze(json.loads(raw), args.before, args.after)
    out["source_result"] = str(args.result.resolve())
    out["source_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
