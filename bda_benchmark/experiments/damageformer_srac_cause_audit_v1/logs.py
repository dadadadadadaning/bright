"""Read existing parent validation logs without loading data or checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

LABELS = ("Background", "Intact", "Damaged", "Destroyed")
FRONT_FORMAT = "SRAC_TRANSFORMER_FRONT_V1"
LOG_LIMITATION = (
    "These are logged class confusion counts (rows=GT, columns=prediction), "
    "not paired-pixel harm or rescue. Marginal confusion matrices cannot identify "
    "which pixels changed from an A1-correct prediction to a front error."
)


def _incomplete_json_suffix(error, raw):
    if isinstance(error, UnicodeDecodeError):
        return error.reason == "unexpected end of data" and error.end == len(raw)
    text = error.doc.rstrip()
    if error.pos >= len(text) or error.msg.startswith("Unterminated string"):
        return True
    tail = text[error.pos:]
    # Valid JSON tokens can be split at any byte by an active append writer.
    if error.msg == "Expecting value" and tail in ("t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul", "-"):
        return True
    if error.msg == "Expecting ',' delimiter" and re.fullmatch(r"(?:\.|[eE][+-]?)", tail):
        return True
    # A Unicode string escape may be truncated inside its four hex digits.
    if error.msg.startswith("Invalid \\uXXXX escape") and re.fullmatch(r"u[0-9a-fA-F]{0,4}", tail):
        return True
    return False


def read_validation_jsonl(path):
    """Read one point-in-time byte view; tolerate only an incomplete final line.

    A malformed newline-terminated record is never silently discarded. A final
    complete JSON object without its newline is accepted.
    """
    content = Path(path).read_bytes()
    lines = content.splitlines(keepends=True)
    rows, ignored = [], False
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index == len(lines) - 1 and not raw.endswith((b"\n", b"\r")) and _incomplete_json_suffix(error, raw):
                ignored = True
                break
            raise ValueError(f"Malformed validation JSON at line {index + 1}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Validation line {index + 1} must contain an object")
        rows.append(row)
    return rows, ignored


def class_metrics(logged):
    """Recompute all class metrics directly from the logged raw confusion."""
    if not isinstance(logged, dict):
        raise ValueError("Missing logged metrics object")
    confusion = logged.get("confusion")
    if not isinstance(confusion, list) or len(confusion) != 4:
        raise ValueError("Expected a four-class confusion matrix")
    for row in confusion:
        if not isinstance(row, list) or len(row) != 4 or any(type(x) is not int or x < 0 for x in row):
            raise ValueError("Confusion entries must be nonnegative integer raw counts")
    support = [sum(row) for row in confusion]
    predicted = [sum(confusion[i][j] for i in range(4)) for j in range(4)]
    count = sum(support)
    if not count:
        raise ValueError("Empty validation confusion")
    if logged.get("count", count) != count:
        raise ValueError("Logged count disagrees with confusion counts")
    classes = {}
    for index, name in enumerate(LABELS):
        tp = confusion[index][index]
        union = support[index] + predicted[index] - tp
        denom_f1 = support[index] + predicted[index]
        classes[name] = dict(
            iou=tp / union if union else None,
            precision=tp / predicted[index] if predicted[index] else None,
            recall=tp / support[index] if support[index] else None,
            f1=2 * tp / denom_f1 if denom_f1 else None,
            support=support[index],
            predicted=predicted[index],
            true_positive=tp,
            false_positive=predicted[index] - tp,
            false_negative=support[index] - tp,
        )
    miou = sum(classes[name]["iou"] or 0.0 for name in LABELS[1:]) / 3
    logged_miou = logged.get("valid_miou")
    if not isinstance(logged_miou, (int, float)) or not math.isfinite(logged_miou):
        raise ValueError("Missing / non-finite logged valid_miou")
    if not math.isclose(logged_miou, miou, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Logged valid_miou disagrees with raw confusion")
    return dict(
        valid_miou=miou,
        count=count,
        confusion=confusion,
        classes=classes,
        damaged_to_intact=dict(
            count=confusion[2][1],
            damaged_support=support[2],
            rate=confusion[2][1] / support[2] if support[2] else None,
        ),
    )


def summarize_logs(front_run, allow_smoke=False):
    root = Path(front_run).resolve()
    if not root.is_dir():
        raise ValueError("--front_run must be an actual existing front run directory")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FRONT_FORMAT or manifest.get("test_accessed") is not False:
        raise ValueError("Expected a parent front manifest declaring no Test access")
    smoke = manifest.get("smoke")
    if type(smoke) is not bool:
        raise ValueError("Manifest must explicitly identify formal versus smoke run")
    if smoke and not allow_smoke:
        raise ValueError("Smoke logs are not Val450 evidence; pass --allow_smoke to label them explicitly")
    arguments = manifest.get("arguments", {})
    if not smoke and arguments.get("steps") != 50000:
        raise ValueError("Formal front manifest must declare 50000 optimizer steps")
    val_list = str(arguments.get("val_list", "")).replace("\\", "/").rsplit("/", 1)[-1]
    if val_list != "val_set.txt":
        raise ValueError("Only a parent run using val_set.txt is accepted; Test is forbidden")
    rows, ignored = read_validation_jsonl(root / "validation.jsonl")
    if not rows:
        raise ValueError("No complete validation record is available")
    trajectory, previous_step = [], 0
    for row in rows:
        step = row.get("step")
        if type(step) is not int or step <= previous_step:
            raise ValueError("Validation steps must be strictly increasing positive integers")
        previous_step = step
        if row.get("test_accessed") is not False:
            raise ValueError("Validation record must explicitly declare no Test access")
        count = row.get("image_count")
        if type(count) is not int or count < 1 or (not smoke and count != 450):
            raise ValueError("Formal validation requires exactly 450 images in every logged record")
        logged_metrics = row.get("metrics", {})
        if not {"front_region", "a1_pixel"}.issubset(logged_metrics):
            raise ValueError("Validation needs front_region and a1_pixel metrics")
        evaluated = {name: class_metrics(value) for name, value in logged_metrics.items()}
        front, a1 = evaluated["front_region"], evaluated["a1_pixel"]
        if [x["support"] for x in front["classes"].values()] != [x["support"] for x in a1["classes"].values()]:
            raise ValueError("A1/front validation GT supports differ")
        trajectory.append(dict(
            step=step,
            image_count=count,
            metrics=evaluated,
            front_minus_a1_miou_pp=100 * (front["valid_miou"] - a1["valid_miou"]),
            damaged_to_intact_count_delta=front["damaged_to_intact"]["count"] - a1["damaged_to_intact"]["count"],
        ))
    best = max(trajectory, key=lambda row: row["metrics"]["front_region"]["valid_miou"])
    return dict(
        format="SRAC_CAUSE_LOG_AUDIT_V1",
        front_run=str(root),
        diagnostic_only=True,
        scope="smoke (not formal Val450 evidence)" if smoke else "logged full Val450",
        smoke=smoke,
        formal_val450=not smoke,
        test_accessed=False,
        checkpoint_loaded=False,
        dataset_loaded=False,
        training_completion_verified=False,
        class_order=list(LABELS),
        confusion_orientation="rows=GT, columns=prediction",
        limitations=LOG_LIMITATION,
        missing_class_policy="Undefined per-class ratios are null; valid_miou uses all three foreground classes, absent union as zero",
        records=len(trajectory),
        ignored_incomplete_final_line=ignored,
        best_selection="Highest logged front_region.valid_miou; earliest step wins ties; does not inspect best.pth",
        best=best,
        latest=trajectory[-1],
        trajectory=trajectory,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--front_run", required=True)
    parser.add_argument("--allow_smoke", action="store_true")
    parser.add_argument("--json", action="store_true", help="Full machine-readable report instead of concise display")
    args = parser.parse_args()
    report = summarize_logs(args.front_run, args.allow_smoke)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return
    def pct(value):
        return "N/A" if value is None else f"{100*value:.4f}%"
    print("Scope:", report["scope"], "records:", report["records"], "unfinished trailing record ignored:", report["ignored_incomplete_final_line"])
    print("step   region FG mIoU   A1 FG mIoU   gap(pp)   region Damaged F1")
    for row in report["trajectory"]:
        region, a1 = row["metrics"]["front_region"], row["metrics"]["a1_pixel"]
        print(f"{row['step']:5d}  {pct(region['valid_miou']):>12s}  {pct(a1['valid_miou']):>10s}  "
              f"{row['front_minus_a1_miou_pp']:+.4f}  {pct(region['classes']['Damaged']['f1'])}")
    for title in ("best", "latest"):
        row = report[title]
        print(f"\n{title.upper()} step={row['step']}")
        for branch in ("a1_pixel", "front_region"):
            m = row["metrics"][branch]
            print(branch, "FG mIoU", pct(m["valid_miou"]))
            for label, values in m["classes"].items():
                print(f"  {label:10s} IoU={pct(values['iou'])} P={pct(values['precision'])} "
                      f"R={pct(values['recall'])} F1={pct(values['f1'])}")
            print("  GT Damaged predicted Intact:", m["damaged_to_intact"])
            print("  confusion:", m["confusion"])
    print("\n" + LOG_LIMITATION)


if __name__ == "__main__":
    main()
