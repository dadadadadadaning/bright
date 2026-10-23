"""Read-only matched fixed2500 Val450 comparison for the severity study.

Recompute metrics from integer counts, verify image journals and the complete
training image/LR trajectory. New calibration parameters may intentionally
change the parameter count; initialization of the shared core must agree.
No GPU, model inference, significance claim or paired-pixel rescue claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from analyze_results import CLASSES, analyze, checked_matrix, metrics

FORMAT = "BRIGHT_SEVERITY_MEMORY_V1"
VARIANTS = {"original", "calibration", "severity_memory", "shuffled_memory"}
MANIFEST_FIELDS = (
    "initial_core_head_sha256", "train_names", "val_names", "actual_training_names",
    "fixed_train_names", "protocol_info", "support_arm", "seed", "fresh_initialization",
)
PLAN_FIELDS = (
    "format", "source", "addon_sha256", "extension_files", "post500_lr_multiplier",
    "scheduler_horizon", "warmup_steps", "maximum_lr", "weight_decay", "grad_clip",
    "optimizer", "endpoint_step", "method", "read_weight", "original_loss", "added_loss",
    "fixed_train_count", "support_arm", "fresh_initialization", "old_head_weights_reused",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_rows(path):
    # Every record is required to parse. A trailing partial write is not skipped.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summed_confusion(rows, arm):
    out = [[0] * 4 for _ in range(4)]
    for row in rows:
        cm = checked_matrix(row["confusion"][arm])
        for r in range(4):
            for c in range(4):
                out[r][c] += cm[r][c]
    return out


def load(report_path):
    report_path = Path(report_path).resolve()
    root = report_path.parent.parent
    report, manifest, summary = read(report_path), read(root / "manifest.json"), read(root / "summary.json")
    if summary.get("format") != FORMAT or manifest.get("format") != FORMAT:
        raise ValueError("Only the new severity study can be compared here")
    if (summary.get("complete") is not True or summary.get("training_complete") is not True
            or summary.get("full_validation_evidence") is not True or summary.get("smoke") is not False
            or summary.get("phase") != "complete" or summary.get("last_complete_step") != 2500
            or summary.get("endpoint_step") != 2500):
        raise ValueError("Run must be complete, non-smoke and finished at fixed2500: " + str(root))
    if report.get("format") != FORMAT + "_EVALUATION":
        raise ValueError("Wrong evaluation format")
    if (report.get("complete") is not True or report.get("final_integrity_pending") is not False
            or report.get("phase") != "complete" or report.get("smoke") is not False
            or report.get("full_validation_evidence") is not True or report.get("journal_verified") is not True
            or report.get("split") != "val" or report.get("step") != 2500 or report.get("image_count") != 450):
        raise ValueError("Require complete non-smoke fixed2500 Val450, with verified journal")
    if any(d.get("test_accessed") is not False for d in (report, manifest, summary)):
        raise ValueError("Test is unavailable in this comparison version")
    variant = report.get("variant")
    if variant not in VARIANTS or manifest["plan"].get("variant") != variant or summary.get("variant") != variant:
        raise ValueError("Variant identity mismatch")
    if report.get("class_names") != CLASSES:
        raise ValueError("Class order must be explicit and unchanged")
    if report.get("seed") != manifest.get("seed") or report.get("seed") != summary.get("seed"):
        raise ValueError("Run seed identity mismatch")
    if manifest["plan"].get("format") != FORMAT or manifest["plan"].get("endpoint_step") != 2500:
        raise ValueError("Plan is not the fixed2500 severity study")
    if manifest["plan"].get("read_weight") != 0 or manifest["plan"].get("added_loss") is not None:
        raise ValueError("Auxiliary loss would confound the isolated forward-path comparison")
    journal = report_path.parent / "images.jsonl"
    if report.get("images_sha256") != sha256(journal):
        raise ValueError("Evaluation journal hash mismatch")
    rows = json_rows(journal)
    names = [row["image"] for row in rows]
    if names != report.get("requested_names") or names != manifest.get("val_names") or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("Evaluation identities/order/count differ from Val450")
    expected_arms = {"grid_probq", variant}
    if set(report["metrics"]) != expected_arms:
        raise ValueError("Unexpected evaluation arms")
    for i, row in enumerate(rows):
        if row.get("index") != i or row.get("test_accessed") is not False or set(row["confusion"]) != expected_arms:
            raise ValueError("Per-image metadata or arms mismatch")
        grid_gt = [sum(r) for r in checked_matrix(row["confusion"]["grid_probq"])]
        if grid_gt != [sum(r) for r in checked_matrix(row["confusion"][variant])]:
            raise ValueError("Within-image arms disagree on GT counts")
    for arm in expected_arms:
        if summed_confusion(rows, arm) != checked_matrix(report["metrics"][arm]["confusion"]):
            raise ValueError("Per-image confusion does not reproduce aggregate")
    train_path = root / "train.jsonl"
    train = json_rows(train_path)
    if len(train) != 2500 or [r.get("step") for r in train] != list(range(1, 2501)):
        raise ValueError("Training journal is not exactly fresh steps 1..2500")
    train_members = set(manifest.get("actual_training_names", []))
    for row in train:
        lr = row.get("lr")
        if (row.get("image") not in train_members or row.get("variant") != variant
                or row.get("optimizer_step") != row["step"] or isinstance(lr, bool)
                or not isinstance(lr, (float, int)) or not math.isfinite(lr) or lr <= 0
                or row.get("read_loss_evaluated") is not False):
            raise ValueError("Training record contradicts the fresh isolated experiment")
    return report, manifest, rows, train, root


def compare(before_path, after_path):
    a, ma, ra, ta, root_a = load(before_path)
    b, mb, rb, tb, root_b = load(after_path)
    if root_a == root_b or a["variant"] == b["variant"]:
        raise ValueError("Choose distinct variants/runs for component comparison")
    for key in ("split", "step", "seed", "class_names", "requested_names"):
        if a.get(key) != b.get(key):
            raise ValueError("Unmatched report field: " + key)
    for key in MANIFEST_FIELDS:
        if key not in ma or key not in mb or ma[key] != mb[key]:
            raise ValueError("Unmatched/missing manifest field: " + key)
    for key in PLAN_FIELDS:
        if key not in ma["plan"] or key not in mb["plan"] or ma["plan"][key] != mb["plan"][key]:
            raise ValueError("Unmatched/missing plan field: " + key)
    ac, bc = dict(ma["runtime_config"]), dict(mb["runtime_config"])
    ac.pop("output", None)
    bc.pop("output", None)
    if ac != bc:
        raise ValueError("Runtime configuration differs beyond output")
    trajectory_keys = ("step", "image", "event", "epoch", "epoch_seed", "lr", "optimizer_step")
    for left, right in zip(ta, tb):
        for key in trajectory_keys:
            if key not in left or key not in right or left[key] != right[key]:
                raise ValueError(f"Unmatched training trajectory at step {left['step']}: {key}")
    merged = dict(a, metrics={"before": a["metrics"][a["variant"]], "after": b["metrics"][b["variant"]]})
    result = analyze(merged, "before", "after")
    per_image, events = [], {}
    for left, right in zip(ra, rb):
        if left["event"] != right["event"]:
            raise ValueError("Per-image event differs")
        lc, rc = left["confusion"][a["variant"]], right["confusion"][b["variant"]]
        lm, rm = metrics(lc), metrics(rc)
        if lm["ground_truth_counts"] != rm["ground_truth_counts"]:
            raise ValueError("Per-image GT class counts differ")
        per_image.append(dict(image=left["image"], event=left["event"],
                              before_miou4_pct=lm["miou4_pct"], after_miou4_pct=rm["miou4_pct"],
                              miou4_difference_pp=rm["miou4_pct"]-lm["miou4_pct"]))
        pair = events.setdefault(left["event"], [[[0]*4 for _ in range(4)] for _ in range(2)])
        for k, cm in enumerate((lc, rc)):
            for r in range(4):
                for c in range(4):
                    pair[k][r][c] += cm[r][c]
    per_event = {}
    for event, pair in events.items():
        before, after = metrics(pair[0]), metrics(pair[1])
        per_event[event] = dict(before=before, after=after,
                                miou4_difference_pp=after["miou4_pct"]-before["miou4_pct"])
    grid_a = a["metrics"]["grid_probq"]["confusion"]
    grid_b = b["metrics"]["grid_probq"]["confusion"]
    grid_delta = [[grid_b[r][c]-grid_a[r][c] for c in range(4)] for r in range(4)]
    grid_ma, grid_mb = metrics(grid_a), metrics(grid_b)
    net = result["net_confusion_count_changes"]
    counts = {"before": ma.get("parameter_count"), "after": mb.get("parameter_count")}
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ValueError("Parameter counts must be explicit positive integers")
    result.update(
        before_variant=a["variant"], after_variant=b["variant"], seed=a["seed"],
        matched_manifest_checked=True, training_trajectory_verified=True,
        initial_core_head_sha256=ma["initial_core_head_sha256"],
        parameter_counts=counts, per_event=per_event,
        grid_confusion_identical=grid_a == grid_b,
        grid_execution_difference=dict(
            max_absolute_confusion_entry=max(abs(x) for row in grid_delta for x in row),
            sum_absolute_confusion_entries=sum(abs(x) for row in grid_delta for x in row),
            miou4_difference_pp=grid_mb["miou4_pct"]-grid_ma["miou4_pct"],
            net_confusion_count_changes=grid_delta,
            note="Aggregate Grid equality is not a proof of per-pixel equality or hardware reliability.",
        ),
        net_error_direction_changes={
            "damaged_to_intact": net[2][1], "damaged_to_background": net[2][0],
            "intact_to_damaged": net[1][2], "background_to_destroyed": net[0][3],
            "intact_to_destroyed": net[1][3], "damaged_to_destroyed": net[2][3],
            "non_destroyed_to_destroyed": sum(net[r][3] for r in range(3)),
            "building_to_background": sum(net[r][0] for r in range(1, 4)),
            "background_to_building": sum(net[0][c] for c in range(1, 4)),
        },
        source_reports={str(Path(p).resolve()): sha256(p) for p in (before_path, after_path)},
        training_journal_sha256={"before": sha256(root_a/"train.jsonl"), "after": sha256(root_b/"train.jsonl")},
        note="Dataset mIoU is recomputed from summed confusion, never averaged over image mIoUs. Net count differences are not paired pixel rescue/harm. No significance claim.",
    )
    if not result["grid_confusion_identical"]:
        result["grid_comparison_note"] = "Frozen Grid differed across executions; inspect the reported magnitude before interpreting tiny method differences."
    return result, per_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    args = p.parse_args()
    result, rows = compare(args.before, args.after)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (args.output / "paired_images.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["differences"], ensure_ascii=False, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
