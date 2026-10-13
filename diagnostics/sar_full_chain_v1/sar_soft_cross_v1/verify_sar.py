"""Read-only matched SAR-cue validation; never trains or changes old evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


EXPECTED_ARMS = {
    "baseline": "baseline", "sar": "sar_cue",
    "rgb": "rgb_capacity", "shifted": "shifted_sar",
}


def check_metric_values(record):
    """Confirm stored numbers follow this experiment's fixed foreground-3 mean."""
    import numpy as np
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.compare import ARMS
    groups = [("global", record["metrics"])]
    groups.extend(("event " + name, value) for name, value in record["per_event"].items())
    for scope, rows in groups:
        for arm in ARMS:
            item = rows[arm]
            cm = np.asarray(item["confusion"], dtype=np.float64)
            if cm.shape != (4, 4) or not np.isfinite(cm).all() or (cm < 0).any():
                raise ValueError(f"Invalid confusion: {scope}/{arm}")
            tp = np.diag(cm)
            union = cm.sum(0) + cm.sum(1) - tp
            expected_iou = [float(n / d) if d > 0 else None for n, d in zip(tp, union)]
            expected_mean = float((tp / np.maximum(union, 1))[1:].mean()) if cm.sum() > 0 else None
            pairs = list(zip(item["iou"], expected_iou)) + [(item["valid_miou"], expected_mean)]
            for actual, expected in pairs:
                if (actual is None) != (expected is None) or (actual is not None and
                        not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)):
                    raise ValueError(f"Stored IoU or foreground mIoU disagrees with confusion: {scope}/{arm}")


def validate(args):
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "bda_benchmark").is_dir():
        raise ValueError("--repo-root must be the existing BRIGHT checkout")
    sys.path.insert(0, str(repo_root))
    sys.dont_write_bytecode = True
    from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.compare import (
        checked_eval, compare, metric_difference,
    )
    from audit_predictions import audit_evaluation

    records, status = {}, {}
    for name, expected in EXPECTED_ARMS.items():
        supplied = getattr(args, name)
        if not supplied:
            status[name] = {"passed": False, "state": "not_supplied"}
            continue
        root = Path(supplied).resolve()
        try:
            _, row = checked_eval(root)
            if row["arm"] != expected:
                raise ValueError(f"Expected arm {expected!r}, got {row['arm']!r}")
            if row.get("q_a1_reproduction", {}).get("passed") is not True:
                raise ValueError("New Q A1 replay gate did not pass")
            check_metric_values(row)
            records[name] = row
            item = {"passed": True, "state": "aggregate_gates_passed", "directory": str(root)}
            if not args.reports_only:
                audit = audit_evaluation(repo_root, root)
                item.update(artifact_audit=audit, passed=audit.get("passed") is True)
                item["state"] = "verified" if item["passed"] else "artifact_audit_failed"
            status[name] = item
        except Exception as exc:
            status[name] = {"passed": False, "state": "failed", "directory": str(root),
                            "error": f"{type(exc).__name__}: {exc}"}

    comparisons, match_errors = {}, {}
    if "baseline" in records:
        for name in ("sar", "rgb", "shifted"):
            if name in records:
                try:
                    comparisons[name] = compare(records["baseline"], records[name])
                except Exception as exc:
                    match_errors[name] = f"{type(exc).__name__}: {exc}"

    complete = bool(not args.reports_only and len(comparisons) == 3 and not match_errors
                    and all(item["passed"] for item in status.values()))
    if complete:
        target_digests = [item["artifact_audit"].get("ordered_targets_sha256")
                          for item in status.values()]
        if not all(target_digests) or len(set(target_digests)) != 1:
            match_errors["validation_targets"] = "Val image order, events or archived labels differ"
            complete = False

    direct = {}
    if len(comparisons) == 3 and not match_errors:
        for readout in ("q_region", "q_grid_probq"):
            sar = records["sar"]["metrics"][readout]
            direct[readout] = {
                "sar_minus_baseline": metric_difference(records["baseline"]["metrics"][readout], sar),
                "sar_minus_rgb_capacity": metric_difference(records["rgb"]["metrics"][readout], sar),
                "sar_minus_shifted_sar": metric_difference(records["shifted"]["metrics"][readout], sar),
            }

    return {
        "format": "BRIGHT_EXTERNAL_SAR_VALIDATION_V1",
        "complete": complete, "reports_only": args.reports_only,
        "test_accessed": False, "training": False,
        "status": status, "match_errors": match_errors,
        "matched_aggregate_comparisons": comparisons,
        "direct_differences_pp": direct,
        "interpretation": (
            "All metrics are Val450 foreground-three-class mIoU and four class IoUs. "
            "complete means matched formal reports and prediction evidence verified, "
            "not positive effects or statistical significance. Compare SAR to both controls; "
            "a single seed cannot establish a stable mechanism. Reports-only mode never "
            "claims complete validation. No results, labels, hashes or source files are repaired."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--baseline", required=True, help="Formal trained-baseline Val directory")
    parser.add_argument("--sar", required=True, help="Formal sar_cue Val directory")
    parser.add_argument("--rgb", help="Formal rgb_capacity Val directory; missing means pending")
    parser.add_argument("--shifted", help="Formal shifted_sar Val directory; missing means pending")
    parser.add_argument("--reports-only", action="store_true", help="Quick report inspection; incomplete evidence")
    parser.add_argument("--output", help="New external JSON file; existing files are never overwritten")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error("--output must be a fresh file")
    if args.output:
        destination = Path(args.output).resolve()
        protected = [Path(args.repo_root).resolve()]
        for name in EXPECTED_ARMS:
            if getattr(args, name):
                root = Path(getattr(args, name)).resolve()
                protected.append(root.parent if root.is_file() else root)
        if any(destination.is_relative_to(root) for root in protected):
            parser.error("--output must be outside the checkout and all source evaluations")
    report = validate(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
