"""Read a completed, immutable Val450 memory metric evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
    digest, safe_child)

from .evaluate import ALL_ARMS, CONTROL_ARMS, FORMAT, METRIC_ARMS, PAIRS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    return p


def read_formal(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("format") != FORMAT or summary.get("complete") is not True
            or summary.get("full_validation_evidence") is not True
            or summary.get("reproduction_passed") is not True
            or summary.get("image_count") != 450
            or summary.get("audit_file") != "audit.json"
            or not isinstance(summary.get("immutable"), dict)
            or not all(value is True for value in summary["immutable"].values())):
        raise ValueError("Summary does not reference a completed formal Val450 audit")
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if (audit.get("format") != FORMAT or audit.get("complete") is not True
            or audit.get("full_validation_evidence") is not True
            or audit.get("image_count") != 450
            or audit.get("reproduction", {}).get("passed") is not True
            or audit.get("test_accessed") is not False
            or not isinstance(audit.get("immutable"), dict)
            or audit["immutable"] != summary["immutable"]
            or not all(audit["immutable"].values())):
        raise ValueError("Audit lacks full Val450 reproduction/immutability evidence")
    if audit.get("image_journal_sha256") != digest(root / "images.jsonl"):
        raise ValueError("Per-image journal SHA differs from the formal audit")
    if (audit.get("metric_arms") != list(METRIC_ARMS)
            or audit.get("control_arms") != list(CONTROL_ARMS)
            or audit.get("comparison_pairs") != list(PAIRS)
            or set(ALL_ARMS) - set(audit.get("metrics", {}))
            or set(PAIRS) - set(audit.get("transitions", {}))
            or set(CONTROL_ARMS) - set(audit.get("support_diagnostics", {}))):
        raise ValueError("Same-pass metric/diversity control evidence is incomplete")
    rows = [json.loads(line) for line in (root / "images.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if (len(rows) != 450
            or [row.get("index") for row in rows] != list(range(450))
            or len({row.get("image") for row in rows}) != 450):
        raise ValueError("Val450 image journal is incomplete or unordered")
    for row in rows:
        path = safe_child(root, row["prediction_file"])
        if digest(path) != row.get("prediction_sha256"):
            raise ValueError("A Val prediction/support artifact SHA changed")
    return audit


def main():
    args = parser().parse_args()
    audit = read_formal(args.output)
    print("Formal Val450 metric comparison; IoU and mIoU below are percentages.")
    for arm in ("a1_pixel", "grid_probq", "j1", "j3", *CONTROL_ARMS,
                *METRIC_ARMS):
        metric = audit["metrics"][arm]
        print(arm, "mIoU=", round(100 * metric["valid_miou"], 6),
              "IoU=", [round(100 * x, 6) for x in metric["iou"]])
    print("Paired pixel changes vs old-bank diverse Top-8; counts are pixels.")
    for arm in METRIC_ARMS:
        transition = audit["transitions"][f"j3_old_diverse__{arm}"]
        damaged = transition["per_class"]["Damaged"]
        joint = transition["joint_gt_reference_candidate"]
        print(arm,
              "miou_delta_pp=", round(100 * (
                  audit["metrics"][arm]["valid_miou"]
                  - audit["metrics"]["j3_old_diverse"]["valid_miou"]), 6),
              "damaged_rescue=", damaged["rescue"],
              "damaged_harm=", damaged["harm"],
              "damaged_net_false_positive=", damaged["delta_fp"],
              "intact_correct_to_damaged=", joint[1][1][2])
    print("Historical 1e-5 reproduction:", audit["reproduction"]["passed"])
    print("Original bank SHA256:", audit["original_bank_sha256"])


if __name__ == "__main__":
    main()
