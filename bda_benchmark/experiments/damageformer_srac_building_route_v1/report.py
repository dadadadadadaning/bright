"""Print only completed formal Val450 S0/capacity/S2/soft comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ARMS, FORMAT


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True, help="Completed independent Val450 evaluation directory")
    return p


def read_formal(path):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("format") != FORMAT + "_EVAL" or summary.get("complete") is not True
            or summary.get("full_validation_evidence") is not True
            or summary.get("image_count") != 450 or summary.get("smoke") is not False
            or summary.get("test_accessed") is not False
            or summary.get("reproduction_passed") is not True
            or summary.get("e0_protocol_comparison_passed") is not True
            or summary.get("q_a1_reproduction_passed") is not True
            or summary.get("native_a1_replay_ok") is not True
            or not summary.get("immutable") or not all(summary["immutable"].values())
            or summary.get("result_file") != "result.json"):
        raise ValueError("Evaluation is not a complete, reproduced, immutable formal Val450")
    result_path = root / "result.json"
    if digest(result_path) != summary.get("result_sha256"):
        raise ValueError("Formal result hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("complete") is not True or result.get("full_validation_evidence") is not True
            or result.get("image_count") != 450):
        raise ValueError("Formal result disagrees with the summary")
    return result


def comparison(result):
    metrics = result["metrics"]
    if any(arm not in metrics for arm in ARMS):
        raise ValueError("Route result is missing an expected arm")
    arms = {arm: dict(miou_pct=100 * metrics[arm]["valid_miou"],
                      iou_pct=[None if x is None else 100 * x for x in metrics[arm]["iou"]])
            for arm in ARMS}
    pairs = (("unsplit", "hard"), ("capacity_control", "hard"), ("hard", "soft"))
    differences = {f"{new}_minus_{old}_pp": arms[new]["miou_pct"] - arms[old]["miou_pct"]
                   for old, new in pairs}
    return dict(q_source=result.get("q_source"), arms=arms, differences_pp=differences,
                building_strata=result.get("predicted_building_strata"))


def main():
    args = parser().parse_args()
    report = comparison(read_formal(args.evaluation))
    print("Formal Val450, percentage units; IoU order Background / Intact / Damaged / Destroyed")
    print("Q source:", report["q_source"])
    for arm in ARMS:
        row = report["arms"][arm]
        classes = " / ".join("NA" if x is None else f"{x:.4f}" for x in row["iou_pct"])
        print(f"{arm:17s} mIoU={row['miou_pct']:.4f}; IoU={classes}")
    for name, delta in report["differences_pp"].items():
        print(f"{name}: {delta:+.4f} pp")
    for stratum, row in (report["building_strata"] or {}).items():
        summary = row["hard_to_soft_summary"]["per_class"]
        for class_name in ("Intact", "Damaged"):
            current = summary[class_name]
            print(f"{stratum} hard→soft {class_name}: rescue={current['rescue']} "
                  f"harm={current['harm']} FP_delta={current['delta_fp']}")


if __name__ == "__main__":
    main()
