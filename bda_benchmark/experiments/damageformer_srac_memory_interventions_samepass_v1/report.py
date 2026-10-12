"""Read and print only a passed formal same-pass intervention audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_from_audit(summary, audit):
    required = ("files", "inherited_code", "new_code", "frozen_models", "runtime", "protocol")
    if (summary.get("complete") is not True
            or summary.get("full_validation_evidence") is not True
            or summary.get("image_count") != 450
            or audit.get("format") != "SRAC_MEMORY_INTERVENTIONS_SAMEPASS_V1"
            or audit.get("complete") is not True
            or audit.get("full_validation_evidence") is not True
            or audit.get("image_count") != 450
            or audit.get("reproduction", {}).get("passed") is not True
            or not all(audit.get("immutable", {}).get(key) is True for key in required)):
        raise ValueError("The Val450 audit did not pass all formal evidence gates")
    arms = ["a1_pixel", "j1"]
    for arm in ("j2", "j3"):
        arms.append(arm)
        arms += [f"{arm}_random_{seed}" for seed in audit["random_seeds"]]
        arms += [f"{arm}_shuffle_labels", f"{arm}_off"]
    rows = []
    for arm in arms:
        score = audit["metrics"][arm]
        tr = audit.get("transitions", {}).get("j1_to_" + arm)
        damaged = tr["classes"][2] if tr is not None else None
        rows.append(dict(
            arm=arm,
            foreground_miou_percent=100 * score["valid_miou"],
            intact_iou_percent=100 * score["iou"][1],
            damaged_iou_percent=100 * score["iou"][2],
            damaged_rescue_vs_j1=None if damaged is None else damaged["rescue"],
            damaged_harm_vs_j1=None if damaged is None else damaged["harm"],
            damaged_false_positive_delta_vs_j1=(None if damaged is None
                                                   else damaged["delta_false_positive"]),
            intact_correct_to_damaged_vs_j1=(None if tr is None
                                              else tr["gt_old_new"][1][1][2]),
        ))
    return rows


def read_report(output):
    output = Path(output)
    summary = _read(output / "summary.json")
    if summary.get("audit_file") != "audit.json":
        raise ValueError("Output is missing its formal audit reference")
    return rows_from_audit(summary, _read(output / "audit.json"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_report(args.output)
    names = ("arm", "foreground_miou_percent", "intact_iou_percent",
             "damaged_iou_percent", "damaged_rescue_vs_j1", "damaged_harm_vs_j1",
             "damaged_false_positive_delta_vs_j1", "intact_correct_to_damaged_vs_j1")
    print("\t".join(names))
    for row in rows:
        print("\t".join("-" if row[name] is None else
                        f"{row[name]:.6f}" if isinstance(row[name], float)
                        else str(row[name]) for name in names))


if __name__ == "__main__":
    main()
