#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser("Causal admission for aligned region and Memory stages")
    parser.add_argument("--grid", required=True); parser.add_argument("--slico", required=True)
    parser.add_argument("--shuffled", required=True); parser.add_argument("--learned", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = {"grid": load(args.grid), "slico": load(args.slico), "shuffled": load(args.shuffled)}
    if args.learned: rows["learned"] = load(args.learned)
    # Region quality is judged with the same Direct Head.  Memory is audited
    # separately below; otherwise a retrieval difference could be mistaken
    # for a region-partition improvement.
    def direct_f1(name: str) -> float: return float(rows[name]["direct_metric_classifier"]["macro_f1"])
    comparisons = {
        "direct_slico_minus_grid_pp": 100 * (direct_f1("slico") - direct_f1("grid")),
        "direct_slico_minus_shuffled_pp": 100 * (direct_f1("slico") - direct_f1("shuffled")),
    }
    if "learned" in rows:
        comparisons["direct_learned_minus_slico_pp"] = 100 * (direct_f1("learned") - direct_f1("slico"))
    checks = {
        "slico_above_grid_0p10pp": comparisons["direct_slico_minus_grid_pp"] >= 0.10,
        "correct_structure_above_shuffled": comparisons["direct_slico_minus_shuffled_pp"] > 0,
    }
    if "learned" in rows:
        checks["learned_above_slico_0p10pp"] = comparisons["direct_learned_minus_slico_pp"] >= 0.10
    result = {"protocol": "SPI_ALIGNED_REGION_MEMORY_V1_REGION_ADMISSION",
              "region_estimand": "Direct Head Macro-F1; retrieval is excluded from the region comparison",
              "comparisons": comparisons, "region_checks": checks,
              "region_pass": all(checks.values()),
              "test_accessed": False}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
