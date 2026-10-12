from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import PROTOCOL


VARIANTS = (
    "sp1_parameter_matched",
    "sp2_slic_relation",
    "sp3_slic_consistency",
    "sp4_shuffled_slic",
    "sp5_regular_grid",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Summarize the corrected SP0-SP5 causal chain")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payloads = {}
    for variant in VARIANTS:
        path = Path(args.result_root) / f"{variant}_step{args.steps}_seed{args.seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol") != PROTOCOL + "_INTERNAL_EVAL":
            raise RuntimeError(f"protocol mismatch in {path}")
        payloads[variant] = value

    parent_values = [row["parent"]["valid_miou"] for row in payloads.values()]
    if max(parent_values) - min(parent_values) > 1e-10:
        raise RuntimeError("SP runs do not reproduce the same frozen P3 parent")

    rows = {
        "sp0_frozen_p3": payloads[VARIANTS[0]]["parent"],
        **{variant: payloads[variant]["variant_metrics"] for variant in VARIANTS},
    }

    def comparison(left: str, right: str) -> dict:
        lhs, rhs = rows[left], rows[right]
        return {
            "left": left,
            "right": right,
            "valid_miou_gain_pp": 100.0 * (lhs["valid_miou"] - rhs["valid_miou"]),
            "damaged_f1_gain_pp": 100.0 * (lhs["damaged_f1"] - rhs["damaged_f1"]),
            "destroyed_f1_gain_pp": 100.0 * (lhs["destroyed_f1"] - rhs["destroyed_f1"]),
        }

    comparisons = {
        "region_relation_sp2_minus_sp1": comparison("sp2_slic_relation", "sp1_parameter_matched"),
        "consistency_sp3_minus_sp2": comparison("sp3_slic_consistency", "sp2_slic_relation"),
        "correct_structure_sp3_minus_sp4": comparison("sp3_slic_consistency", "sp4_shuffled_slic"),
        "slic_sp3_minus_grid_sp5": comparison("sp3_slic_consistency", "sp5_regular_grid"),
    }
    checks = {
        name: row["valid_miou_gain_pp"] > 0.0
        for name, row in comparisons.items()
    }
    result = {
        "protocol": PROTOCOL + "_SP0_SP5_SUMMARY",
        "steps": args.steps,
        "seed": args.seed,
        "rows": rows,
        "comparisons": comparisons,
        "mechanism_pattern_checks": checks,
        "mechanism_pattern_pass": all(checks.values()),
        "warning": "Positive point estimates are necessary but formal claims still require paired confidence intervals on the frozen confirmation split.",
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
