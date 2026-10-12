#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.spi_building_relational_memory_v1.core import PROTOCOL, VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser("Summarise staged building relation ablations")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    root = Path(args.result_root); rows = {}
    for variant in VARIANTS:
        path = root / f"{variant}_step{args.steps}.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        rows[variant] = {
            "direct_macro_f1": value["direct"]["macro_f1"],
            "memory_macro_f1": value["memory"]["macro_f1"],
            "memory_minus_direct_pp": value["memory_minus_direct_pp"],
            "correct_minus_random_pp": value["correct_minus_random_pp"],
            "p3_error_recovery": value["p3_to_memory"]["recovery_rate"],
            "p3_correct_harm": value["p3_to_memory"]["harm_rate"],
            "net_correction_count": value["p3_to_memory"]["net_correction_count"],
            "worst_event_accuracy": min(value["memory"]["event_accuracy"].values()),
        }
    def delta(left: str, right: str, field: str) -> float | None:
        if left not in rows or right not in rows:
            return None
        return 100.0 * (rows[left][field] - rows[right][field])
    comparisons = {
        "cross_modal_unaligned_minus_rgb_memory_pp": delta("m1_unaligned_multistat", "m0_rgb_mean", "memory_macro_f1"),
        "alignment_m2_minus_m1_memory_pp": delta("m2_aligned_multistat", "m1_unaligned_multistat", "memory_macro_f1"),
        "multiring_m3_minus_m2_memory_pp": delta("m3_aligned_multiring", "m2_aligned_multistat", "memory_macro_f1"),
        "relation_m4_minus_m3_memory_pp": delta("m4_aligned_relation", "m3_aligned_multiring", "memory_macro_f1"),
        "correct_relation_m4_minus_shuffled_m5_memory_pp": delta("m4_aligned_relation", "m5_shuffled_relation", "memory_macro_f1"),
        "quality_m6_minus_m4_memory_pp": delta("m6_aligned_relation_quality", "m4_aligned_relation", "memory_macro_f1"),
        "alignment_m2_minus_m1_direct_pp": delta("m2_aligned_multistat", "m1_unaligned_multistat", "direct_macro_f1"),
        "multiring_m3_minus_m2_direct_pp": delta("m3_aligned_multiring", "m2_aligned_multistat", "direct_macro_f1"),
        "relation_m4_minus_m3_direct_pp": delta("m4_aligned_relation", "m3_aligned_multiring", "direct_macro_f1"),
        "correct_relation_m4_minus_shuffled_m5_direct_pp": delta("m4_aligned_relation", "m5_shuffled_relation", "direct_macro_f1"),
        "quality_m6_minus_m4_direct_pp": delta("m6_aligned_relation_quality", "m4_aligned_relation", "direct_macro_f1"),
    }
    final = rows.get("m6_aligned_relation_quality", {})
    checks = {
        "aligned_above_unaligned": comparisons["alignment_m2_minus_m1_memory_pp"] is not None and comparisons["alignment_m2_minus_m1_memory_pp"] > 0,
        "multiring_nonnegative": comparisons["multiring_m3_minus_m2_memory_pp"] is not None and comparisons["multiring_m3_minus_m2_memory_pp"] >= 0,
        "relation_above_multistat": comparisons["relation_m4_minus_m3_memory_pp"] is not None and comparisons["relation_m4_minus_m3_memory_pp"] > 0,
        "correct_relation_above_shuffled": comparisons["correct_relation_m4_minus_shuffled_m5_memory_pp"] is not None and comparisons["correct_relation_m4_minus_shuffled_m5_memory_pp"] > 0,
        "quality_nonnegative": comparisons["quality_m6_minus_m4_memory_pp"] is not None and comparisons["quality_m6_minus_m4_memory_pp"] >= 0,
        "memory_above_direct_ge_0_20pp": final.get("memory_minus_direct_pp", -999) >= 0.20,
        "correct_above_random_ge_15pp": final.get("correct_minus_random_pp", -999) >= 15.0,
        "p3_error_recovery_ge_25pct": final.get("p3_error_recovery", -1) >= 0.25,
    }
    result = {
        "protocol": PROTOCOL + "_SUMMARY",
        "rows": rows,
        "comparisons_pp": comparisons,
        "checks": checks,
        "pass": bool(checks) and all(checks.values()),
        "writeback_locked": True,
        "note": "Building-level retrieval admission only; not pixel-mIoU or Test evidence.",
        "test_accessed": False,
    }
    destination = Path(args.output_json); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
