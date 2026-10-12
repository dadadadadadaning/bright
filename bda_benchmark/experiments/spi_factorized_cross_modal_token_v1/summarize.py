#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("t0_concat", "t1_factorized", "t2_aligned", "t3_multiring")


def main() -> None:
    parser = argparse.ArgumentParser("Summarize T0-T3 factorized retrieval evidence")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    rows = {}
    for variant in VARIANTS:
        path = root / f"{variant}_val_cross_event_step{args.steps}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        memory = data["correct_memory"]
        direct = data["direct_metric_classifier"]
        complement = data["p3_memory_complementarity"]
        event_values = memory["event_accuracy"]
        rows[variant] = {
            "memory_macro_f1": memory["macro_f1"],
            "direct_macro_f1": direct["macro_f1"],
            "memory_minus_direct_pp": data["memory_minus_direct_macro_f1_pp"],
            "correct_minus_random_pp": data["correct_minus_random_macro_f1_pp"],
            "p3_wrong_memory_correct_coverage": complement["oracle_correction_coverage"],
            "ungated_harm_rate": complement["ungated_harm_rate"],
            "worst_event_accuracy": min(event_values.values()),
            "worst_event": min(event_values, key=event_values.get),
            "mean_source_event_count": data["retrieval_diagnostics"]["mean_source_event_count"],
        }
    comparisons = {}
    for name, left, right in (
        ("local_alignment_t2_minus_t1", "t2_aligned", "t1_factorized"),
        ("multiring_t3_minus_t2", "t3_multiring", "t2_aligned"),
    ):
        if left in rows and right in rows:
            comparisons[name] = 100.0 * (
                rows[left]["memory_macro_f1"] - rows[right]["memory_macro_f1"]
            )
    checks = {}
    if "t3_multiring" in rows:
        final = rows["t3_multiring"]
        checks = {
            "memory_above_direct": final["memory_minus_direct_pp"] > 0.0,
            "p3_error_recovery_ge_25pct": final["p3_wrong_memory_correct_coverage"] >= 0.25,
            "correct_above_random_ge_15pp": final["correct_minus_random_pp"] >= 15.0,
            "no_event_accuracy_below_40pct": final["worst_event_accuracy"] >= 0.40,
        }
        if "local_alignment_t2_minus_t1" in comparisons:
            checks["t2_above_t1"] = comparisons["local_alignment_t2_minus_t1"] > 0.0
        if "multiring_t3_minus_t2" in comparisons:
            checks["t3_above_t2"] = comparisons["multiring_t3_minus_t2"] > 0.0
    payload = {
        "protocol": "SPI_FACTORIZED_CROSS_MODAL_TOKEN_V1_SUMMARY",
        "rows": rows,
        "comparisons_pp": comparisons,
        "checks": checks,
        "pass": bool(checks) and all(checks.values()),
        "note": "This is a bare-retrieval admission only; it is not pixel-mIoU or Test evidence.",
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
