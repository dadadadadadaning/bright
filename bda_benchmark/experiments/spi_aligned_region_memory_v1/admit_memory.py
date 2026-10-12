#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Memory admission against Direct, random labels, and P3 error recovery")
    parser.add_argument("--audit", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    row = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    complement = row["p3_memory_complementarity"]
    metrics = {
        "memory_minus_direct_pp": float(row["memory_minus_direct_macro_f1_pp"]),
        "correct_minus_random_pp": float(row["correct_minus_random_macro_f1_pp"]),
        "p3_error_recovery": float(complement["oracle_correction_coverage"]),
        "ungated_harm_rate": float(complement["ungated_harm_rate"]),
    }
    checks = {
        "memory_above_direct_0p20pp": metrics["memory_minus_direct_pp"] >= 0.20,
        "correct_above_random_15pp": metrics["correct_minus_random_pp"] >= 15.0,
        "p3_error_recovery_ge_25pct": metrics["p3_error_recovery"] >= 0.25,
    }
    payload = {"protocol": "SPI_ALIGNED_REGION_MEMORY_V1_MEMORY_ADMISSION", "metrics": metrics,
               "checks": checks, "pass": all(checks.values()),
               "next_stage": "V2 Utility Verifier" if all(checks.values()) else "STOP: do not train writeback",
               "test_accessed": False}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

