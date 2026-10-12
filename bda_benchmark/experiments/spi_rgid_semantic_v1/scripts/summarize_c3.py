from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the locked C3-vs-C2 gate.")
    parser.add_argument("result_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.result_json).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected_protocol = "SPI_LOCKED_A6_GEOMETRY_CAUSAL_STANDARD_ML_VAL450_v1"
    if payload.get("protocol") != expected_protocol:
        raise ValueError(f"unexpected protocol: {payload.get('protocol')!r}")
    if payload.get("official_instances_accessed") is not False:
        raise ValueError("official-instance access flag is not false")
    if payload.get("original_standard_test_accessed") is not False:
        raise ValueError("test-access flag is not false")
    if not payload.get("p3_reproduction", {}).get("passed", False):
        raise ValueError("locked P3 was not reproduced")

    reports = payload.get("reports", {})
    required = {"c0_raw_p3", "c2_loccc_a6", "c3_rgid_g3_a6"}
    missing = required.difference(reports)
    if missing:
        raise ValueError(f"C3 audit is incomplete; missing {sorted(missing)}")
    c0 = reports["c0_raw_p3"]
    c2 = reports["c2_loccc_a6"]
    c3 = reports["c3_rgid_g3_a6"]
    summary = {
        "c2_miou_percent": 100.0 * c2["valid_miou"],
        "c3_miou_percent": 100.0 * c3["valid_miou"],
        "c3_minus_c2_miou_pp": 100.0 * (c3["valid_miou"] - c2["valid_miou"]),
        "c3_minus_p3_miou_pp": 100.0 * (c3["valid_miou"] - c0["valid_miou"]),
        "c3_minus_p3_damaged_f1_pp": 100.0
        * (c3["damaged_f1"] - c0["damaged_f1"]),
        "event_wins_vs_p3": payload["event_win_counts_vs_c0"]["c3_rgid_g3_a6"],
        "stratified_bootstrap": payload["event_stratified_image_bootstrap"][
            "comparisons"
        ]["c3_rgid_g3_a6_minus_c2_loccc_a6"],
        "registered_decision": payload["decision"]["rgid_geometry"],
    }
    print(json.dumps(summary, indent=2))
    print(
        "NEXT="
        + (
            "ENABLE_POST_C3_SPLIT_ON_DEMAND"
            if summary["registered_decision"]["pass"]
            else "STOP_JOINT_LEARNING_AND_DIAGNOSE_CENTER_GEOMETRY"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
