#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    m2 = json.loads(Path(args.m2).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = {
        "reference": m2["variant"], "candidate": candidate["variant"],
        "gain_over_m2_pp": candidate["gain_pp"] - m2["gain_pp"],
        "damaged_f1_delta_pp": 100.0 * (candidate["metrics"]["damaged_f1"] - m2["metrics"]["damaged_f1"]),
        "event_nonnegative_vs_p3": candidate["nonnegative_events"],
        "checks": {
            "candidate_over_m2_ge_0_20pp": candidate["gain_pp"] - m2["gain_pp"] >= 0.20,
            "damaged_f1_not_below_m2": candidate["metrics"]["damaged_f1"] >= m2["metrics"]["damaged_f1"],
            "at_least_10_events_nonnegative_vs_p3": candidate["nonnegative_events"] >= 10,
            "candidate_passed_absolute_gate": bool(candidate.get("pass", False)),
        },
        "test_accessed": False,
    }
    result["pass"] = all(result["checks"].values())
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

