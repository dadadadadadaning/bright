#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    variants = ("a0_rgb", "a0_pseudo", "a0_rgb_pseudo", "b0", "b1", "b2")
    records = {}
    for variant in variants:
        path = root / variant.upper() / "training_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed feature audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_accessed") is not False:
            raise RuntimeError(f"feature audit {variant} has an invalid test-access flag")
        records[variant] = payload["best_metrics"]
    b0 = records["b0"]
    a0_rgb = records["a0_rgb"]
    a0_rs = records["a0_rgb_pseudo"]
    a0_checks = {
        "rgb_sar_macro_better_than_rgb": a0_rs["macro_f1"] > a0_rgb["macro_f1"],
        "rgb_sar_event_not_lower": a0_rs["event_macro_f1"] >= a0_rgb["event_macro_f1"],
        "rgb_sar_damaged_not_lower": a0_rs["recall"][1] >= a0_rgb["recall"][1],
    }
    a0_passed = all(a0_checks.values())
    candidates = sorted(("b0", "b1", "b2"), key=lambda key: (
        records[key]["event_macro_f1"], records[key]["macro_f1"], records[key]["recall"][1]
    ), reverse=True)
    winner = candidates[0]
    win = records[winner]
    external = winner in ("b1", "b2")
    checks = {
        "external_source_wins": external,
        "macro_f1_gain_0p5pp": 100 * (win["macro_f1"] - b0["macro_f1"]) >= 0.5,
        "damaged_recall_gain_2pp": 100 * (win["recall"][1] - b0["recall"][1]) >= 2.0,
        "event_macro_not_lower": win["event_macro_f1"] >= b0["event_macro_f1"],
    }
    passed = a0_passed and checks["external_source_wins"] and checks["event_macro_not_lower"] and (
        checks["macro_f1_gain_0p5pp"] or checks["damaged_recall_gain_2pp"]
    )
    report = {
        "protocol": "P3_RESET_feature_source_decision_v1",
        "records": records,
        "a0_checks": a0_checks,
        "a0_passed": a0_passed,
        "ranking": candidates,
        "winner": winner,
        "checks": checks,
        "passed": passed,
        "decision": "PROCEED_TO_C0_C2" if passed else "STOP_EXTERNAL_FEATURE_REPLACEMENT",
        "test_accessed": False,
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
