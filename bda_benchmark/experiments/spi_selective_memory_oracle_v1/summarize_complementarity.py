#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser("Summarize recovery funnel and Direct/Memory complementarity")
    parser.add_argument("jsons", nargs="+")
    args = parser.parse_args()
    for value in args.jsons:
        path = Path(value)
        data = json.loads(path.read_text(encoding="utf-8"))
        funnel = data["recovery_funnel"]
        print("=" * 88)
        print(path.name)
        print(f"P3 mIoU                 : {percent(data['parent']['metrics']['valid_miou'])}")
        print(f"Direct Oracle gain      : {data['direct_head_oracle']['gain_pp']:+.4f} pp")
        print(f"Memory Oracle gain      : {data['memory_oracle']['gain_pp']:+.4f} pp")
        print(f"Union Oracle gain       : {data['union_oracle']['gain_pp']:+.4f} pp")
        print(f"Union minus Direct      : {data['union_increment_over_direct_oracle_pp']:+.4f} pp")
        print(f"Memory complementarity : {'PASS' if data['memory_complementarity_pass'] else 'FAIL'}")
        print("Recovery funnel (share of all P3 Affected misses):")
        for key in (
            "inside_predicted_building_rate", "inside_candidate_slic_rate",
            "memory_affected_advice_rate", "memory_changed_to_affected_rate",
            "memory_oracle_recovered_rate", "direct_oracle_recovered_rate",
            "union_oracle_recovered_rate",
        ):
            print(f"  {key:38s} {percent(funnel[key])}")
        print("Proposal overlap:", data["proposal_overlap"])


if __name__ == "__main__":
    main()

