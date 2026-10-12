#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser("Summarize ChangeMamba token and safe Memory admission")
    parser.add_argument("--k1", required=True)
    parser.add_argument("--k2", required=True)
    parser.add_argument("--k3", required=True)
    parser.add_argument("--v1", default=None)
    parser.add_argument("--v2", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audits = {name: load(path) for name, path in (("k1", args.k1), ("k2", args.k2), ("k3", args.k3))}
    rows = {}
    for name, value in audits.items():
        comp = value["complementarity"]
        rows[name] = {
            "direct_macro_f1": value["direct"]["macro_f1"],
            "memory_macro_f1": value["memory"]["macro_f1"],
            "memory_minus_direct_pp": value["memory_minus_direct_macro_f1_pp"],
            "direct_recovery": comp["direct_recovery_rate"],
            "memory_recovery": comp["memory_recovery_rate"],
            "memory_only_rescue": comp["memory_only_rescue"],
            "memory_harm_rate": comp["memory_harm_rate"],
        }
    result = {
        "protocol": "SPI_CHANGEMAMBA_SELECTIVE_MEMORY_V1_SUMMARY",
        "tokens": rows, "test_accessed": False,
    }
    if args.v1 and args.v2:
        v1, v2 = load(args.v1), load(args.v2)
        result["safe_writeback"] = {
            "v1_direct_gain_pp": v1["gain_pp"],
            "v2_memory_gain_pp": v2["gain_pp"],
            "memory_incremental_pp": v2["gain_pp"] - v1["gain_pp"],
            "v1_pass": v1["pass"], "v2_pass": v2["pass"],
            "memory_earns_final_role": v2["pass"] and v2["gain_pp"] > v1["gain_pp"] + 0.20,
        }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

