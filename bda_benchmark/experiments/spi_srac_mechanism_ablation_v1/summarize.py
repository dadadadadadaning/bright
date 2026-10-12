#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Summarize completed mechanism audits")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    evidence, rag, router = {}, {}, {}
    for path in sorted(root.glob("*_evidence.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        evidence[path.stem.removesuffix("_evidence")] = {
            "direct_macro_f1": row["direct"]["macro_f1"],
            "knn_macro_f1": row["knn"]["macro_f1"],
            "knn_minus_direct_pp": row["knn_minus_direct_macro_f1_pp"],
            **row["complementarity"],
        }
    for path in sorted(root.glob("*_rag_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rag[path.stem] = {
            "direct_macro_f1": row["direct"]["macro_f1"],
            "knn_macro_f1": row["knn"]["macro_f1"],
            "rag_macro_f1": row["rag"]["macro_f1"],
            **row["comparisons_pp"],
            **row["complementarity"],
        }
    for path in sorted(root.glob("*_val450.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        router[path.stem] = {
            "gain_pp": row["gain_pp"],
            "valid_miou": row["metrics"]["valid_miou"],
            "damaged_f1": row["metrics"].get("damaged_f1"),
            "nonnegative_events": row["nonnegative_events"],
            "acceptance_rate": row["acceptance_rate"],
            "pass": row["pass"],
        }
    router_comparisons = {}
    for name, row in router.items():
        if not name.endswith("_v2_rag_val450"):
            continue
        prefix = name.removesuffix("_v2_rag_val450")
        direct_name = prefix + "_v1_direct_val450"
        if direct_name in router:
            router_comparisons[prefix] = {
                "v2_minus_v1_gain_pp": row["gain_pp"] - router[direct_name]["gain_pp"],
                "memory_incremental_value": row["gain_pp"] > router[direct_name]["gain_pp"],
            }
    payload = {
        "protocol": "SPI_SRAC_MECHANISM_ABLATION_V1_SUMMARY",
        "evidence": evidence,
        "rag": rag,
        "router": router,
        "router_comparisons": router_comparisons,
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
