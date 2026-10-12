# -*- coding: utf-8 -*-
"""Create one comparison table for the four isolated CMR-V1 matchers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("CMR-V1 matcher comparison")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for path_string in args.inputs:
        path = Path(path_string)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "CMR_V1_ISOLATED_top20_counterfactual_set_verifier_v1":
            raise RuntimeError(f"unexpected matcher result protocol: {path}")
        validation = payload["results"]["v1_c_counterfactual_combined"]["validation"]
        locked = validation.get("out_of_fold_locked_operating_point")
        rows.append({
            "method": payload["method"],
            "rescue_vs_harm_auroc": validation["auroc"],
            "event_macro_auroc": validation["event_macro_auroc"],
            "locked_operating_point": locked,
            "passed": bool(payload["passed"]),
            "source": str(path.resolve()),
        })
    rows.sort(key=lambda item: float(item["rescue_vs_harm_auroc"]), reverse=True)
    output_payload = {
        "protocol": "CMR_V1_ISOLATED_matcher_comparison_v1",
        "ranking_metric": "validation_rescue_vs_harm_auroc",
        "rows": rows,
        "best_method": rows[0]["method"] if rows else None,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 96)
    print("CMR-V1 isolated matcher comparison")
    for row in rows:
        point = row["locked_operating_point"] or {}
        print(
            f"{row['method']:<8} AUC={row['rescue_vs_harm_auroc']:.4f} "
            f"event={row['event_macro_auroc']:.4f} "
            f"precision={point.get('precision')} net={point.get('net_correction')}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
