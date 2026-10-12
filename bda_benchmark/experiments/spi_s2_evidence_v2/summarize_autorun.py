from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOL = "SPI_S2_EVIDENCE_V2_AUTORUN_SUMMARY_v1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Summarize independently trained Evidence V2 branches")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    if selection.get("protocol") != "SPI_S2_EVIDENCE_V2_AUTORUN_SELECTION_v1":
        raise RuntimeError("unexpected autorun selection protocol")
    rows = []
    for group in selection.get("selected_groups", []):
        path = Path(args.result_root) / (
            f"{group}_verifier_internal_step{args.steps}_seed{args.seed}.json"
        )
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol") != "SPI_S2_SINGLE_EVIDENCE_VERIFIER_INTERNAL_CALIBRATION_v2":
            raise RuntimeError(f"unexpected verifier protocol: {path}")
        if (
            result.get("validation_accessed", True)
            or result.get("official_instances_accessed", True)
            or result.get("test_accessed", True)
        ):
            raise RuntimeError(f"verifier result accessed a forbidden split: {path}")
        rows.append(
            {
                "group": group,
                "gain_vs_fixed_s2_miou_pp": result["gain_vs_fixed_s2_miou_pp"],
                "selector_auroc": result["selector"]["auroc"],
                "accepted": result["changes"]["accepted"],
                "corrected": result["changes"]["corrected"],
                "harmed": result["changes"]["harmed"],
                "bootstrap_ci95_low_pp": result["bootstrap"]["ci95_low_pp"],
                "checks": result["checks"],
                "decision": result["decision"],
                "path": str(path.resolve()),
            }
        )
    passed = [
        row["group"]
        for row in rows
        if row["decision"] == "FREEZE_METHOD_THEN_ALLOW_ONE_VAL450_RUN"
    ]
    if len(passed) >= 2:
        decision = "ELIGIBLE_FOR_MANUAL_CONTROLLED_S8_DESIGN"
    elif len(passed) == 1:
        decision = "FREEZE_BEST_SINGLE_BRANCH_NO_S8_YET"
    else:
        decision = "STOP_EVIDENCE_VERIFIER_V2"
    payload = {
        "protocol": PROTOCOL,
        "selection_policy": selection["policy"],
        "branches": rows,
        "independently_passed": passed,
        "decision": decision,
        "automatic_joint_training_started": False,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Scientific decision: {decision}")
    print("No joint model and no Val450 run were started automatically.")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
