from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOL = "SPI_S2_EVIDENCE_V2_AUTORUN_SELECTION_v1"
VALID_GROUPS = ("relation", "matching", "slic")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Select admitted single-evidence verifier branches")
    parser.add_argument("--admission-json", required=True)
    parser.add_argument("--policy", choices=("best", "all_admitted"), default="best")
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-output", required=True)
    args = parser.parse_args(argv)

    admission_path = Path(args.admission_json)
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("protocol") != "SPI_S2_PARAMETER_FREE_EVIDENCE_AUDIT_v2":
        raise RuntimeError("unexpected parameter-free admission protocol")
    if (
        admission.get("validation_accessed", True)
        or admission.get("official_instances_accessed", True)
        or admission.get("test_accessed", True)
    ):
        raise RuntimeError("admission result accessed a forbidden split")

    admitted = admission.get("admitted_groups", [])
    unknown = [group for group in admitted if group not in VALID_GROUPS]
    if unknown:
        raise RuntimeError(f"admission contains unsupported groups: {unknown}")
    results = admission.get("results", {})
    ranked = sorted(
        admitted,
        key=lambda group: (
            float(results[group]["calibration_auroc"]),
            float(results[group]["event_macro_auroc"]),
            -VALID_GROUPS.index(group),
        ),
        reverse=True,
    )
    selected = ranked[:1] if args.policy == "best" else ranked
    payload = {
        "protocol": PROTOCOL,
        "policy": args.policy,
        "admission_json": str(admission_path.resolve()),
        "admission_decision": admission.get("decision"),
        "ranked_admitted_groups": ranked,
        "selected_groups": selected,
        "selection_metrics": {
            group: {
                "calibration_auroc": results[group]["calibration_auroc"],
                "event_macro_auroc": results[group]["event_macro_auroc"],
                "event_direction_correct": results[group]["event_direction_correct"],
                "event_total": results[group]["event_total"],
            }
            for group in ranked
        },
        "decision": "RUN_SINGLE_EVIDENCE_BRANCHES" if selected else "STOP_NO_EVIDENCE_ADMITTED",
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    groups_output = Path(args.groups_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups_output.write_text("".join(f"{group}\n" for group in selected), encoding="utf-8")
    print(f"Selection policy: {args.policy}")
    print(f"Ranked admitted: {ranked}")
    print(f"Selected: {selected}")
    print(f"Decision: {payload['decision']}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
