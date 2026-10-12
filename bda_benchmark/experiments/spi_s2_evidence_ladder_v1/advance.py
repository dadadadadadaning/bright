from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .contracts import load_protocol, stage_specs, validate_result


def load_results(root: Path, specs) -> dict[str, list[tuple[Path, dict]]]:
    output = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        stage_id = payload.get("stage_id")
        if payload.get("protocol") != "SPI_S2_EVIDENCE_RESULT_V1" or stage_id not in specs:
            continue
        parent_path = payload.get("parent_result")
        validate_result(payload, specs[stage_id], parent_result_path=parent_path)
        output[stage_id].append((path, payload))
    return output


def accepted(results, stage_id: str) -> list[tuple[Path, dict]]:
    return [item for item in results.get(stage_id, ()) if item[1].get("decision") == "ACCEPT"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Show safely eligible S2 evidence experiments")
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    specs = stage_specs(protocol)
    results = load_results(root, specs)
    eligible = []
    blocked = {}
    if not accepted(results, "s1_s2_audit"):
        eligible.append("s1_s2_audit")
    elif not accepted(results, "s2_direction_policy"):
        eligible.append("s2_direction_policy")
    elif not accepted(results, "s3_base_utility_gate"):
        eligible.append("s3_base_utility_gate")
    else:
        for stage_id in ("s4_memory_verifier", "s5_matching_relation", "s6_selective_rgid", "s7_soft_superpixel"):
            if not accepted(results, stage_id):
                eligible.append(stage_id)
        accepted_evidence = [
            stage_id for stage_id in ("s4_memory_verifier", "s5_matching_relation", "s6_selective_rgid", "s7_soft_superpixel")
            if accepted(results, stage_id)
        ]
        if len(accepted_evidence) >= 2:
            eligible.append("s8_combination")
        else:
            blocked["s8_combination"] = f"needs >=2 accepted evidence stages; observed {accepted_evidence}"
    status = {
        stage_id: [
            {"path": str(path), "decision": payload["decision"], "split": payload["split"]}
            for path, payload in rows
        ]
        for stage_id, rows in results.items()
    }
    print(json.dumps({"eligible": eligible, "blocked": blocked, "recorded": status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

