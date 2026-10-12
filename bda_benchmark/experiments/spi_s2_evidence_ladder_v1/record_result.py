from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .contracts import (
    decide_stage,
    load_protocol,
    sha256_file,
    stage_specs,
    validate_result,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Record one isolated ladder result")
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-result", default="")
    parser.add_argument("--source-artifact", action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--split", choices=("internal_calibration", "val450"), required=True)
    args = parser.parse_args(argv)

    protocol = load_protocol()
    specs = stage_specs(protocol)
    if args.stage_id not in specs:
        raise KeyError(f"unknown stage: {args.stage_id}")
    spec = specs[args.stage_id]
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    decision, checks = decide_stage(args.stage_id, summary, protocol)
    payload = {
        "protocol": "SPI_S2_EVIDENCE_RESULT_V1",
        "ladder_protocol": protocol["protocol"],
        "stage_id": args.stage_id,
        "split": args.split,
        "summary": summary,
        "checks": checks,
        "decision": decision,
        "parent_result": str(Path(args.parent_result).resolve()) if args.parent_result else None,
        "parent_result_sha256": sha256_file(args.parent_result) if args.parent_result else None,
        "source_artifacts": {
            str(Path(path).resolve()): sha256_file(path) for path in args.source_artifact
        },
        "checkpoints": {
            str(Path(path).resolve()): sha256_file(path) for path in args.checkpoint
        },
        "val450_accessed": args.split == "val450",
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    validate_result(payload, spec, parent_result_path=args.parent_result or None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"stage": args.stage_id, "decision": decision, "checks": checks}, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
    # A scientific rejection is a valid completed experiment, not a process error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
