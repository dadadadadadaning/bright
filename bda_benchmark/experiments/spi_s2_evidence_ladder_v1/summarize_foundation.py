from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Create S1/S2 summaries from the Train-only direction audit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=("audit", "direction"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if payload.get("protocol") != "SPI_S2_DIRECTION_POLICY_TRAIN_ONLY_v1":
        raise RuntimeError("foundation input is not a Train-only S2 direction audit")
    if args.kind == "audit":
        audit = payload["global_audit"]
        summary = {
            "changed": audit["changed"],
            "corrected": audit["corrected"],
            "harmed": audit["harmed"],
            "neutral": audit["neutral"],
            "direction_table": audit["directions"],
            "soft_nll_audit": audit["soft_unchanged"],
        }
    else:
        policy = payload["direction_policy"]
        summary = {
            "accepted_directions": [name for name, accepted in policy["policy"].items() if accepted],
            "event_direction_stability": policy["evidence"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

