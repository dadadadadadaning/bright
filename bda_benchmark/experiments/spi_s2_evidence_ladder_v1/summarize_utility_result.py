from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Convert a Utility Gate audit to ladder summary")
    parser.add_argument("--input", required=True)
    parser.add_argument("--variant", default="u3_utility_gate")
    parser.add_argument("--parent-variant", default="u1_fixed_s2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    reports = payload["reports"]
    changes = payload["change_audit"]
    current = reports[args.variant]
    parent = reports[args.parent_variant]
    current_change = changes[args.variant]
    parent_change = changes[args.parent_variant]
    events = payload["event_wins"][args.variant]
    ranking = payload["gate_statistics"]["selector_ranking"]
    summary = {
        "delta_miou_pp": 100.0 * (current["valid_miou"] - parent["valid_miou"]),
        "harm_reduction_fraction": 1.0 - current_change["harmed"] / max(parent_change["harmed"], 1),
        "selector_auroc": ranking["auroc"],
        "event_nonnegative_fraction": events["nonnegative_vs_fixed_s2"] / max(events["total"], 1),
        "minimum_foreground_delta_pp": min(
            100.0 * (current[name] - parent[name])
            for name in ("intact_iou", "damaged_iou", "destroyed_iou")
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

