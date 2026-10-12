from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Summarize completed FSA/RCM pilot audits")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)

    root = Path(args.result_root)
    rows = []
    for variant in ("fsa", "rcm"):
        path = root / f"{variant}_internal_step{args.steps}_seed{args.seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("variant") != variant:
            raise RuntimeError(f"variant mismatch in {path}")
        if int(payload.get("optimizer_steps", -1)) != args.steps:
            raise RuntimeError(f"optimizer-step mismatch in {path}")
        if int(payload.get("processed_images", -1)) != 291:
            raise RuntimeError(f"incomplete calibration audit in {path}")
        if payload.get("official_instances_accessed", True) or payload.get("test_accessed", True):
            raise RuntimeError(f"forbidden evaluation access recorded in {path}")
        rows.append(
            {
                "variant": variant,
                "miou_gain_pp": float(payload["gains"]["miou_pp"]),
                "damaged_f1_gain_pp": float(payload["gains"]["damaged_f1_pp"]),
                "bootstrap_ci95_low_pp": float(payload["bootstrap"]["ci95_low_pp"]),
                "decision": str(payload["decision"]),
                "path": str(path),
            }
        )

    accepted = [row["variant"] for row in rows if row["decision"] == "PROCEED_TO_FULL_CONFIRMATION"]
    print(json.dumps({"results": rows, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted:
        print("Scientific decision: both adapters stopped after pilot; do not combine them.")
    elif len(accepted) == 1:
        print(f"Scientific decision: only {accepted[0]} may enter an isolated full confirmation.")
    else:
        print("Scientific decision: confirm FSA and RCM separately before any combination.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
