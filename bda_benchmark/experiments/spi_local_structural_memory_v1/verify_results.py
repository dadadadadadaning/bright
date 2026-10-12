from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Verify LSM staged outputs")
    parser.add_argument("--result-root", required=True); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv); root = Path(args.result_root)
    e0 = root / f"lsm_e0_local_retrieval_seed{args.seed}.json"
    e1 = root / f"lsm_e1_dependency_seed{args.seed}.json"
    if not e0.is_file():
        print(f"[INCOMPLETE] missing {e0}"); return 3
    first = json.loads(e0.read_text(encoding="utf-8")); print("LSM-E0:", first["decision"])
    if first["decision"] != "ADMIT_LSM_E1_TRUE_NEIGHBOUR_DEPENDENCY":
        print("E1 is correctly locked after the failed E0 audit. No pixel writeback is allowed."); return 0
    if not e1.is_file():
        print(f"[INCOMPLETE] E0 admitted but missing {e1}"); return 3
    second = json.loads(e1.read_text(encoding="utf-8")); print("LSM-E1:", second["decision"])
    if second["decision"] != "ADMIT_LSM_E2_PIXEL_LEVEL_INTEGRATION_DESIGN":
        print("E2 is correctly locked: true local neighbours did not demonstrate causal benefit."); return 0
    print("E1 admitted a separate design review for E2; this verifier never runs pixel writeback or Val450/Test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
