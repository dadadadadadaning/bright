from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser("Verify QSD-RCM result files")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv); root = Path(args.result_root)
    names = [f"qsd_d0_e0_retrieval_seed{args.seed}.json",
             f"qsd_e1_e2_value_gate_seed{args.seed}.json",
             f"qsd_summary_seed{args.seed}.json"]
    for name in names:
        path = root / name
        if not path.is_file():
            print(f"[MISSING] {path}"); return 3
        payload = json.loads(path.read_text(encoding="utf-8"))
        forbidden = (payload.get("meta_confirm_accessed"), payload.get("val450_accessed"),
                     payload.get("test_accessed"))
        if any(forbidden):
            raise RuntimeError(f"forbidden split access recorded in {path}")
        print(f"[OK] {name}")
    summary = json.loads((root / names[-1]).read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

