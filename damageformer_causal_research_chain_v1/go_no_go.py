#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .core import write_json


def read(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser("Advisory Go/No-Go report; never touches Test")
    p.add_argument("--eval_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_di_auroc_gain", type=float, default=0.01)
    p.add_argument("--min_memory_auroc_gain", type=float, default=0.01)
    args = p.parse_args()
    root = Path(args.eval_root)
    report: Dict[str, Any] = {"protocol": "CAUSAL_CHAIN_GO_NO_GO", "gates": {}, "test_accessed": False}

    a = read(root / "A_region_evidence.json")
    if a:
        p25 = float(a["regions_per_building"]["p25"])
        report["gates"]["G1_region"] = {
            "available": True,
            "regions_p25": p25,
            "peak_delta_di_auroc": a.get("peak_delta_di_auroc"),
            "advisory": "GO" if p25 >= 1.0 else "REVIEW",
            "note": "Cross-building=0 is a structural invariant checked in V2; this cache audit focuses on local freedom/evidence.",
        }
    else:
        report["gates"]["G1_region"] = {"available": False}

    c = read(root / "C_expert_oracle.json")
    if c:
        report["gates"]["G7_expert_headroom"] = {
            "available": True,
            "candidate_damaged_prevalence": c.get("candidate_damaged_prevalence"),
            "perfect_expert_max_rescue": c.get("perfect_expert_max_rescue"),
            "advisory": "GO" if float(c.get("candidate_damaged_prevalence", 0.0)) >= 0.05 else "REVIEW",
        }
    else:
        report["gates"]["G7_expert_headroom"] = {"available": False}

    p1 = read(root / "D_prototype/k1_exclude_event.json")
    p4 = read(root / "D_prototype/k4_exclude_event.json")
    if p1 and p4:
        a1 = float(p1.get("proto_delta_rescue_auroc", float("nan")))
        a4 = float(p4.get("proto_delta_rescue_auroc", float("nan")))
        report["gates"]["G8_multi_prototype"] = {
            "available": True,
            "single_auroc": a1,
            "multi4_auroc": a4,
            "gain": a4 - a1,
            "advisory": "GO_MULTI" if a4 - a1 >= args.min_di_auroc_gain else "SINGLE_MAY_BE_ENOUGH",
        }
    else:
        report["gates"]["G8_multi_prototype"] = {"available": False}

    mem = read(root / "E_memory/none_exclude_event.json")
    proto = p4 or p1
    if mem and proto:
        ma = float(mem.get("memory_delta_rescue_auroc", float("nan")))
        pa = float(proto.get("proto_delta_rescue_auroc", float("nan")))
        report["gates"]["G9_instance_memory"] = {
            "available": True,
            "prototype_auroc": pa,
            "memory_auroc": ma,
            "gain": ma - pa,
            "advisory": "GO_MEMORY" if ma - pa >= args.min_memory_auroc_gain else "MEMORY_NOT_JUSTIFIED",
        }
    else:
        report["gates"]["G9_instance_memory"] = {"available": False}

    real = mem
    shuffled = read(root / "E_memory/shuffle_labels_exclude_event.json")
    wrong = read(root / "E_memory/wrong_regions_exclude_event.json")
    if real and shuffled and wrong:
        r = float(real["memory_delta_rescue_auroc"])
        s = float(shuffled["memory_delta_rescue_auroc"])
        w = float(wrong["memory_delta_rescue_auroc"])
        report["gates"]["G9_memory_controls"] = {
            "available": True,
            "real": r,
            "shuffle_labels": s,
            "wrong_regions": w,
            "advisory": "GO" if r > max(s, w) else "NO_GO_CONTROL_FAILURE",
        }
    else:
        report["gates"]["G9_memory_controls"] = {"available": False}

    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
