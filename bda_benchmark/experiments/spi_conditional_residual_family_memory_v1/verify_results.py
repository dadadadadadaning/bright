from __future__ import annotations
import argparse, json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--result-root", required=True); p.add_argument("--seed", type=int, default=7341)
    a = p.parse_args(argv); path = Path(a.result_root) / f"crfm_conditional_family_seed{a.seed}.json"
    if not path.is_file(): print(f"[MISSING] {path}"); return 3
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("meta_confirm_accessed") or row.get("val450_accessed") or row.get("test_accessed"):
        raise RuntimeError("forbidden split access recorded")
    print("[OK]", path.name); print("risk candidate fraction:", row["calibration_candidate_fraction"])
    for name, report in row["reports"].items():
        print(f"{name:22s} AUC={report['macro_ovr_auroc']:.4f} F1={report['macro_f1']:.4f} event-F1={report['event_macro_f1']:.4f}")
    print("nonnegative events:", row["nonnegative_events_true_vs_p3"]); print("checks:", row["checks"]); print("decision:", row["decision"])
    return 0


if __name__ == "__main__": raise SystemExit(main())

