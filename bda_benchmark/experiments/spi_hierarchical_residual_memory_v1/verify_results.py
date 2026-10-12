from __future__ import annotations
import argparse, json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--result-root", required=True); p.add_argument("--seed", type=int, default=7341)
    a = p.parse_args(argv); root = Path(a.result_root)
    names = [f"hrm_d1_target_granularity_seed{a.seed}.json", f"hrm_e0_correct_residual_banks_seed{a.seed}.json", f"hrm_summary_seed{a.seed}.json"]
    payloads = []
    for name in names:
        path = root / name
        if not path.is_file(): print(f"[MISSING] {path}"); return 3
        row = json.loads(path.read_text(encoding="utf-8")); payloads.append(row)
        if row.get("meta_confirm_accessed") or row.get("val450_accessed") or row.get("test_accessed"):
            raise RuntimeError(f"forbidden split access: {path}")
        print(f"[OK] {name}")
    d1, e0, summary = payloads
    print("\nTarget granularity (learned retrieval):")
    for name, row in d1["reports"]["r1_error_focused_metric"].items():
        print(f"{name:18s} AUC={row['all']['macro_ovr_auroc']:.4f} purity={row['all']['neighbour_purity']:.4f} error-purity={row['p3_errors_only']['neighbour_purity']:.4f}")
    print("\nDual-bank error detection:")
    for name in ("true_dual_bank", "shuffled_bank_control", "p3_uncertainty_control"):
        row = e0[name]; print(f"{name:26s} AUC={row['auroc']:.4f} event={row['event_macro_auroc']:.4f} correct={row['correct_events']}/{row['evaluable_events']}")
    print("Family purity:", e0["residual_error_family"]["neighbour_purity"])
    print("Checks:", e0["checks"]); print("Final:", summary); return 0


if __name__ == "__main__": raise SystemExit(main())
