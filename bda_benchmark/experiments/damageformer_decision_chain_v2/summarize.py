from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    selection = read(root / "stage1" / "selected.json")
    relation = read(root / "stage2" / "summary.json")
    utility = read(root / "stage3" / "summary.json")
    selected_variant = selection["selected"]["variant"]
    c0 = next(row for row in selection["rows"] if row["variant"] == "C0")
    best = selection["selected"]
    stage1_val = read(root / "stage1" / selected_variant / "summary.json")["val_action"]["0.02"]
    rel_val = relation["val"]
    util_selected = utility["selected_by_oof"]
    result = {
        "protocol": "DAMAGEFORMER_DECISION_CHAIN_V2_SUMMARY",
        "experiment_1_region_evidence": {
            "selected_variant": selected_variant,
            "cal_increment_over_C0_pp": best["cal_delta_valid_miou_pp"] - c0["cal_delta_valid_miou_pp"],
            "selected_val_delta_miou_pp": stage1_val["delta_valid_miou_pp"],
            "pass": selected_variant != "C0" and best["cal_delta_valid_miou_pp"] > c0["cal_delta_valid_miou_pp"],
        },
        "experiment_2_history_increment": {
            "fixed_sum_minus_current_val_pp": relation["admission"]["fixed_sum_minus_current_val_pp"],
            "real_above_shuffled": relation["admission"]["real_above_shuffled"],
            "pass": relation["admission"]["fixed_sum_minus_current_val_pp"] > 0 and relation["admission"]["real_above_shuffled"],
        },
        "experiment_3_five_dimensional_utility": {
            "selected_by_event_oof": util_selected,
            "oof_gain_over_current_pp": utility["admission"]["selected_oof_gain_over_current_pp"],
            "selected_val_delta_miou_pp": utility["val_results"][util_selected]["0.02"]["delta_valid_miou_pp"],
            "pass": utility["admission"]["selected_is_learned_utility"] and utility["admission"]["selected_oof_gain_over_current_pp"] > 0,
        },
        "decision": (
            "GO_5D_UTILITY"
            if utility["admission"]["selected_is_learned_utility"] and utility["admission"]["selected_oof_gain_over_current_pp"] > 0
            else "STOP_AT_BEST_SIMPLER_SCORE"
        ),
        "note": "Pass flags are causal screening, not permission to access Test; multi-seed and event-bootstrap remain required.",
        "test_accessed": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

