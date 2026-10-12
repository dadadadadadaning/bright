from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .common import (
    candidate_mask, clean_id_candidate, evaluate_coverages,
    event_balanced_prototype_score, load_stage1_cache, standardizer,
    write_json, z_apply,
)


def main() -> None:
    parser = argparse.ArgumentParser("Specialist + Prototype conditional increment")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--specialist_scores", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_stage1_cache(args.train_cache, "train")
    val = load_stage1_cache(args.val_cache, "val")
    specialist = torch.load(args.specialist_scores, map_location="cpu", weights_only=False)
    fit = specialist["fit_mask"].bool()
    cal = specialist["cal_mask"].bool()
    current_train = specialist["train_scores"].float()
    current_val = specialist["val_scores"].float()
    cal_rows = torch.where(cal & candidate_mask(train))[0]
    val_rows = torch.where(candidate_mask(val))[0]
    support = clean_id_candidate(train, fit)
    history_cal = event_balanced_prototype_score(train, train, support, cal_rows)
    history_val = event_balanced_prototype_score(train, val, support, val_rows)

    current_mean, current_std = standardizer(current_train[cal_rows])
    history_mean, history_std = standardizer(history_cal)
    zc_cal = z_apply(current_train[cal_rows], current_mean, current_std)
    zh_cal = z_apply(history_cal, history_mean, history_std)
    zc_val = z_apply(current_val[val_rows], current_mean, current_std)
    zh_val = z_apply(history_val, history_mean, history_std)

    def scatter(n: int, rows: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        result = torch.full((n,), -1.0e9)
        result[rows] = values.float()
        return result

    train_candidate = cal & candidate_mask(train)
    val_candidate = candidate_mask(val)
    rng = random.Random(args.seed)
    permutation = list(range(len(zh_val)))
    rng.shuffle(permutation)
    schemes_cal = {
        "current_only": zc_cal,
        "prototype_only": zh_cal,
        "fixed_sum": zc_cal + zh_cal,
    }
    schemes_val = {
        "current_only": zc_val,
        "prototype_only": zh_val,
        "fixed_sum": zc_val + zh_val,
        "shuffled_history_control": zc_val + zh_val[torch.tensor(permutation)],
        "wrong_sign_history_control": zc_val - zh_val,
    }
    cal_results = {
        name: evaluate_coverages(train, scatter(len(train["labels"]), cal_rows, score), train_candidate, name)
        for name, score in schemes_cal.items()
    }
    val_results = {
        name: evaluate_coverages(val, scatter(len(val["labels"]), val_rows, score), val_candidate, name)
        for name, score in schemes_val.items()
    }

    q_current = float(torch.quantile(zc_cal, 0.90))
    q_history = float(torch.quantile(zh_cal, 0.90))
    labels = val["labels"].long()[val_rows]
    purity = val["label_purity"].float()[val_rows]
    positive_utility = []
    from .common import utility_targets
    positive_utility = utility_targets(val, val_rows) > 0
    groups = {}
    for name, mask in {
        "HH": (zc_val >= q_current) & (zh_val >= q_history),
        "HL": (zc_val >= q_current) & (zh_val < q_history),
        "LH": (zc_val < q_current) & (zh_val >= q_history),
        "LL": (zc_val < q_current) & (zh_val < q_history),
    }.items():
        clean = mask & (purity >= 0.70)
        groups[name] = {
            "count": int(mask.sum()),
            "clean_damaged_prevalence": float(((labels == 1) & clean).sum() / clean.sum().clamp_min(1)),
            "positive_utility_rate": float(positive_utility[mask].float().mean()) if bool(mask.any()) else None,
        }

    result = {
        "protocol": "DAMAGEFORMER_DECISION_CHAIN_V2_RELATION",
        "specialist_variant": specialist["variant"],
        "normalization_source": "Train-Cal candidates only",
        "prototype": "Train-Fit support, class/event-balanced, same-event excluded",
        "train_cal": cal_results,
        "val": val_results,
        "relation_groups": groups,
        "admission": {
            "fixed_sum_minus_current_val_pp": (
                val_results["fixed_sum"]["0.02"]["delta_valid_miou_pp"]
                - val_results["current_only"]["0.02"]["delta_valid_miou_pp"]
            ),
            "real_above_shuffled": (
                val_results["fixed_sum"]["0.02"]["delta_valid_miou_pp"]
                > val_results["shuffled_history_control"]["0.02"]["delta_valid_miou_pp"]
            ),
        },
        "test_accessed": False,
    }
    write_json(output / "summary.json", result)
    torch.save({
        "protocol": result["protocol"], "cal_rows": cal_rows, "val_rows": val_rows,
        "current_cal_raw": current_train[cal_rows], "history_cal_raw": history_cal,
        "current_val_raw": current_val[val_rows], "history_val_raw": history_val,
        "current_mean": current_mean, "current_std": current_std,
        "history_mean": history_mean, "history_std": history_std,
        "test_accessed": False,
    }, output / "evidence.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

