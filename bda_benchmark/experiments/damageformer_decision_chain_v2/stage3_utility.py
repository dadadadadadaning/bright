from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.damageformer_evidence_relation_memory_v1.cache import building_area

from .common import (
    candidate_mask, evaluate_coverages, five_features, load_stage1_cache,
    seed_all, standardizer, utility_targets, write_json, z_apply,
)


class UtilityModel(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(5, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        else:
            raise ValueError(kind)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def fit_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    kind: str,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    seed_all(seed)
    mean = train_x.mean(0)
    std = train_x.std(0, unbiased=False).clamp_min(1.0e-5)
    target_mean = train_y.mean()
    target_std = train_y.std(unbiased=False).clamp_min(1.0e-5)
    x = (train_x - mean) / std
    q = (query_x - mean) / std
    target = (train_y - target_mean) / target_std
    model = UtilityModel(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(int(steps)):
        size = min(512, len(x))
        rows = torch.randint(0, len(x), (size,), generator=generator)
        prediction = model(x[rows].to(device))
        loss = F.smooth_l1_loss(prediction, target[rows].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(q), 4096):
            parts.append(model(q[start:start + 4096].to(device)).float().cpu())
    prediction = torch.cat(parts) * target_std + target_mean
    state = {
        "kind": kind, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_mean": mean, "feature_std": std,
        "target_mean": target_mean, "target_std": target_std,
    }
    return prediction, state


def scatter(total: int, rows: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    result = torch.full((total,), -1.0e9)
    result[rows] = values.float()
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Five-dimensional event-OOF utility")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_stage1_cache(args.train_cache, "train")
    val = load_stage1_cache(args.val_cache, "val")
    evidence = torch.load(args.evidence, map_location="cpu", weights_only=False)
    cal_rows = evidence["cal_rows"].long()
    val_rows = evidence["val_rows"].long()
    current_cal = evidence["current_cal_raw"].float()
    history_cal = evidence["history_cal_raw"].float()
    current_val = evidence["current_val_raw"].float()
    history_val = evidence["history_val_raw"].float()
    target = utility_targets(train, cal_rows)
    cal_area = building_area(train)[cal_rows].float()
    val_area = building_area(val)[val_rows].float()
    events = [str(train["events"][int(row)]) for row in cal_rows]
    unique_events = sorted(set(events))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    oof = {name: torch.empty(len(cal_rows)) for name in ("current", "history", "sum", "linear5", "mlp5")}
    folds = []
    for fold, held_event in enumerate(unique_events):
        held = torch.tensor([event == held_event for event in events], dtype=torch.bool)
        fit = ~held
        if not bool(held.any()) or int(fit.sum()) < 2:
            continue
        cm, cs = standardizer(current_cal[fit])
        hm, hs = standardizer(history_cal[fit])
        zc_fit, zh_fit = z_apply(current_cal[fit], cm, cs), z_apply(history_cal[fit], hm, hs)
        zc_held, zh_held = z_apply(current_cal[held], cm, cs), z_apply(history_cal[held], hm, hs)
        x_fit = five_features(zc_fit, zh_fit, cal_area[fit])
        x_held = five_features(zc_held, zh_held, cal_area[held])
        oof["current"][held] = zc_held
        oof["history"][held] = zh_held
        oof["sum"][held] = zc_held + zh_held
        for kind, name in (("linear", "linear5"), ("mlp", "mlp5")):
            prediction, _ = fit_predict(
                x_fit, target[fit], x_held, kind, args.steps,
                args.seed + 1009 * (fold + 1) + (0 if kind == "linear" else 17), device,
            )
            oof[name][held] = prediction
        folds.append({"held_event": held_event, "fit": int(fit.sum()), "held": int(held.sum())})
        print(json.dumps(folds[-1]), flush=True)

    cal_candidate = torch.zeros(len(train["labels"]), dtype=torch.bool)
    cal_candidate[cal_rows] = True
    oof_results = {
        name: evaluate_coverages(train, scatter(len(train["labels"]), cal_rows, values), cal_candidate, f"oof_{name}")
        for name, values in oof.items()
    }
    selected = max(oof_results, key=lambda name: oof_results[name]["0.02"]["delta_valid_miou_pp"])

    cm, cs = standardizer(current_cal)
    hm, hs = standardizer(history_cal)
    zc_cal, zh_cal = z_apply(current_cal, cm, cs), z_apply(history_cal, hm, hs)
    zc_val, zh_val = z_apply(current_val, cm, cs), z_apply(history_val, hm, hs)
    x_cal = five_features(zc_cal, zh_cal, cal_area)
    x_val = five_features(zc_val, zh_val, val_area)
    val_scores = {"current": zc_val, "history": zh_val, "sum": zc_val + zh_val}
    states = {}
    for kind, name in (("linear", "linear5"), ("mlp", "mlp5")):
        prediction, state = fit_predict(
            x_cal, target, x_val, kind, args.steps,
            args.seed + (0 if kind == "linear" else 17), device,
        )
        val_scores[name] = prediction
        states[name] = state
        torch.save(state, output / f"{name}.pth")

    val_candidate = candidate_mask(val)
    val_results = {
        name: evaluate_coverages(val, scatter(len(val["labels"]), val_rows, values), val_candidate, f"val_{name}")
        for name, values in val_scores.items()
    }
    result = {
        "protocol": "DAMAGEFORMER_DECISION_CHAIN_V2_UTILITY",
        "features": ["z_current", "z_history", "min", "absolute_disagreement", "log_area"],
        "checkpoint_selection": "leave-one-event-out OOF delta_valid_miou_pp@2%",
        "folds": folds,
        "oof_results": oof_results,
        "selected_by_oof": selected,
        "val_results": val_results,
        "admission": {
            "selected_is_learned_utility": selected in {"linear5", "mlp5"},
            "selected_oof_gain_over_current_pp": (
                oof_results[selected]["0.02"]["delta_valid_miou_pp"]
                - oof_results["current"]["0.02"]["delta_valid_miou_pp"]
            ),
        },
        "test_accessed": False,
    }
    write_json(output / "summary.json", result)
    torch.save({
        "selected": selected, "val_rows": val_rows,
        "val_scores": {name: value.half() for name, value in val_scores.items()},
        "test_accessed": False,
    }, output / "val_scores.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

