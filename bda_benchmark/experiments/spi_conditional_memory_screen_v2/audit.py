from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_independent_building_memory_v1.audit import (
    GROUPS, compare_report, load_independent, row_events,
)
from bda_benchmark.experiments.spi_independent_building_memory_v1.core import (
    group_matrix, multiclass_report,
)

from . import PROTOCOL
from .core import (ResidualCalibrator, bounded_event_class_rows,
                   fit_standardizer, hierarchical_cross_event_topk,
                   memory_residual_probability, p3_diagnostics,
                   soft_cross_entropy, standardize)


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def train_arm(name, train_x, cal_x, train, cal, args, device):
    torch.manual_seed(args.seed + sum(map(ord, name)))
    model = ResidualCalibrator(train_x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 101 + sum(map(ord, name)))
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(train_x), (min(args.batch_size, len(train_x)),), generator=generator)
        logits = model(train_x[ids].to(device), train["p3_probabilities"][ids].to(device))
        loss = soft_cross_entropy(logits, train["class_fractions"][ids].to(device), train["label_purity"][ids].to(device))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            print(json.dumps({"arm": name, "step": step, "loss": float(loss.detach())}), flush=True)
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(cal_x), args.batch_size):
            stop = min(start + args.batch_size, len(cal_x))
            output.append(model(cal_x[start:stop].to(device), cal["p3_probabilities"][start:stop].to(device)).softmax(1).cpu())
    return torch.cat(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Conditional independent-information and hierarchical retrieval screen")
    parser.add_argument("--train-cards", required=True); parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=128); parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--capacity-per-event-class", type=int, default=256)
    parser.add_argument("--coarse-k", type=int, default=64); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=.25); parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = load_independent(args.train_cards); cal = load_independent(args.calibration_cards)
    train_events, cal_events = row_events(train), row_events(cal)
    target = cal["labels"].long(); p3 = cal["p3_probabilities"].float()

    train_p3, cal_p3 = p3_diagnostics(train["p3_probabilities"]), p3_diagnostics(cal["p3_probabilities"])
    mean, std = fit_standardizer(train_p3); train_p3 = standardize(train_p3, mean, std); cal_p3 = standardize(cal_p3, mean, std)
    train_ind, stat = group_matrix(train, GROUPS["complete_independent"])
    cal_ind, _ = group_matrix(cal, GROUPS["complete_independent"], stat)
    train_real, cal_real = torch.cat((train_p3, train_ind), 1), torch.cat((cal_p3, cal_ind), 1)
    generator = torch.Generator().manual_seed(args.seed + 17)
    train_shuffle = torch.cat((train_p3, train_ind[torch.randperm(len(train_ind), generator=generator)]), 1)
    cal_shuffle = torch.cat((cal_p3, cal_ind[torch.randperm(len(cal_ind), generator=generator)]), 1)

    probabilities = {
        "b0_raw_p3": p3,
        "b1_p3_only_calibrator": train_arm("p3_only", train_p3, cal_p3, train, cal, args, device),
        "b2_p3_plus_true_independent": train_arm("true_independent", train_real, cal_real, train, cal, args, device),
        "b3_p3_plus_shuffled_independent": train_arm("shuffled_independent", train_shuffle, cal_shuffle, train, cal, args, device),
    }
    reports = {name: multiclass_report(target, probability, cal_events) for name, probability in probabilities.items()}
    base_auc = reports["b1_p3_only_calibrator"]["macro_ovr_auroc"]
    true_auc = reports["b2_p3_plus_true_independent"]["macro_ovr_auroc"]
    shuffled_auc = reports["b3_p3_plus_shuffled_independent"]["macro_ovr_auroc"]

    # Factorized retrieval: pre-RGB structure proposes candidates; post-SAR evidence re-ranks them.
    tr_structure, s_stat = group_matrix(train, ("rgb_structure", "geometry"))
    ca_structure, _ = group_matrix(cal, ("rgb_structure", "geometry"), s_stat)
    tr_damage, d_stat = group_matrix(train, ("sar_physics", "relative"))
    ca_damage, _ = group_matrix(cal, ("sar_physics", "relative"), d_stat)
    tr_quality, q_stat = group_matrix(train, ("quality",))
    ca_quality, _ = group_matrix(cal, ("quality",), q_stat)
    rows = bounded_event_class_rows(train["labels"], train_events, args.capacity_per_event_class, args.seed)
    indices_local, scores = hierarchical_cross_event_topk(
        ca_structure, tr_structure[rows], ca_damage, tr_damage[rows], ca_quality, tr_quality[rows],
        cal_events, [train_events[i] for i in rows.tolist()], coarse_k=args.coarse_k, topk=args.topk)
    indices = rows[indices_local]
    residual_probability = memory_residual_probability(
        p3, indices, scores, train["class_fractions"], train["p3_probabilities"], beta=args.beta)
    generator = torch.Generator().manual_seed(args.seed + 29)
    shuffled_values = rows[torch.randperm(len(rows), generator=generator)]
    shuffled_residual_probability = memory_residual_probability(
        p3, shuffled_values[indices_local], scores, train["class_fractions"], train["p3_probabilities"], beta=args.beta)
    reports["b4_hierarchical_residual_memory"] = multiclass_report(target, residual_probability, cal_events)
    reports["b5_shuffled_residual_values"] = multiclass_report(target, shuffled_residual_probability, cal_events)
    retrieval_gain = reports["b4_hierarchical_residual_memory"]["macro_ovr_auroc"] - reports["b0_raw_p3"]["macro_ovr_auroc"]
    retrieval_causal_gain = reports["b4_hierarchical_residual_memory"]["macro_ovr_auroc"] - reports["b5_shuffled_residual_values"]["macro_ovr_auroc"]
    checks = {
        "true_independent_beats_p3_only_by_0p01_auc": true_auc >= base_auc + .01,
        "true_independent_beats_shuffled_by_0p01_auc": true_auc >= shuffled_auc + .01,
        "hierarchical_residual_beats_raw_p3_by_0p01_auc": retrieval_gain >= .01,
        "true_retrieval_beats_shuffled_values_by_0p01_auc": retrieval_causal_gain >= .01,
    }
    payload = {
        "protocol": PROTOCOL, "reports": reports,
        "gains": {"true_independent_vs_p3_only_auc": true_auc - base_auc,
                  "true_independent_vs_shuffled_auc": true_auc - shuffled_auc,
                  "hierarchical_residual_vs_raw_p3_auc": retrieval_gain,
                  "hierarchical_residual_vs_shuffled_values_auc": retrieval_causal_gain},
        "memory": {"rows": int(len(rows)), "capacity_per_event_class": args.capacity_per_event_class,
                   "coarse_k": args.coarse_k, "topk": args.topk, "beta": args.beta,
                   "value": "class_fractions_minus_p3_probabilities"},
        "checks": checks,
        "decision": "ADMIT_EXTERNAL_ENCODER_MEMORY_E1" if all(checks.values()) else "STOP_CURRENT_CARDS_NO_CONDITIONAL_MEMORY_INFORMATION",
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    save(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

