from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1 import (
    TARGET_PROTOCOL,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.core import (
    metric_labels,
    per_image_selective_histograms,
)
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import (
    matrix,
    sha256,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    metrics_from_histogram,
)

from . import PROTOCOL
from .core import (
    OBJECTIVES,
    GroupDROState,
    bootstrap_event_mean,
    cluster_events,
    draw_balanced_indices,
    evaluate_heldout_event,
    event_losses,
    impact_weights,
    robust_loss,
    select_source_threshold,
)


DEFAULT_GROUPS = (
    "spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation",
)


class LinearVerifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(values).squeeze(1)


def validate_inputs(cache: dict, targets: dict, split: str, source: str) -> None:
    if cache.get("split") != split or targets.get("split") != split:
        raise RuntimeError(f"unexpected {split} cache/target split")
    if targets.get("protocol") != TARGET_PROTOCOL:
        raise RuntimeError(f"unexpected {split} target protocol")
    if targets.get("source_cache_sha256") != sha256(source):
        raise RuntimeError(f"{split} target/cache fingerprint mismatch")
    if len(targets["metric_utility_pp"]) != len(cache["image_index"]):
        raise RuntimeError(f"{split} target/cluster count mismatch")
    forbidden = (
        "meta_confirm_accessed", "val450_accessed",
        "official_instances_accessed", "test_accessed",
    )
    if any(cache.get(name, True) for name in forbidden):
        raise RuntimeError(f"{split} cache reports forbidden split access")


def score_model(
    model: nn.Module,
    values: torch.Tensor,
    device: torch.device,
    batch_size: int = 16384,
) -> torch.Tensor:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size].to(device)
            rows.append(model(batch).sigmoid().cpu())
    return torch.cat(rows)


def train_fold(
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    utility_all: torch.Tensor,
    events_all: list[str],
    source_events: list[str],
    objective: str,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    groupdro_eta: float,
    groupdro_decay: float,
    vrex_lambda: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict]:
    source_set = set(source_events)
    valid = torch.tensor([
        label >= 0 and event in source_set
        for label, event in zip(labels_all.tolist(), events_all)
    ])
    source_rows = valid.nonzero(as_tuple=False).flatten()
    if not len(source_rows):
        raise RuntimeError("LOEO fold has no eligible source clusters")
    fit = x_all[source_rows].float()
    mean = fit.mean(0)
    std = fit.std(0, unbiased=False).clamp_min(1e-4)
    fit = ((fit - mean) / std).clamp(-8, 8)
    labels = labels_all[source_rows].float()
    weights = impact_weights(utility_all[source_rows])
    local_events = [events_all[int(index)] for index in source_rows]
    event_to_id = {event: index for index, event in enumerate(source_events)}
    rows_by_event = [
        torch.tensor([
            index for index, event in enumerate(local_events) if event == name
        ], dtype=torch.long)
        for name in source_events
    ]
    if any(not len(rows) for rows in rows_by_event):
        raise RuntimeError("at least one source event has no eligible utility labels")

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = LinearVerifier(fit.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    pos_weight = ((labels == 0).sum() / labels.sum().clamp_min(1.0)).to(device)
    dro = GroupDROState.create(
        len(source_events), decay=groupdro_decay, eta=groupdro_eta,
    ) if objective == "groupdro" else None
    last_event_weights = torch.full((len(source_events),), 1.0 / len(source_events))

    for _ in range(steps):
        if objective == "erm":
            ids = torch.randint(
                len(fit), (min(batch_size, len(fit)),), generator=generator,
            )
            logits = model(fit[ids].to(device))
            rows = F.binary_cross_entropy_with_logits(
                logits, labels[ids].to(device), reduction="none",
                pos_weight=pos_weight,
            )
            batch_weights = weights[ids].to(device)
            loss = (rows * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
        else:
            per_event = max(1, batch_size // len(source_events))
            ids, event_ids = draw_balanced_indices(
                rows_by_event, per_event, generator,
            )
            logits = model(fit[ids].to(device))
            losses = event_losses(
                logits, labels[ids].to(device), weights[ids].to(device),
                event_ids.to(device), len(source_events), pos_weight=pos_weight,
            )
            loss, last_event_weights = robust_loss(
                losses, objective, groupdro=dro, vrex_lambda=vrex_lambda,
            )
            last_event_weights = last_event_weights.detach().cpu()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite verifier loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    diagnostics = {
        "source_clusters": int(len(source_rows)),
        "source_events": source_events,
        "effective_event_weights": {
            event: float(last_event_weights[index])
            for index, event in enumerate(source_events)
        },
    }
    return model, mean, std, diagnostics


def run_objective(
    objective: str,
    train: dict,
    calibration: dict,
    train_targets: dict,
    calibration_targets: dict,
    train_matrix: torch.Tensor,
    calibration_matrix: torch.Tensor,
    args: argparse.Namespace,
    objective_index: int,
) -> dict:
    train_events = cluster_events(train)
    train_event_set = set(train_events)
    calibration_event_set = set(map(str, calibration["image_events"]))
    if train_event_set != calibration_event_set:
        raise RuntimeError(
            "meta_train/meta_calibration event sets differ; LOEO composite would be incomplete"
        )
    all_events = sorted(train_event_set)
    if len(all_events) < 3:
        raise RuntimeError("LOEO requires at least three common events")
    train_labels = metric_labels(
        train_targets["metric_utility_pp"], args.epsilon_pp,
    )
    train_utility = train_targets["metric_utility_pp"].float()
    device = torch.device(args.device)
    fold_rows = {}
    composite = torch.empty_like(calibration["fixed_s2_histograms"], dtype=torch.float64)
    thresholds = []
    rejected_clusters = 0.0
    total_clusters = 0

    for fold_index, heldout in enumerate(all_events):
        source_events = [event for event in all_events if event != heldout]
        model, mean, std, diagnostics = train_fold(
            train_matrix, train_labels, train_utility, train_events,
            source_events, objective,
            steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            groupdro_eta=args.groupdro_eta, groupdro_decay=args.groupdro_decay,
            vrex_lambda=args.vrex_lambda,
            seed=args.seed + 1000 * objective_index + fold_index,
            device=device,
        )
        standardized = ((calibration_matrix.float() - mean) / std).clamp(-8, 8)
        scores = score_model(model, standardized, device)
        selected = select_source_threshold(
            calibration, scores, source_events,
            maximum_reject_fraction=args.maximum_reject_fraction,
            minimum_nonnegative_fraction=args.minimum_nonnegative_fraction,
        )
        heldout_row, heldout_histograms = evaluate_heldout_event(
            calibration, scores, heldout, selected["threshold"],
        )
        image_ids = torch.tensor([
            index for index, event in enumerate(calibration["image_events"])
            if str(event) == heldout
        ], dtype=torch.long)
        composite[image_ids] = heldout_histograms
        heldout_row["source_threshold_selection"] = selected
        heldout_row["training"] = diagnostics
        fold_rows[heldout] = heldout_row
        thresholds.append(selected["threshold"])
        rejected_clusters += heldout_row["reject_fraction"] * heldout_row["clusters"]
        total_clusters += heldout_row["clusters"]

    fixed = metrics_from_histogram(calibration["fixed_s2_histograms"].sum(0))
    overall = metrics_from_histogram(composite.sum(0))
    gains = [row["gain_vs_fixed_s2_miou_pp"] for row in fold_rows.values()]
    bootstrap = bootstrap_event_mean(
        gains, args.bootstrap_iterations,
        args.seed + 10000 + objective_index,
    )
    oracle_accept = calibration_targets["metric_utility_pp"].float() > 0
    oracle_hist = per_image_selective_histograms(calibration, oracle_accept)
    oracle = metrics_from_histogram(oracle_hist.sum(0))
    oracle_gain = 100.0 * (oracle["valid_miou"] - fixed["valid_miou"])
    gain = 100.0 * (overall["valid_miou"] - fixed["valid_miou"])
    recovery = gain / max(oracle_gain, 1e-12)
    rejection = rejected_clusters / max(total_clusters, 1)
    checks = {
        "gain_vs_fixed_s2_ge_0_30pp": gain >= args.minimum_gain_pp,
        "oracle_recovery_ge_0_25": recovery >= args.minimum_oracle_recovery,
        "event_bootstrap_ci_low_gt_zero": bootstrap["ci95_low_pp"] > 0,
        "damaged_f1_not_below_fixed_s2": overall["damaged_f1"] >= fixed["damaged_f1"],
        "nonnegative_events_ge_10": sum(value >= 0 for value in gains) >= min(10, len(gains)),
        "reject_fraction_within_cap": rejection <= args.maximum_reject_fraction,
        "all_source_threshold_constraints_met": all(
            row["source_threshold_selection"]["source_constraints_met"]
            for row in fold_rows.values()
        ),
    }
    return {
        "objective": objective,
        "scope": "verifier_level_leave_one_event_out",
        "claim_limit": "P3 exposure to the held-out event is not controlled",
        "fixed_s2": fixed,
        "overall": overall,
        "gain_vs_fixed_s2_miou_pp": gain,
        "event_macro_gain_pp": float(sum(gains) / len(gains)),
        "event_worst_gain_pp": float(min(gains)),
        "event_nonnegative": sum(value >= 0 for value in gains),
        "event_total": len(gains),
        "reject_fraction": rejection,
        "mean_source_selected_threshold": float(sum(thresholds) / len(thresholds)),
        "metric_oracle_gain_pp": oracle_gain,
        "oracle_recovery_ratio": recovery,
        "bootstrap": bootstrap,
        "folds": fold_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        "Isolated verifier-level leave-one-event-out objective comparison",
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--calibration-targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--groupdro-eta", type=float, default=0.05)
    parser.add_argument("--groupdro-decay", type=float, default=0.9)
    parser.add_argument("--vrex-lambda", type=float, default=1.0)
    parser.add_argument("--epsilon-pp", type=float, default=0.0)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-nonnegative-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-oracle-recovery", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    objectives = tuple(dict.fromkeys(
        value.strip() for value in args.objectives.split(",") if value.strip()
    ))
    unknown = sorted(set(objectives) - set(OBJECTIVES))
    if unknown:
        raise ValueError(f"unknown objectives: {unknown}")
    groups = tuple(dict.fromkeys(
        value.strip() for value in args.groups.split(",") if value.strip()
    ))
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(
        args.calibration_cache, map_location="cpu", weights_only=False,
    )
    train_targets = torch.load(
        args.train_targets, map_location="cpu", weights_only=False,
    )
    calibration_targets = torch.load(
        args.calibration_targets, map_location="cpu", weights_only=False,
    )
    validate_inputs(train, train_targets, "meta_train", args.train_cache)
    validate_inputs(
        calibration, calibration_targets, "meta_calibration",
        args.calibration_cache,
    )
    train_matrix = matrix(train, groups, "meta_train")
    calibration_matrix = matrix(calibration, groups, "meta_calibration")

    variants = {}
    for objective_index, objective in enumerate(objectives):
        print(f"Running isolated LOEO objective: {objective}", flush=True)
        variants[objective] = run_objective(
            objective, train, calibration, train_targets, calibration_targets,
            train_matrix, calibration_matrix, args, objective_index,
        )
        row = variants[objective]
        print(json.dumps({
            "objective": objective,
            "gain_pp": row["gain_vs_fixed_s2_miou_pp"],
            "event_macro_gain_pp": row["event_macro_gain_pp"],
            "event_worst_gain_pp": row["event_worst_gain_pp"],
            "event_nonnegative": f"{row['event_nonnegative']}/{row['event_total']}",
            "passed": row["passed"],
        }), flush=True)

    admitted = [row for row in variants.values() if row["passed"]]
    best = max(admitted, key=lambda row: (
        row["event_macro_gain_pp"], row["event_worst_gain_pp"],
        row["gain_vs_fixed_s2_miou_pp"], -row["reject_fraction"],
    )) if admitted else None
    decision = (
        f"ADMIT_{best['objective'].upper()}_FOR_FROZEN_META_CONFIRM"
        if best else "STOP_EVENT_ROBUST_VERIFIER_NO_LOEO_GAIN"
    )
    payload = {
        "protocol": PROTOCOL,
        "configuration": vars(args),
        "feature_groups": groups,
        "variants": variants,
        "selected_objective": best["objective"] if best else None,
        "decision": decision,
        "scope": "verifier-level LOEO only; not end-to-end unseen-event proof",
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "selected_objective": payload["selected_objective"],
        "saved": str(output),
    }, indent=2), flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
