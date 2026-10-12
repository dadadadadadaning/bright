from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_class_safe_macu_v2.core import (
    cluster_event_mask, dual_targets, frozen_gradients, image_event_mask,
    predict, train_dual_linear,
)
from bda_benchmark.experiments.spi_class_safe_macu_v21.core import (
    apply_arm_policy, select_arm_policy,
)
from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL as D1_PROTOCOL
from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.train_verifier import (
    matrix, sha256,
)

from . import REPLAY_PROTOCOL


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != D1_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"unexpected {split} D1 cache")
    forbidden = ("meta_confirm_accessed", "val450_accessed", "test_accessed")
    if any(cache.get(key, True) for key in forbidden):
        raise RuntimeError(f"{split} cache provenance is contaminated")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Replay B1 under strict 14-fold LOEO and retain OOF scores")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--v21-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=.70)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing repeated B1 OOF replay: {output}")
    previous = json.loads(Path(args.v21_json).read_text(encoding="utf-8"))
    if previous.get("decision") != "STOP_CS_MACU_V21_AFTER_LOEO":
        raise RuntimeError("B1 replay requires the frozen V2.1 negative result")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train")
    _validate(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("P3 provenance differs between caches")
    raw_groups = args.groups.strip().lower()
    if raw_groups in {"", "0", "none", "false", "disabled"}:
        print(
            f"Normalized disabled/empty --groups={args.groups!r} to "
            f"{','.join(DEFAULT_GROUPS)}", flush=True,
        )
        groups = DEFAULT_GROUPS
    else:
        groups = tuple(dict.fromkeys(
            value.strip() for value in args.groups.split(",") if value.strip()
        ))
    raw_train = matrix(train, groups, "meta_train")
    raw_calibration = matrix(calibration, groups, "meta_calibration")
    events = sorted(set(train["image_events"]) | set(calibration["image_events"]))
    if len(events) != 14:
        raise RuntimeError(f"expected 14 events, observed {len(events)}")

    rows = {}
    for split, cache in (("meta_train", train), ("meta_calibration", calibration)):
        count = len(cache["image_index"])
        rows[split] = {
            "metric_score": torch.full((count,), float("nan")),
            "harm_score": torch.full((count,), float("nan")),
            "accepted_b1": torch.zeros(count, dtype=torch.bool),
            # dual_targets returns float32 utilities; keep the OOF container in
            # the same dtype because indexed assignment requires an exact
            # source/destination dtype match in PyTorch.
            "linear_metric_utility": torch.full((count,), float("nan"), dtype=torch.float32),
            "linear_damaged_utility": torch.full((count,), float("nan"), dtype=torch.float32),
        }
    folds = {}
    for fold_index, event in enumerate(events):
        train_cluster = cluster_event_mask(train, event, equal=False)
        train_images = image_event_mask(train, event, equal=False)
        inner_cluster = cluster_event_mask(calibration, event, equal=False)
        inner_images = image_event_mask(calibration, event, equal=False)
        held_train, held_calibration = ~train_cluster, ~inner_cluster
        metric_gradient, damaged_gradient = frozen_gradients(train, train_images)
        train_targets = dual_targets(train, metric_gradient, damaged_gradient)
        calibration_targets = dual_targets(calibration, metric_gradient, damaged_gradient)
        mean = raw_train[train_cluster].mean(0)
        std = raw_train[train_cluster].std(0, unbiased=False).clamp_min(1e-4)
        train_features = ((raw_train - mean) / std).clamp(-8, 8)
        calibration_features = ((raw_calibration - mean) / std).clamp(-8, 8)
        print(f"B1 OOF replay {fold_index+1}/14 {event}: training", flush=True)
        model = train_dual_linear(
            train_features, train, train_cluster, *train_targets,
            steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed + fold_index * 1009,
        )
        train_metric, train_harm = predict(model, train_features)
        cal_metric, cal_harm = predict(model, calibration_features)
        policy, diagnostics = select_arm_policy(
            "b1_benefit_only", calibration, cal_metric, cal_harm,
            inner_cluster, inner_images,
            maximum_reject_fraction=args.maximum_reject_fraction,
        )
        for split, held, metric, harm, targets in (
            ("meta_train", held_train, train_metric, train_harm, train_targets),
            ("meta_calibration", held_calibration, cal_metric, cal_harm, calibration_targets),
        ):
            rows[split]["metric_score"][held] = metric[held]
            rows[split]["harm_score"][held] = harm[held]
            rows[split]["accepted_b1"][held] = apply_arm_policy(
                "b1_benefit_only", metric[held], harm[held], policy
            )
            rows[split]["linear_metric_utility"][held] = targets[2][held]
            rows[split]["linear_damaged_utility"][held] = targets[3][held]
        folds[event] = {"policy": policy, "threshold_diagnostics": diagnostics}
        print(
            f"B1 OOF replay {fold_index+1}/14 {event}: "
            f"threshold={policy['metric_threshold']:.2f}", flush=True,
        )

    for split in rows:
        for name, values in rows[split].items():
            if values.dtype != torch.bool and not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"OOF replay incomplete: {split}/{name}")
    payload = {
        "protocol": REPLAY_PROTOCOL,
        "configuration": vars(args),
        "groups": groups,
        "folds": folds,
        "rows": rows,
        "provenance": {
            "train_cache_sha256": sha256(args.train_cache),
            "calibration_cache_sha256": sha256(args.calibration_cache),
            "v21_json_sha256": sha256(args.v21_json),
        },
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
