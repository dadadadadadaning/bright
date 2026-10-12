from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence import average_precision
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PROTOCOL


MAIN_GROUPS = ("spatial", "rgb_role", "sar_role", "direct", "matching", "relation")
DIAGNOSTIC_GROUPS = tuple(
    f"{name}_stage4" for name in MAIN_GROUPS if name != "spatial"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utility_labels(cache: dict, margin: float) -> torch.Tensor:
    ratio = cache["utility"]["utility_ratio"].float()
    labels = torch.full(ratio.shape, -1, dtype=torch.long)
    labels[ratio >= margin] = 1
    labels[ratio <= -margin] = 0
    return labels


def weighted_auroc(labels: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor) -> float:
    labels = labels.long().flatten()
    scores = scores.float().flatten()
    weights = weights.float().flatten().clamp_min(0.0)
    positive = weights[labels == 1].sum()
    negative = weights[labels == 0].sum()
    if float(positive) <= 0 or float(negative) <= 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    y = labels[order]
    w = weights[order]
    _, groups = torch.unique_consecutive(scores[order], return_inverse=True)
    group_count = int(groups.max()) + 1
    positive_weight = torch.zeros(group_count).scatter_add_(
        0, groups, w * (y == 1)
    )
    negative_weight = torch.zeros(group_count).scatter_add_(
        0, groups, w * (y == 0)
    )
    negative_before = torch.cumsum(negative_weight, 0) - negative_weight
    concordant = (
        positive_weight * (negative_before + 0.5 * negative_weight)
    ).sum()
    return float(concordant / (positive * negative))


def _best_scalar(values: torch.Tensor, labels: torch.Tensor, names: tuple[str, ...]) -> dict:
    rows = []
    for index, name in enumerate(names):
        raw = binary_auroc(labels, values[:, index])
        orientation = 1.0 if raw >= 0.5 else -1.0
        oriented = values[:, index] * orientation
        rows.append(
            {
                "index": index, "name": name, "orientation": orientation,
                "train_auroc": max(raw, 1.0 - raw),
                "mean": float(oriented.mean()),
                "std": float(oriented.std(unbiased=False).clamp_min(1e-4)),
            }
        )
    return max(rows, key=lambda row: (row["train_auroc"], -row["index"]))


def select_direction_policy(
    cache: dict,
    labels: torch.Tensor,
    group: str,
    minimum: int,
    eligible: torch.Tensor | None = None,
) -> dict:
    valid = labels >= 0
    if eligible is not None:
        valid &= eligible.bool()
    values = cache["features"][group].float()
    names = tuple(cache["feature_names"][group])
    global_policy = _best_scalar(values[valid], labels[valid], names)
    directions = cache["direction_code"].long()
    policies = {}
    for direction in sorted(torch.unique(directions[valid]).tolist()):
        mask = valid & (directions == direction)
        enough = int((labels[mask] == 0).sum()) >= minimum and int((labels[mask] == 1).sum()) >= minimum
        policies[str(direction)] = (
            _best_scalar(values[mask], labels[mask], names)
            if enough else dict(global_policy)
        )
        policies[str(direction)]["direction_specific"] = bool(enough)
    return {"global": global_policy, "directions": policies}


def policy_scores(cache: dict, group: str, policy: dict) -> torch.Tensor:
    values = cache["features"][group].float()
    directions = cache["direction_code"].long()
    scores = torch.empty(values.shape[0])
    for direction in torch.unique(directions).tolist():
        mask = directions == direction
        current = policy["directions"].get(str(direction), policy["global"])
        raw = values[mask, int(current["index"])] * float(current["orientation"])
        scores[mask] = (raw - float(current["mean"])) / max(float(current["std"]), 1e-4)
    return scores


def _cluster_events(cache: dict) -> list[str]:
    return [cache["image_events"][int(index)] for index in cache["image_index"].long()]


def evaluate_cross_event(
    train: dict,
    train_labels: torch.Tensor,
    cache: dict,
    labels: torch.Tensor,
    group: str,
    minimum: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Evaluate event E using a scalar policy selected without event E."""

    valid = labels >= 0
    score_all = torch.empty(len(labels), dtype=torch.float32)
    train_events = _cluster_events(train)
    calibration_events = _cluster_events(cache)
    heldout_policies = {}
    for event in sorted(set(cache["image_events"])):
        train_eligible = torch.tensor([value != event for value in train_events])
        policy = select_direction_policy(
            train, train_labels, group, minimum, eligible=train_eligible
        )
        event_mask = torch.tensor([value == event for value in calibration_events])
        score_all[event_mask] = policy_scores(cache, group, policy)[event_mask]
        heldout_policies[event] = policy

    target = labels[valid]
    score = score_all[valid]
    weights = cache["utility"]["changed"].float()[valid]
    image_rows = cache["image_index"].long()[valid]
    events = {}
    for event in sorted(set(cache["image_events"])):
        image_ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        mask = torch.isin(image_rows, image_ids)
        current = target[mask]
        if int((current == 0).sum()) and int((current == 1).sum()):
            events[event] = binary_auroc(current, score[mask])
    aucs = list(events.values())
    return (
        {
            "calibration_auroc": binary_auroc(target, score),
            "pixel_weighted_auroc": weighted_auroc(target, score, weights),
            "average_precision": average_precision(target, score),
            "count": int(valid.sum()), "positive": int((target == 1).sum()),
            "negative": int((target == 0).sum()),
            "event_macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "correct_events": sum(value > 0.5 for value in aucs),
            "event_total": len(aucs), "events": events,
            "direction_policy_by_heldout_event": heldout_policies,
        },
        target,
        score,
    )


def permutation_p(labels: torch.Tensor, scores: torch.Tensor, observed: float, iterations: int, seed: int) -> float:
    # The score order is fixed across permutations.  Pre-computing mid-ranks
    # avoids sorting every 100k-cluster vector hundreds of times.
    ranks = torch.from_numpy(
        rankdata(scores.detach().cpu().numpy(), method="average")
    ).to(torch.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    generator = torch.Generator().manual_seed(seed)
    exceed = 1
    for _ in range(iterations):
        shuffled = labels[torch.randperm(len(labels), generator=generator)]
        rank_sum = ranks[shuffled == 1].sum()
        auc = float(
            (rank_sum - positives * (positives + 1) / 2.0)
            / (positives * negatives)
        )
        exceed += int(auc >= observed)
    return exceed / (iterations + 1)


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"expected {PROTOCOL} {split}")
    if cache.get("val450_accessed", True) or cache.get("official_instances_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError("cluster evidence cache accessed a forbidden split")
    if split != "meta_confirm" and cache.get("meta_confirm_accessed", True):
        raise RuntimeError("training/calibration cache accessed meta_confirm")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Parameter-free cluster-level RGB-SAR evidence audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--oracle-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--minimum-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-event-macro", type=float, default=0.58)
    parser.add_argument("--minimum-correct-events", type=int, default=10)
    parser.add_argument("--minimum-control-gain", type=float, default=0.02)
    parser.add_argument("--minimum-class-count", type=int, default=200)
    parser.add_argument("--minimum-direction-class-count", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train")
    _validate(calibration, "meta_calibration")
    oracle = json.loads(Path(args.oracle_json).read_text(encoding="utf-8"))
    if oracle.get("decision") != "ADMIT_S2_CLUSTER_EVIDENCE_E0":
        raise RuntimeError("Cluster Evidence refused: Cluster Oracle did not pass")
    if train["p3_sha256"] != calibration["p3_sha256"] or set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("train/calibration provenance mismatch or image overlap")

    train_labels = utility_labels(train, args.utility_margin)
    calibration_labels = utility_labels(calibration, args.utility_margin)
    available = tuple(
        name for name in MAIN_GROUPS + DIAGNOSTIC_GROUPS
        if name in train["features"] and name in calibration["features"]
    )
    results, score_rows = {}, {}
    for index, group in enumerate(available):
        row, target, score = evaluate_cross_event(
            train, train_labels, calibration, calibration_labels, group,
            args.minimum_direction_class_count,
        )
        row["permutation_p"] = (
            permutation_p(
                target, score, row["calibration_auroc"], args.permutations,
                args.seed + index,
            )
            if group in ("rgb_role", "sar_role", "matching", "relation")
            else None
        )
        results[group], score_rows[group] = row, score

    spatial_auc = results["spatial"]["calibration_auroc"]
    direct_auc = results.get("direct", results["spatial"])["calibration_auroc"]
    checks, admitted = {}, []
    controls = {
        "rgb_role": spatial_auc,
        "sar_role": spatial_auc,
        "matching": max(direct_auc, results.get("shuffled_matching", {"calibration_auroc": 0.5})["calibration_auroc"]),
        "relation": max(direct_auc, results.get("shuffled_relation", {"calibration_auroc": 0.5})["calibration_auroc"]),
    }
    # Evaluate stored shuffled controls as diagnostics using the same protocol.
    for group in (
        "shuffled_direct", "shuffled_matching", "shuffled_relation",
        "shuffled_direct_stage4", "shuffled_matching_stage4",
        "shuffled_relation_stage4",
    ):
        if group not in train["features"] or group not in calibration["features"]:
            continue
        row, target, score = evaluate_cross_event(
            train, train_labels, calibration, calibration_labels, group,
            args.minimum_direction_class_count,
        )
        row["permutation_p"] = None
        results[group] = row
    for group in ("matching", "relation"):
        if group in results:
            control_name = f"shuffled_{group}"
            controls[group] = max(direct_auc, results[control_name]["calibration_auroc"])

    for group in ("rgb_role", "sar_role", "matching", "relation"):
        if group not in results:
            continue
        row = results[group]
        group_checks = {
            "auroc_ge_threshold": row["calibration_auroc"] >= args.minimum_auroc,
            "event_macro_ge_threshold": row["event_macro_auroc"] >= args.minimum_event_macro,
            "correct_events_ge_threshold": row["correct_events"] >= args.minimum_correct_events,
            "at_least_ten_evaluable_events": row["event_total"] >= args.minimum_correct_events,
            "positive_count_ge_threshold": row["positive"] >= args.minimum_class_count,
            "negative_count_ge_threshold": row["negative"] >= args.minimum_class_count,
            "beats_control_by_margin": row["calibration_auroc"] >= controls[group] + args.minimum_control_gain,
            "permutation_p_lt_0_05": row["permutation_p"] < 0.05,
        }
        checks[group] = group_checks
        if all(group_checks.values()):
            admitted.append(group)
    true_cross_modal = any(name in admitted for name in ("matching", "relation"))
    complementary_unimodal = "rgb_role" in admitted and "sar_role" in admitted
    decision = (
        "ADMIT_CLUSTER_E1_SINGLE_VERIFIER"
        if true_cross_modal or complementary_unimodal
        else "STOP_CLUSTER_EVIDENCE_NO_CROSS_EVENT_SIGNAL"
    )
    payload = {
        "protocol": "SPI_S2_CLUSTER_EVIDENCE_E0_AUDIT_v1",
        "configuration": vars(args),
        "train_cache_sha256": _sha256(args.train_cache),
        "calibration_cache_sha256": _sha256(args.calibration_cache),
        "oracle_json": str(Path(args.oracle_json).resolve()),
        "results": results, "controls": controls, "checks": checks,
        "admitted_groups": admitted, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for group, row in results.items():
        print(
            f"Cluster-E0 {group:20s} AUC={row['calibration_auroc']:.4f} "
            f"weighted={row['pixel_weighted_auroc']:.4f} event={row['event_macro_auroc']:.4f} "
            f"correct={row['correct_events']}/{row['event_total']} "
            f"p={'diagnostic' if row['permutation_p'] is None else format(row['permutation_p'], '.5f')}",
            flush=True,
        )
    print(f"Admitted: {admitted}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
