from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.audit_e0 import (
    permutation_p, policy_scores, select_direction_policy, weighted_auroc,
)
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import EVIDENCE_AUDIT_PROTOCOL, EVIDENCE_CACHE_PROTOCOL
from .core import AO_OUTCOME_CORRECTED, AO_OUTCOME_HARMED


MAIN_GROUPS = ("probability", "rgb_structure", "sar_texture", "instance_relative", "cross_relation")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(cache: dict) -> torch.Tensor:
    outcome = cache["outcome"].long()
    labels = torch.full_like(outcome, -1)
    labels[outcome == AO_OUTCOME_CORRECTED] = 1
    labels[outcome == AO_OUTCOME_HARMED] = 0
    return labels


def _events(cache: dict) -> list[str]:
    return [cache["image_events"][int(i)] for i in cache["image_index"].long()]


def _instance_auc(cache: dict, labels: torch.Tensor, scores: torch.Tensor) -> tuple[float, int, int]:
    valid = labels >= 0
    groups: dict[tuple[int, int], list[int]] = {}
    for index in torch.where(valid)[0].tolist():
        key = (int(cache["image_index"][index]), int(cache["region_label"][index]))
        groups.setdefault(key, []).append(index)
    targets, values = [], []
    for indices in groups.values():
        y = labels[indices]
        corrected, harmed = int((y == 1).sum()), int((y == 0).sum())
        if corrected == harmed:
            continue
        targets.append(int(corrected > harmed)); values.append(float(scores[indices].mean()))
    target = torch.tensor(targets); value = torch.tensor(values)
    positive, negative = int((target == 1).sum()), int((target == 0).sum())
    auc = binary_auroc(target, value) if positive and negative else float("nan")
    return auc, positive, negative


def _impact_weights(cache: dict, valid: torch.Tensor) -> torch.Tensor:
    """Balance transition directions so a frequent easy direction cannot dominate."""

    direction = cache["direction_code"].long()[valid]
    counts = torch.bincount(direction, minlength=4).float().clamp_min(1)
    weights = 1.0 / counts[direction]
    return weights / weights.mean()


def _evaluate(train: dict, calibration: dict, group: str, minimum_direction_count: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    train_labels = _labels(train); labels = _labels(calibration)
    train_events = _events(train); calibration_events = _events(calibration)
    scores = torch.empty(len(labels), dtype=torch.float32)
    policies = {}
    for event in sorted(set(calibration["image_events"])):
        eligible = torch.tensor([value != event for value in train_events])
        policy = select_direction_policy(train, train_labels, group, minimum_direction_count, eligible)
        event_mask = torch.tensor([value == event for value in calibration_events])
        scores[event_mask] = policy_scores(calibration, group, policy)[event_mask]
        policies[event] = policy
    valid = labels >= 0; target = labels[valid]; score = scores[valid]
    events = {}
    calibration_image = calibration["image_index"].long()[valid]
    for event in sorted(set(calibration["image_events"])):
        ids = torch.tensor([i for i, value in enumerate(calibration["image_events"]) if value == event])
        mask = torch.isin(calibration_image, ids); current = target[mask]
        if int((current == 0).sum()) and int((current == 1).sum()):
            events[event] = binary_auroc(current, score[mask])
    instance_auc, instance_positive, instance_negative = _instance_auc(calibration, labels, scores)
    aucs = list(events.values())
    return {
        "pixel_auroc": binary_auroc(target, score),
        "impact_weighted_auroc": weighted_auroc(target, score, _impact_weights(calibration, valid)),
        "instance_auroc": instance_auc, "instance_positive": instance_positive,
        "instance_negative": instance_negative,
        "event_macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "correct_events": sum(value > .5 for value in aucs), "event_total": len(aucs),
        "positive": int((target == 1).sum()), "negative": int((target == 0).sum()),
        "events": events, "direction_policy_by_heldout_event": policies,
    }, target, score


def _validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != EVIDENCE_CACHE_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"expected {EVIDENCE_CACHE_PROTOCOL} {split}")
    if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True):
        raise RuntimeError("AO-E1 cache accessed a forbidden split")
    if cache.get("official_instances_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError("AO-E1 cache accessed forbidden labels/test")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("AO-E1 parameter-free occurrence evidence audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--detail-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-pixel-auroc", type=float, default=.65)
    parser.add_argument("--minimum-instance-auroc", type=float, default=.60)
    parser.add_argument("--minimum-impact-auroc", type=float, default=.65)
    parser.add_argument("--minimum-event-macro", type=float, default=.60)
    parser.add_argument("--minimum-correct-events", type=int, default=10)
    parser.add_argument("--minimum-control-gain", type=float, default=.02)
    parser.add_argument("--minimum-class-count", type=int, default=200)
    parser.add_argument("--minimum-direction-class-count", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    detail = json.loads(Path(args.detail_json).read_text(encoding="utf-8"))
    if detail.get("decision") != "ADMIT_AO_E1_EVIDENCE_AUDIT":
        raise RuntimeError("AO-E1 refused: AO-D0 was not admitted")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    _validate(train, "meta_train"); _validate(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"]:
        raise RuntimeError("AO-E1 P3 fingerprints differ")
    if set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("AO-E1 train/calibration images overlap")
    results = {}
    for index, group in enumerate(MAIN_GROUPS + ("shuffled_relation",)):
        row, labels, scores = _evaluate(train, calibration, group, args.minimum_direction_class_count)
        row["permutation_p"] = permutation_p(labels, scores, row["pixel_auroc"], args.permutations, args.seed + index)
        results[group] = row
    probability_auc = results["probability"]["pixel_auroc"]
    controls = {
        "rgb_structure": probability_auc, "sar_texture": probability_auc,
        "instance_relative": probability_auc,
        "cross_relation": max(probability_auc, results["shuffled_relation"]["pixel_auroc"]),
    }
    checks, admitted = {}, []
    for group, control in controls.items():
        row = results[group]
        current = {
            "pixel_auroc_ge_threshold": row["pixel_auroc"] >= args.minimum_pixel_auroc,
            "instance_auroc_ge_threshold": row["instance_auroc"] >= args.minimum_instance_auroc,
            "impact_weighted_auroc_ge_threshold": row["impact_weighted_auroc"] >= args.minimum_impact_auroc,
            "event_macro_ge_threshold": row["event_macro_auroc"] >= args.minimum_event_macro,
            "correct_events_ge_threshold": row["correct_events"] >= args.minimum_correct_events,
            "positive_count_ge_threshold": row["positive"] >= args.minimum_class_count,
            "negative_count_ge_threshold": row["negative"] >= args.minimum_class_count,
            "beats_probability_or_shuffle_by_margin": row["pixel_auroc"] >= control + args.minimum_control_gain,
            "permutation_p_lt_0_05": row["permutation_p"] < .05,
        }
        checks[group] = current
        if all(current.values()): admitted.append(group)
    decision = "ADMIT_AO_E2_SINGLE_OCCURRENCE_HEAD" if admitted else "STOP_AO_E1_NO_DEPLOYABLE_CROSS_EVENT_EVIDENCE"
    payload = {
        "protocol": EVIDENCE_AUDIT_PROTOCOL, "configuration": vars(args),
        "train_cache_sha256": _sha256(args.train_cache), "calibration_cache_sha256": _sha256(args.calibration_cache),
        "results": results, "controls": controls, "checks": checks, "admitted_groups": admitted,
        "decision": decision, "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, row in results.items():
        print(f"AO-E1 {name:20s} pixel={row['pixel_auroc']:.4f} impact={row['impact_weighted_auroc']:.4f} instance={row['instance_auroc']:.4f} event={row['event_macro_auroc']:.4f} correct={row['correct_events']}/{row['event_total']} p={row['permutation_p']:.5f}")
    print(f"Admitted: {admitted}"); print(f"Decision: {decision}"); print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
