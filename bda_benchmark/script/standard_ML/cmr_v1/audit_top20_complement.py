# -*- coding: utf-8 -*-
"""CMR-V1A: quantify how much Raw-Mean Top-K can complement frozen P3.

This audit consumes existing CMR-P0 and CMR-M3 candidate caches only.  It does
not retrain P3, rebuild Memory, touch Test, or change any prediction.  The core
question is deliberately conditional on P3 errors:

    When P3 is wrong, is a GT-matching historical case already present in the
    Raw-Mean Top-K candidate set?

That separates candidate-recall failure from candidate-selection failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch


CLASS_NAMES = {0: "intact", 1: "damaged", 2: "destroyed"}


def _load_p0(path: str, role: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metadata", {}).get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("V1A requires the frozen CMR-P0 cache")
    if payload["metadata"].get("role") != role:
        raise RuntimeError(f"P0 role mismatch: {payload['metadata'].get('role')} != {role}")
    if bool(payload["metadata"].get("test_accessed", True)):
        raise RuntimeError("V1A refuses Test-derived P0 caches")
    return payload


def _load_candidates(path: str, role: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("V1A requires frozen Raw-Mean Top-K candidates")
    if payload.get("role") != role:
        raise RuntimeError(f"candidate role mismatch: {payload.get('role')} != {role}")
    if bool(payload.get("test_accessed", True)):
        raise RuntimeError("V1A refuses Test-derived candidates")
    return payload


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _first_correct_rank(candidate_labels: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """1-based rank of first GT-matching case; 0 means absent from coarse Top-K."""
    correct = candidate_labels.long() == truth.long()[:, None]
    has = correct.any(dim=1)
    first = correct.float().argmax(dim=1).long() + 1
    return torch.where(has, first, torch.zeros_like(first))


def _rank_summary(first_rank: torch.Tensor, mask: torch.Tensor, ks: Iterable[int]) -> Dict[str, object]:
    ranks = first_rank[mask]
    total = int(ranks.numel())
    present = ranks > 0
    present_ranks = ranks[present]
    result: Dict[str, object] = {
        "queries": total,
        "present_at_maxk": int(present.sum()),
        "absent_at_maxk": int((~present).sum()),
        "first_correct_rank_mean": (
            None if present_ranks.numel() == 0 else float(present_ranks.float().mean())
        ),
        "first_correct_rank_median": (
            None if present_ranks.numel() == 0 else float(present_ranks.float().median())
        ),
    }
    for k in ks:
        hit = int(((ranks > 0) & (ranks <= int(k))).sum())
        result[f"hit@{k}"] = hit
        result[f"recall@{k}"] = _safe_rate(hit, total)
    return result


def _base_top1_complement(
    truth: torch.Tensor, base: torch.Tensor, candidate_labels: torch.Tensor
) -> Dict[str, int | float]:
    top1 = candidate_labels[:, 0].long()
    base_ok = base == truth
    top1_ok = top1 == truth
    both = int((base_ok & top1_ok).sum())
    rescue = int((~base_ok & top1_ok).sum())
    harm = int((base_ok & ~top1_ok).sum())
    neither = int((~base_ok & ~top1_ok).sum())
    return {
        "p3_correct_top1_correct": both,
        "p3_wrong_top1_correct_rescue": rescue,
        "p3_correct_top1_wrong_harm": harm,
        "p3_wrong_top1_wrong": neither,
        "net_complement": rescue - harm,
    }


def _mean_or_none(values: torch.Tensor) -> float | None:
    return None if values.numel() == 0 else float(values.float().mean())


def _harm_exposure(
    candidate_labels: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    ks: Iterable[int],
) -> Dict[str, object]:
    """Wrong-class availability among P3-correct queries (potential harm)."""
    total = int(mask.sum())
    result: Dict[str, object] = {"queries": total}
    for k in ks:
        prefix = candidate_labels[:, : int(k)]
        any_wrong = (prefix != truth[:, None]).any(dim=1) & mask
        row: Dict[str, object] = {
            "any_wrong_class": int(any_wrong.sum()),
            "any_wrong_class_rate": _safe_rate(int(any_wrong.sum()), total),
        }
        for cls, name in CLASS_NAMES.items():
            exposed = (prefix == cls).any(dim=1) & (truth != cls) & mask
            row[f"contains_wrong_{name}"] = int(exposed.sum())
            row[f"contains_wrong_{name}_rate"] = _safe_rate(int(exposed.sum()), total)
        result[f"top@{k}"] = row
    return result


def _neighborhood_summary(
    labels: torch.Tensor,
    scores: torch.Tensor,
    candidate_events: torch.Tensor,
    truth: torch.Tensor,
    first_rank: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, object]:
    rows = torch.nonzero(mask, as_tuple=False).flatten()
    if rows.numel() == 0:
        return {"queries": 0}
    class_counts = torch.stack([(labels[rows] == cls).sum(dim=1) for cls in CLASS_NAMES], dim=1)
    correct = labels[rows] == truth[rows, None]
    wrong = ~correct
    neg_inf = scores.new_full(scores[rows].shape, float("-inf"))
    best_correct = torch.where(correct, scores[rows], neg_inf).max(dim=1).values
    best_wrong = torch.where(wrong, scores[rows], neg_inf).max(dim=1).values
    both = torch.isfinite(best_correct) & torch.isfinite(best_wrong)

    unique_events, max_event_fractions, correct_unique_events = [], [], []
    for local, row in enumerate(rows.tolist()):
        events = candidate_events[row]
        _, counts = torch.unique(events, return_counts=True)
        unique_events.append(int(counts.numel()))
        max_event_fractions.append(float(counts.max()) / float(events.numel()))
        correct_events = events[correct[local]]
        correct_unique_events.append(int(torch.unique(correct_events).numel()))
    return {
        "queries": int(rows.numel()),
        "mean_class_counts": {
            name: float(class_counts[:, cls].float().mean()) for cls, name in CLASS_NAMES.items()
        },
        "first_correct_rank_mean_including_absent_as_zero": _mean_or_none(first_rank[rows]),
        "best_correct_similarity_mean_when_present": _mean_or_none(best_correct[torch.isfinite(best_correct)]),
        "best_wrong_similarity_mean_when_present": _mean_or_none(best_wrong[torch.isfinite(best_wrong)]),
        "correct_vs_wrong_margin_mean_when_both_present": _mean_or_none(
            best_correct[both] - best_wrong[both]
        ),
        "unique_events_at_k_mean": float(torch.tensor(unique_events).float().mean()),
        "max_event_fraction_at_k_mean": float(torch.tensor(max_event_fractions).mean()),
        "correct_class_unique_events_at_k_mean": float(
            torch.tensor(correct_unique_events).float().mean()
        ),
    }


def analyse_split(p0: Mapping[str, object], candidates: Mapping[str, object]) -> Dict[str, object]:
    truth = p0["labels"].long().flatten().cpu()
    base = p0["base_predictions"].long().flatten().cpu()
    labels = candidates["candidate_labels"].long().cpu()
    if labels.shape[0] != truth.numel() or candidates["query_labels"].numel() != truth.numel():
        raise RuntimeError("P0/candidate rows do not align")
    if not torch.equal(candidates["query_labels"].long().flatten().cpu(), truth):
        raise RuntimeError("candidate GT labels differ from aligned P0 labels")

    max_k = int(labels.shape[1])
    ks = [k for k in (1, 3, 5, 10, 20) if k <= max_k]
    if max_k not in ks:
        ks.append(max_k)
        ks.sort()
    first_rank = _first_correct_rank(labels, truth)
    base_wrong = base != truth
    scores = candidates["candidate_scores"].float().cpu()
    indices = candidates["candidate_indices"].long().cpu()
    memory_events = candidates["memory_events"].long().flatten().cpu()
    if scores.shape != labels.shape or indices.shape != labels.shape:
        raise RuntimeError("candidate labels/scores/indices do not align")
    if bool((indices < 0).any()) or int(indices.max()) >= int(memory_events.numel()):
        raise RuntimeError("candidate cache contains an invalid memory index")
    candidate_events = memory_events[indices]

    per_class: Dict[str, object] = {}
    for cls, name in CLASS_NAMES.items():
        cls_mask = truth == cls
        wrong_mask = cls_mask & base_wrong
        top1_correct = labels[:, 0] == truth
        topmax_correct_exists = first_rank > 0
        unused = wrong_mask & topmax_correct_exists & ~top1_correct
        per_class[name] = {
            "all_queries": _rank_summary(first_rank, cls_mask, ks),
            "p3_wrong_queries": _rank_summary(first_rank, wrong_mask, ks),
            "p3_wrong_count": int(wrong_mask.sum()),
            "unused_headroom_at_maxk": int(unused.sum()),
            "unused_headroom_rate_over_p3_errors": _safe_rate(int(unused.sum()), int(wrong_mask.sum())),
        }

    # The main rescue route used by CMR-V0/M4A: GT Damaged but P3 says Intact.
    missed_damaged = (truth == 1) & (base == 0)
    damaged_in_candidates = labels == 1
    first_damaged = torch.where(
        damaged_in_candidates.any(dim=1),
        damaged_in_candidates.float().argmax(dim=1).long() + 1,
        torch.zeros(truth.numel(), dtype=torch.long),
    )
    top1_damaged = labels[:, 0] == 1
    has_damaged_maxk = first_damaged > 0
    unused_damaged = missed_damaged & has_damaged_maxk & ~top1_damaged

    event_rows: Dict[str, object] = {}
    events = list(map(str, p0["events"]))
    for event in sorted(set(events)):
        event_mask = torch.tensor([item == event for item in events], dtype=torch.bool)
        event_missed = event_mask & missed_damaged
        if not bool(event_missed.any()):
            continue
        event_rows[event] = _rank_summary(first_damaged, event_missed, ks)

    transitions: Dict[str, object] = {}
    for gt, gt_name in CLASS_NAMES.items():
        for pred, pred_name in CLASS_NAMES.items():
            transition_mask = (truth == gt) & (base == pred)
            key = f"{gt_name}_to_{pred_name}"
            transitions[key] = {
                "gt": gt_name,
                "p3_prediction": pred_name,
                **_rank_summary(first_rank, transition_mask, ks),
                "neighborhood": _neighborhood_summary(
                    labels, scores, candidate_events, truth, first_rank, transition_mask
                ),
            }

    base_correct = base == truth
    harm_per_class = {
        name: _harm_exposure(labels, truth, base_correct & (truth == cls), ks)
        for cls, name in CLASS_NAMES.items()
    }

    return {
        "queries": int(truth.numel()),
        "coarse_k": max_k,
        "p3_accuracy": float((base == truth).float().mean()),
        "p3_errors": int(base_wrong.sum()),
        "all_p3_errors": _rank_summary(first_rank, base_wrong, ks),
        "confusion_transitions_3x3": transitions,
        "per_class": per_class,
        "harm_exposure": {
            "definition": "wrong-class case exists in Top-K while frozen P3 is correct",
            "all_p3_correct": _harm_exposure(labels, truth, base_correct, ks),
            "per_gt_class": harm_per_class,
        },
        "neighborhood_diagnostics": {
            "all_queries": _neighborhood_summary(
                labels, scores, candidate_events, truth, first_rank,
                torch.ones_like(truth, dtype=torch.bool),
            ),
            "p3_wrong": _neighborhood_summary(
                labels, scores, candidate_events, truth, first_rank, base_wrong
            ),
            "p3_correct": _neighborhood_summary(
                labels, scores, candidate_events, truth, first_rank, base_correct
            ),
        },
        "p3_vs_raw_top1": _base_top1_complement(truth, base, labels),
        "missed_damaged_route": {
            "definition": "GT=Damaged and frozen-P3 prediction=Intact",
            "queries": int(missed_damaged.sum()),
            "damaged_case_recall": _rank_summary(first_damaged, missed_damaged, ks),
            "raw_top1_damaged": int((missed_damaged & top1_damaged).sum()),
            "has_damaged_in_maxk": int((missed_damaged & has_damaged_maxk).sum()),
            "unused_headroom_at_maxk": int(unused_damaged.sum()),
            "unused_headroom_rate_over_missed_damaged": _safe_rate(
                int(unused_damaged.sum()), int(missed_damaged.sum())
            ),
            "per_event": event_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-V1A P3 x Raw-Mean Top-K complement audit")
    parser.add_argument("--train_p0", required=True)
    parser.add_argument("--val_p0", required=True)
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_p0 = _load_p0(args.train_p0, "train")
    val_p0 = _load_p0(args.val_p0, "query")
    train_candidates = _load_candidates(args.train_candidates, "train")
    val_candidates = _load_candidates(args.val_candidates, "validation")
    if train_p0["metadata"].get("p3_checkpoint_sha256") != val_p0["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("Train/Validation P0 use different P3 checkpoints")

    payload = {
        "protocol": "CMR_V1_ISOLATED_A_p3_rawmean_topk_complement_v2",
        "purpose": "separate candidate recall from candidate selection under frozen P3",
        "train": analyse_split(train_p0, train_candidates),
        "validation": analyse_split(val_p0, val_candidates),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    row = payload["validation"]["missed_damaged_route"]
    print("=" * 104)
    print("CMR-V1A | P3 x Raw-Mean Top-K complement")
    print(f"Validation P3 errors              : {payload['validation']['p3_errors']}")
    print(f"P3-missed Damaged (D->I)          : {row['queries']}")
    for key, value in row["damaged_case_recall"].items():
        if key.startswith("recall@"):
            print(f"{key:<34}: {value}")
    print(f"Unused D headroom at Top-K        : {row['unused_headroom_at_maxk']}")
    print(f"Unused headroom / missed Damaged  : {row['unused_headroom_rate_over_missed_damaged']}")
    print("Validation off-diagonal transitions:")
    for name, transition in payload["validation"]["confusion_transitions_3x3"].items():
        if transition["gt"] != transition["p3_prediction"]:
            recalls = ", ".join(
                f"{key}={value:.4f}" for key, value in transition.items()
                if key.startswith("recall@") and value is not None
            )
            print(f"  {name:<24} N={transition['queries']:<5} {recalls}")
    print(f"Saved                             : {output}")
    print("Test accessed                     : False")


if __name__ == "__main__":
    main()
