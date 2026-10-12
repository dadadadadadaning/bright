# -*- coding: utf-8 -*-
"""CMR-P0: controlled frozen-P3 pooling audit.

Variants are selected with train-only leave-one-event-out retrieval.  The
locked winner is checked once on validation.  The script never modifies P3,
never fuses Memory into predictions, and refuses caches marked as test data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from model.cmr.pooling import CMR_KEY_VARIANTS
from script.standard_ML.rm_region_retrieval_runtime import (
    binary_auroc,
    class_balanced_knn,
    complement_audit,
    metrics_from_predictions,
    validate_disjoint_ids,
)


def load_cache(path: str, role: str) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"CMR cache must be a mapping: {path}")
    required = (
        "key_variants", "labels", "base_predictions", "base_probabilities",
        "events", "image_ids", "metadata",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"CMR cache {path} is missing: {missing}")
    metadata = payload["metadata"]
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("CMR cache protocol mismatch; rebuild the cache")
    if metadata.get("role") != role:
        raise RuntimeError(
            f"CMR cache role mismatch: {metadata.get('role')} != {role}"
        )
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("CMR-P0 refuses test-derived caches")
    observed = tuple(payload["key_variants"])
    if observed != CMR_KEY_VARIANTS:
        raise RuntimeError(f"CMR key variants changed: {observed}")
    return payload


def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float | None:
    scores = scores.float().flatten().cpu()
    targets = targets.bool().flatten().cpu()
    if scores.numel() != targets.numel() or int(targets.sum()) == 0:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ordered = targets[order].float()
    precision = ordered.cumsum(0) / torch.arange(
        1, ordered.numel() + 1, dtype=torch.float32
    )
    return float((precision * ordered).sum() / ordered.sum())


def rescue_harm_audit(
    truth: torch.Tensor,
    base_prediction: torch.Tensor,
    retrieval_prediction: torch.Tensor,
    retrieval_scores: torch.Tensor,
) -> Dict[str, object]:
    truth = truth.long().flatten().cpu()
    base = base_prediction.long().flatten().cpu()
    retrieval = retrieval_prediction.long().flatten().cpu()
    scores = retrieval_scores.float().cpu()
    proposed = (base == 0) & (retrieval == 1)
    auditable = proposed & ((truth == 0) | (truth == 1))
    indices = torch.nonzero(auditable, as_tuple=False).flatten()
    if indices.numel() == 0:
        return {
            "auditable_proposals": 0,
            "rescues": 0,
            "harms": 0,
            "rescue_prevalence": 0.0,
            "retrieval_margin_auroc": None,
            "retrieval_margin_auprc": None,
        }
    target = truth[indices] == 1
    selected = scores[indices]
    margin = selected[:, 1] - torch.maximum(selected[:, 0], selected[:, 2])
    auc_value = (
        binary_auroc(margin, target)
        if int(target.sum()) > 0 and int((~target).sum()) > 0
        else None
    )
    auc = (
        float(auc_value)
        if auc_value is not None and math.isfinite(float(auc_value))
        else None
    )
    return {
        "auditable_proposals": int(indices.numel()),
        "rescues": int(target.sum()),
        "harms": int((~target).sum()),
        "rescue_prevalence": float(target.float().mean()),
        "retrieval_margin_auroc": auc,
        "retrieval_margin_auprc": binary_average_precision(margin, target),
    }


def _event_delta(
    candidate: Mapping[str, object], base: Mapping[str, object]
) -> Dict[str, float]:
    return {
        event: 100.0 * (
            float(candidate["events"][event]["macro_f1"])
            - float(base["events"][event]["macro_f1"])
        )
        for event in sorted(set(candidate["events"]).intersection(base["events"]))
    }


def evaluate_variant(
    variant: str,
    memory_keys: torch.Tensor,
    memory_labels: torch.Tensor,
    memory_events: Sequence[str],
    query_keys: torch.Tensor,
    query: Mapping[str, object],
    *,
    topk: int,
    beta: float,
    max_neighbors_per_event: int,
    device: torch.device,
) -> Dict[str, object]:
    retrieval = class_balanced_knn(
        query_keys.to(device),
        memory_keys.to(device),
        memory_labels,
        topk_per_class=topk,
        beta=beta,
        query_events=query["events"],
        memory_events=memory_events,
        exclude_same_event=True,
        max_neighbors_per_event=max_neighbors_per_event,
    )
    metrics = metrics_from_predictions(
        query["labels"], retrieval["prediction"], query["events"]
    )
    base_metrics = metrics_from_predictions(
        query["labels"], query["base_predictions"], query["events"]
    )
    delta = _event_delta(metrics, base_metrics)
    return {
        "variant": variant,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "complement": complement_audit(
            query["labels"], query["base_predictions"], retrieval["prediction"]
        ),
        "rescue_harm": rescue_harm_audit(
            query["labels"],
            query["base_predictions"],
            retrieval["prediction"],
            retrieval["scores"],
        ),
        "event_macro_f1_delta_pp_vs_p3": delta,
        "nonnegative_events_vs_p3": sum(value >= 0.0 for value in delta.values()),
    }


def _selection_tuple(result: Mapping[str, object]) -> tuple[float, int, int, float, float]:
    raw_auc = result["rescue_harm"]["retrieval_margin_auroc"]
    auc = float(raw_auc) if raw_auc is not None else -1.0
    if not math.isfinite(auc):
        auc = -1.0
    return (
        auc,
        int(result["complement"]["per_class"]["damaged"]["net_complement"]),
        int(result["nonnegative_events_vs_p3"]),
        float(result["metrics"]["event_macro_f1"]),
        float(result["metrics"]["macro_f1"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser("Audit frozen P3 CMR pooling variants")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max_neighbors_per_event", type=int, default=1)
    args = parser.parse_args()

    train = load_cache(args.train_cache, "train")
    query = load_cache(args.query_cache, "query")
    validate_disjoint_ids(train["image_ids"], query["image_ids"])
    memory_indices = train.get("memory_indices")
    if memory_indices is None:
        raise KeyError("CMR train cache is missing class-event memory_indices")
    memory_indices = memory_indices.long()
    memory_labels = train["labels"][memory_indices]
    memory_events = [train["events"][i] for i in memory_indices.tolist()]
    device = torch.device(args.device)

    train_results: Dict[str, object] = {}
    validation_results: Dict[str, object] = {}
    for variant in CMR_KEY_VARIANTS:
        memory_keys = train["key_variants"][variant][memory_indices]
        train_results[variant] = evaluate_variant(
            variant,
            memory_keys,
            memory_labels,
            memory_events,
            train["key_variants"][variant],
            train,
            topk=args.topk,
            beta=args.beta,
            max_neighbors_per_event=args.max_neighbors_per_event,
            device=device,
        )
    selected = max(CMR_KEY_VARIANTS, key=lambda name: _selection_tuple(train_results[name]))

    # Report every frozen alternative for scientific diagnosis, but only the
    # train-locked winner participates in the advancement gate.
    for variant in CMR_KEY_VARIANTS:
        memory_keys = train["key_variants"][variant][memory_indices]
        validation_results[variant] = evaluate_variant(
            variant,
            memory_keys,
            memory_labels,
            memory_events,
            query["key_variants"][variant],
            query,
            topk=args.topk,
            beta=args.beta,
            max_neighbors_per_event=args.max_neighbors_per_event,
            device=device,
        )

    winner = validation_results[selected]
    baseline = validation_results["mean"]
    winner_raw_auc = winner["rescue_harm"]["retrieval_margin_auroc"]
    baseline_raw_auc = baseline["rescue_harm"]["retrieval_margin_auroc"]
    winner_auc = float(winner_raw_auc) if winner_raw_auc is not None else float("nan")
    baseline_auc = float(baseline_raw_auc) if baseline_raw_auc is not None else float("nan")
    checks = {
        "selected_variant_is_not_mean": selected != "mean",
        "rescue_harm_auroc_at_least_0_70": math.isfinite(winner_auc) and winner_auc >= 0.70,
        "rescue_harm_auroc_plus_0_05_vs_mean": (
            math.isfinite(winner_auc)
            and math.isfinite(baseline_auc)
            and winner_auc >= baseline_auc + 0.05
        ),
        "damaged_net_complement_positive": (
            winner["complement"]["per_class"]["damaged"]["net_complement"] > 0
        ),
        "ten_events_nondecreasing_vs_p3": winner["nonnegative_events_vs_p3"] >= 10,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "CMR_P0_frozen_pooling_audit_v1",
        "train_cache": str(Path(args.train_cache).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "selection_split": "train natural regions with leave-one-event-out retrieval",
        "selected_variant": selected,
        "selection_order": "rescue/harm AUROC -> damaged net -> nonnegative events -> event F1 -> macro F1",
        "train_results": train_results,
        "validation_results": validation_results,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_CMR_KEY_TO_REJECTOR_AUDIT_DO_NOT_FUSE"
            if passed
            else "STOP_OR_REVISE_POOLING_DO_NOT_FUSE"
        ),
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print("CMR-P0 frozen pooling audit")
    print(f"Selected on train : {selected}")
    for variant in CMR_KEY_VARIANTS:
        result = validation_results[variant]
        auc = result['rescue_harm']['retrieval_margin_auroc']
        ap = result['rescue_harm']['retrieval_margin_auprc']
        auc_text = "N/A" if auc is None else f"{auc:.4f}"
        ap_text = "N/A" if ap is None else f"{ap:.4f}"
        print(
            f"{variant:<18} | macro={100*result['metrics']['macro_f1']:.4f} | "
            f"event={100*result['metrics']['event_macro_f1']:.4f} | "
            f"D-net={result['complement']['per_class']['damaged']['net_complement']:+d} | "
            f"I-net={result['complement']['per_class']['intact']['net_complement']:+d} | "
            f"RH-AUC={auc_text} | RH-AP={ap_text}"
        )
    print(f"Checks            : {checks}")
    print(f"Decision          : {payload['decision']}")
    print(f"Saved             : {output}")
    print("Test accessed     : False")


if __name__ == "__main__":
    main()
