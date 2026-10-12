#!/usr/bin/env python3
"""Three-way P3 / T3 Direct / T3 Memory complementarity audit.

This audit never writes predictions back to pixels.  It measures whether the
Memory supplies correct evidence exactly where both P3 and the Direct head
fail, and exports per-token signals for a later Utility Verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import write_json
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    PROTOCOL,
    FactorizedKeyEncoder,
    groups_from_payload,
    retrieve_diverse_topk,
    standardize_geometry,
)
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import report


CLASS_NAMES = {0: "background", 1: "intact", 2: "damaged", 3: "destroyed"}


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def correctness_partition(
    target: torch.Tensor,
    p3: torch.Tensor,
    direct: torch.Tensor,
    memory: torch.Tensor,
) -> dict:
    """Return mutually exclusive correctness cells and incremental recovery."""
    p = p3 == target
    d = direct == target
    m = memory == target
    cube = {}
    for p_value in (False, True):
        for d_value in (False, True):
            for m_value in (False, True):
                name = (
                    f"p3_{'correct' if p_value else 'wrong'}__"
                    f"direct_{'correct' if d_value else 'wrong'}__"
                    f"memory_{'correct' if m_value else 'wrong'}"
                )
                cube[name] = int(((p == p_value) & (d == d_value) & (m == m_value)).sum())

    p3_wrong = ~p
    p3_correct = p
    direct_only = p3_wrong & d & ~m
    memory_only = p3_wrong & ~d & m
    both_rescue = p3_wrong & d & m
    neither = p3_wrong & ~d & ~m
    direct_remaining = p3_wrong & ~d
    memory_remaining = p3_wrong & ~m
    union = p3_wrong & (d | m)
    direct_harm = p3_correct & ~d
    memory_harm = p3_correct & ~m
    disagreement = direct != memory
    disagreement_count = int(disagreement.sum())
    return {
        "total": int(target.numel()),
        "correctness_cube": cube,
        "p3_region_accuracy": float(p.float().mean()),
        "direct_region_accuracy": float(d.float().mean()),
        "memory_region_accuracy": float(m.float().mean()),
        "p3_wrong": {
            "total": int(p3_wrong.sum()),
            "direct_correct": int((p3_wrong & d).sum()),
            "memory_correct": int((p3_wrong & m).sum()),
            "both_correct": int(both_rescue.sum()),
            "direct_only_correct": int(direct_only.sum()),
            "memory_only_correct": int(memory_only.sum()),
            "neither_correct": int(neither.sum()),
            "direct_recovery_rate": _ratio(int((p3_wrong & d).sum()), int(p3_wrong.sum())),
            "memory_recovery_rate": _ratio(int((p3_wrong & m).sum()), int(p3_wrong.sum())),
            "oracle_union_recovery_rate": _ratio(int(union.sum()), int(p3_wrong.sum())),
            "memory_incremental_over_direct_rate": _ratio(
                int(memory_only.sum()), int(direct_remaining.sum())
            ),
            "direct_incremental_over_memory_rate": _ratio(
                int(direct_only.sum()), int(memory_remaining.sum())
            ),
        },
        "p3_correct": {
            "total": int(p3_correct.sum()),
            "direct_wrong_potential_harm": int(direct_harm.sum()),
            "memory_wrong_potential_harm": int(memory_harm.sum()),
            "direct_ungated_harm_rate": _ratio(int(direct_harm.sum()), int(p3_correct.sum())),
            "memory_ungated_harm_rate": _ratio(int(memory_harm.sum()), int(p3_correct.sum())),
        },
        "direct_memory_disagreement": {
            "total": disagreement_count,
            "fraction": _ratio(disagreement_count, int(target.numel())),
            "direct_correct": int((disagreement & d).sum()),
            "memory_correct": int((disagreement & m).sum()),
            "both_wrong": int((disagreement & ~d & ~m).sum()),
            "direct_win_rate": _ratio(int((disagreement & d).sum()), disagreement_count),
            "memory_win_rate": _ratio(int((disagreement & m).sum()), disagreement_count),
        },
        "oracle_region_accuracy": {
            "p3_or_direct": float((p | d).float().mean()),
            "p3_or_memory": float((p | m).float().mean()),
            "p3_or_direct_or_memory": float((p | d | m).float().mean()),
        },
    }


def direction_report(source: torch.Tensor, proposal: torch.Tensor, target: torch.Tensor) -> dict:
    rows = {}
    changed = source != proposal
    for left in range(4):
        for right in (1, 2, 3):
            mask = changed & (source == left) & (proposal == right)
            count = int(mask.sum())
            if count == 0:
                continue
            proposal_correct = int((proposal[mask] == target[mask]).sum())
            source_correct = int((source[mask] == target[mask]).sum())
            target_counts = {
                CLASS_NAMES[label]: int((target[mask] == label).sum())
                for label in (1, 2, 3)
            }
            rows[f"{CLASS_NAMES[left]}_to_{CLASS_NAMES[right]}"] = {
                "count": count,
                "proposal_correct_rate": _ratio(proposal_correct, count),
                "source_correct_rate": _ratio(source_correct, count),
                "net_correct_delta": proposal_correct - source_correct,
                "target_distribution": target_counts,
            }
    return {
        "changed_count": int(changed.sum()),
        "changed_fraction": _ratio(int(changed.sum()), int(target.numel())),
        "directions": rows,
    }


def signal_summary(mask: torch.Tensor, signals: Mapping[str, torch.Tensor]) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "means": {name: None for name in signals}}
    return {
        "count": count,
        "means": {name: float(value[mask].float().mean()) for name, value in signals.items()},
    }


def confidence_groups(
    target: torch.Tensor,
    p3: torch.Tensor,
    direct: torch.Tensor,
    memory: torch.Tensor,
    signals: Mapping[str, torch.Tensor],
) -> dict:
    p = p3 == target
    d = direct == target
    m = memory == target
    masks = {
        "memory_unique_rescue": ~p & ~d & m,
        "direct_unique_rescue": ~p & d & ~m,
        "both_rescue": ~p & d & m,
        "neither_rescues": ~p & ~d & ~m,
        "memory_harms_p3_correct": p & ~m,
        "direct_harms_p3_correct": p & ~d,
        "direct_memory_disagree_direct_wins": (direct != memory) & d,
        "direct_memory_disagree_memory_wins": (direct != memory) & m,
        "direct_memory_disagree_both_wrong": (direct != memory) & ~d & ~m,
    }
    return {name: signal_summary(mask, signals) for name, mask in masks.items()}


def subset_report(
    mask: torch.Tensor,
    target: torch.Tensor,
    p3: torch.Tensor,
    direct: torch.Tensor,
    memory: torch.Tensor,
) -> dict:
    if not bool(mask.any()):
        return {"count": 0}
    result = correctness_partition(target[mask], p3[mask], direct[mask], memory[mask])
    # The full correctness cube is available globally; keep event/class rows compact.
    result.pop("correctness_cube", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Audit complementarity of P3, T3 Direct and T3 Memory")
    parser.add_argument("--query_records", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--geometry_weight", type=float, default=0.0)
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    query = torch.load(args.query_records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.metric, map_location="cpu", weights_only=False)
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "t3_multiring":
        raise RuntimeError("this audit requires the t3_multiring metric checkpoint")
    if memory.get("variant") != "t3_multiring":
        raise RuntimeError("this audit requires the t3_multiring Memory")
    if memory["metadata"].get("metric_sha256") != sha256_file(args.metric):
        raise RuntimeError("Memory was not built with the supplied T3 metric")
    if "test" in str(query.get("metadata", {}).get("source_list", "")).lower():
        raise RuntimeError("complementarity audit forbids test-list access")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FactorizedKeyEncoder(
        checkpoint["group_dims"], checkpoint["variant"],
        key_dim=checkpoint["key_dim"], hidden_dim=checkpoint["hidden_dim"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    memory_keys = memory["keys"].to(device).float()
    memory_labels = memory["labels"].to(device)
    geometry_mean = checkpoint["geometry_mean"].to(device)
    geometry_std = checkpoint["geometry_std"].to(device)
    memory_geometry = standardize_geometry(
        memory["geometry"].to(device), geometry_mean, geometry_std
    )

    direct_probabilities, memory_posteriors = [], []
    direct_predictions, memory_predictions = [], []
    memory_margins, agreements, similarities = [], [], []
    source_counts, source_entropies = [], []
    total = int(query["labels"].numel())
    with torch.inference_mode():
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            indices = torch.arange(start, stop)
            key, logits = model(groups_from_payload(query, indices, device))
            direct_probability = torch.softmax(logits.float(), dim=1)
            query_geometry = standardize_geometry(
                query["groups"]["geometry"][indices].to(device), geometry_mean, geometry_std
            )
            result = retrieve_diverse_topk(
                key,
                memory_keys,
                memory_labels,
                query_geometry=query_geometry,
                memory_geometry=memory_geometry,
                geometry_weight=args.geometry_weight,
                query_images=query["images"][start:stop],
                memory_images=memory["images"],
                query_events=query["events"][start:stop],
                memory_events=memory["events"],
                exclude_same_image=args.exclude_same_image,
                exclude_same_event=args.exclude_same_event,
                topk=args.topk,
                max_per_event=args.max_per_event,
            )
            direct_probabilities.append(direct_probability.cpu())
            memory_posteriors.append(result["posterior"].cpu())
            direct_predictions.append((direct_probability.argmax(1) + 1).cpu())
            memory_predictions.append(result["prediction"].cpu())
            memory_margins.append(result["margin"].cpu())
            agreements.append(result["agreement"].cpu())
            similarities.append(result["max_similarity"].cpu())
            source_counts.append(result["source_event_count"].cpu())
            source_entropies.append(result["source_event_entropy"].cpu())
            if stop % 5000 < args.batch_size or stop == total:
                print(f"complementarity {stop}/{total}", flush=True)

    target = query["labels"].long()
    p3 = query["p3_region_labels"].long()
    direct_probability = torch.cat(direct_probabilities)
    memory_posterior = torch.cat(memory_posteriors)
    direct = torch.cat(direct_predictions)
    memory_prediction = torch.cat(memory_predictions)
    direct_top2 = direct_probability.topk(2, dim=1).values
    p3_state = query["groups"]["p3_state"].float()
    quality = query["groups"]["quality"].float()
    signals = {
        "p3_mean_probability_max": p3_state[:, 0:4].max(1).values,
        "p3_entropy_mean": p3_state[:, 8],
        "p3_occurrence_margin_mean": p3_state[:, 11],
        "direct_confidence": direct_top2[:, 0],
        "direct_margin": direct_top2[:, 0] - direct_top2[:, 1],
        "memory_margin": torch.cat(memory_margins),
        "memory_agreement": torch.cat(agreements),
        "memory_max_similarity": torch.cat(similarities),
        "memory_source_event_count": torch.cat(source_counts).float(),
        "memory_source_event_entropy": torch.cat(source_entropies),
        "alignment_entropy_mean": quality[:, 0],
        "alignment_offset_mean": quality[:, 2],
        "alignment_match_margin_mean": quality[:, 7],
        "building_probability_mean": quality[:, 8],
    }

    global_partition = correctness_partition(target, p3, direct, memory_prediction)
    events = {}
    for event in sorted(set(query["events"])):
        mask = torch.tensor([str(value) == event for value in query["events"]])
        events[event] = subset_report(mask, target, p3, direct, memory_prediction)
    classes = {
        CLASS_NAMES[label]: subset_report(
            target == label, target, p3, direct, memory_prediction
        )
        for label in (1, 2, 3)
    }
    payload = {
        "protocol": PROTOCOL + "_T3_DIRECT_MEMORY_COMPLEMENTARITY",
        "variant": "t3_multiring",
        "overall_metrics": {
            "direct": report(target, direct, query["events"], include_event_class_metrics=True),
            "memory": report(
                target, memory_prediction, query["events"], include_event_class_metrics=True
            ),
            "p3_region_accuracy": global_partition["p3_region_accuracy"],
        },
        "complementarity": global_partition,
        "confidence_group_diagnostics": confidence_groups(
            target, p3, direct, memory_prediction, signals
        ),
        "proposal_directions": {
            "direct_vs_p3": direction_report(p3, direct, target),
            "memory_vs_p3": direction_report(p3, memory_prediction, target),
            "memory_vs_direct": direction_report(direct, memory_prediction, target),
        },
        "per_class": classes,
        "per_event": events,
        "contract": {
            "estimand": "region-level complementarity; not pixel-mIoU",
            "memory_is_incremental_only_if_memory_only_correct_is_non-trivial": True,
            "no_gt_is_used_for_prediction": True,
            "gt_is_used_only_for_offline_audit": True,
            "exclude_same_image": args.exclude_same_image,
            "exclude_same_event": args.exclude_same_event,
            "geometry_weight": args.geometry_weight,
        },
        "query_count": total,
        "detail_pt": str(Path(args.output_pt).resolve()),
        "test_accessed": False,
    }
    detail = {
        "protocol": payload["protocol"] + "_DETAIL",
        "target": target,
        "p3_prediction": p3,
        "direct_prediction": direct,
        "memory_prediction": memory_prediction,
        "direct_probability": direct_probability.half(),
        "memory_posterior": memory_posterior.half(),
        "signals": {name: value.half() for name, value in signals.items()},
        "direct_utility_target": ((direct == target).long() - (p3 == target).long()),
        "memory_utility_target": (
            (memory_prediction == target).long() - (p3 == target).long()
        ),
        "memory_residual_over_direct_target": (
            (memory_prediction == target).long() - (direct == target).long()
        ),
        "events": list(query["events"]),
        "images": list(query["images"]),
        "region_ids": query["region_ids"],
        "component_ids": query["component_ids"],
        "metadata": {
            "query_records": str(Path(args.query_records).resolve()),
            "query_records_sha256": sha256_file(args.query_records),
            "metric": str(Path(args.metric).resolve()),
            "metric_sha256": sha256_file(args.metric),
            "memory": str(Path(args.memory).resolve()),
            "memory_sha256": sha256_file(args.memory),
            "test_accessed": False,
        },
    }
    detail_path = Path(args.output_pt)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detail, detail_path)
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved detail: {detail_path}")


if __name__ == "__main__":
    main()
