#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

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
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import (
    prediction_from_neighbours,
    report,
)


def complementarity(target: torch.Tensor, p3: torch.Tensor, memory: torch.Tensor) -> dict:
    p3_correct = p3 == target
    memory_correct = memory == target
    counts = {
        "both_correct": int((p3_correct & memory_correct).sum()),
        "p3_correct_memory_wrong": int((p3_correct & ~memory_correct).sum()),
        "p3_wrong_memory_correct": int((~p3_correct & memory_correct).sum()),
        "both_wrong": int((~p3_correct & ~memory_correct).sum()),
    }
    p3_wrong = int((~p3_correct).sum())
    p3_right = int(p3_correct.sum())
    counts.update({
        "oracle_correction_coverage": counts["p3_wrong_memory_correct"] / max(p3_wrong, 1),
        "ungated_harm_rate": counts["p3_correct_memory_wrong"] / max(p3_right, 1),
        "p3_region_accuracy": float(p3_correct.float().mean()),
        "memory_region_accuracy": float(memory_correct.float().mean()),
        "memory_minus_p3_region_accuracy_pp": 100.0 * float(
            memory_correct.float().mean() - p3_correct.float().mean()
        ),
    })
    return counts


def main() -> None:
    parser = argparse.ArgumentParser("Bare K2 retrieval and complementarity audit")
    parser.add_argument("--query_records", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--geometry_weight", type=float, default=0.0)
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--shuffle_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.shuffle_repeats < 1:
        raise ValueError("shuffle_repeats must be at least one")

    query = torch.load(args.query_records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.metric, map_location="cpu", weights_only=False)
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    if memory.get("variant") != checkpoint.get("variant"):
        raise RuntimeError("Memory and metric Key variants do not match")
    if memory["metadata"].get("metric_sha256") != sha256_file(args.metric):
        raise RuntimeError("Memory was not built with this metric checkpoint")

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
    memory_geometry = standardize_geometry(memory["geometry"].to(device), geometry_mean, geometry_std)

    generator = torch.Generator().manual_seed(args.seed)
    shuffled_sets = [
        memory["labels"][torch.randperm(memory["labels"].numel(), generator=generator)].to(device)
        for _ in range(args.shuffle_repeats)
    ]
    predictions, direct_predictions = [], []
    random_predictions: list[list[torch.Tensor]] = [[] for _ in shuffled_sets]
    margins, agreements, similarities, source_counts, source_entropies = [], [], [], [], []
    with torch.inference_mode():
        total = query["labels"].numel()
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            index = torch.arange(start, stop)
            key, logits = model(groups_from_payload(query, index, device))
            query_geometry = standardize_geometry(
                query["groups"]["geometry"][index].to(device), geometry_mean, geometry_std
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
            predictions.append(result["prediction"].cpu())
            direct_predictions.append((logits.argmax(1) + 1).cpu())
            for repeat, random_labels in enumerate(shuffled_sets):
                random_predictions[repeat].append(
                    prediction_from_neighbours(
                        result["indices"], result["scores"], random_labels
                    ).cpu()
                )
            margins.append(result["margin"].cpu())
            agreements.append(result["agreement"].cpu())
            similarities.append(result["max_similarity"].cpu())
            source_counts.append(result["source_event_count"].cpu())
            source_entropies.append(result["source_event_entropy"].cpu())
            if stop % 5000 < args.batch_size or stop == total:
                print(f"audit {stop}/{total}", flush=True)

    target = query["labels"]
    prediction = torch.cat(predictions)
    direct_prediction = torch.cat(direct_predictions)
    correct = report(target, prediction, query["events"], include_event_class_metrics=True)
    direct = report(target, direct_prediction, query["events"], include_event_class_metrics=True)
    random_reports = [report(target, torch.cat(rows), query["events"]) for rows in random_predictions]
    random_values = torch.tensor([row["macro_f1"] for row in random_reports])
    payload = {
        "protocol": PROTOCOL + "_BARE_RETRIEVAL_AUDIT",
        "variant": checkpoint["variant"],
        "correct_memory": correct,
        "direct_metric_classifier": direct,
        "memory_minus_direct_macro_f1_pp": 100.0 * (correct["macro_f1"] - direct["macro_f1"]),
        "p3_memory_complementarity": complementarity(
            target, query["p3_region_labels"], prediction
        ),
        "random_label_controls": {
            "method": "class-count-preserving random permutation of fixed retrieved neighbours",
            "seed": args.seed,
            "repeats": args.shuffle_repeats,
            "macro_f1_values": random_values.tolist(),
            "macro_f1_mean": float(random_values.mean()),
            "macro_f1_std": float(random_values.std(unbiased=False)),
        },
        "correct_minus_random_macro_f1_pp": 100.0 * (
            correct["macro_f1"] - float(random_values.mean())
        ),
        "retrieval_diagnostics": {
            "mean_margin": float(torch.cat(margins).mean()),
            "mean_agreement": float(torch.cat(agreements).mean()),
            "mean_max_similarity": float(torch.cat(similarities).mean()),
            "mean_source_event_count": float(torch.cat(source_counts).float().mean()),
            "mean_source_event_entropy": float(torch.cat(source_entropies).mean()),
        },
        "contract": {
            "p3_state_in_key": bool(checkpoint["p3_state_in_key"]),
            "geometry_weight": args.geometry_weight,
            "quality_used_for_retrieval": False,
            "p3_state_reserved_for_future_verifier": checkpoint["variant"] != "t0_concat",
        },
        "exclude_same_image": args.exclude_same_image,
        "exclude_same_event": args.exclude_same_event,
        "query_count": int(target.numel()),
        "test_accessed": False,
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
