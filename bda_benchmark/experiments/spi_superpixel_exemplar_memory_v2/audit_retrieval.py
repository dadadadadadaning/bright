#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import torch

from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import write_json
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import (
    PROTOCOL, MetricKeyEncoder, retrieve_global_topk,
)


CLASS_NAMES = ("intact", "damaged", "destroyed")


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    recall = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    precision = confusion.diag().float() / confusion.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy": float(confusion.diag().sum().float() / confusion.sum().clamp_min(1)),
        "macro_f1": float(f1.mean()),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "support": confusion.sum(1).tolist(),
        "confusion": confusion.tolist(),
    }


def report(
    target: torch.Tensor,
    prediction: torch.Tensor,
    events,
    *,
    include_event_class_metrics: bool = False,
) -> dict:
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for left, right in zip(target.tolist(), prediction.tolist()):
        confusion[left - 1, right - 1] += 1
    result = metrics_from_confusion(confusion)
    event_accuracy = {}
    event_class_metrics = {}
    for event in sorted(set(events)):
        mask = torch.tensor([value == event for value in events])
        event_accuracy[event] = float((prediction[mask] == target[mask]).float().mean().item())
        if include_event_class_metrics:
            event_confusion = torch.zeros(3, 3, dtype=torch.long)
            for left, right in zip(target[mask].tolist(), prediction[mask].tolist()):
                event_confusion[left - 1, right - 1] += 1
            detailed = metrics_from_confusion(event_confusion)
            detailed["class_names"] = list(CLASS_NAMES)
            event_class_metrics[event] = detailed
    result["class_names"] = list(CLASS_NAMES)
    result["event_accuracy"] = event_accuracy
    if include_event_class_metrics:
        result["event_class_metrics"] = event_class_metrics
    return result


def prediction_from_neighbours(
    indices: torch.Tensor,
    scores: torch.Tensor,
    memory_labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Re-label fixed retrieved neighbours without repeating similarity search."""
    neighbour_labels = memory_labels[indices].long() - 1
    weight = torch.softmax(scores / float(temperature), dim=1)
    posterior = scores.new_zeros((scores.shape[0], 3))
    posterior.scatter_add_(1, neighbour_labels, weight)
    return posterior.argmax(1) + 1


def main() -> None:
    parser = argparse.ArgumentParser("Bare retrieval audit before any pixel writeback")
    parser.add_argument("--query_records", required=True); parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True); parser.add_argument("--output_json", required=True)
    parser.add_argument("--exclude_same_image", action="store_true"); parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shuffle_repeats", type=int, default=10)
    args = parser.parse_args()
    if args.shuffle_repeats < 1:
        raise ValueError("shuffle_repeats must be at least one")
    query = torch.load(args.query_records, map_location="cpu", weights_only=False)
    metric = torch.load(args.metric, map_location="cpu", weights_only=False)
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MetricKeyEncoder(metric["input_dim"], metric["key_dim"]).to(device).eval()
    model.load_state_dict(metric["model"])
    memory_keys = memory["keys"].to(device)
    memory_labels = memory["labels"].to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    labels_cpu = memory["labels"].cpu()
    shuffled_label_sets = [
        labels_cpu[torch.randperm(labels_cpu.numel(), generator=generator)].to(device)
        for _ in range(args.shuffle_repeats)
    ]
    predictions, direct_predictions = [], []
    shuffled_predictions = [[] for _ in range(args.shuffle_repeats)]
    margins, agreements, similarities = [], [], []
    with torch.inference_mode():
        for start in range(0, query["descriptors"].shape[0], 512):
            stop = min(start + 512, query["descriptors"].shape[0])
            key, direct_logits = model(query["descriptors"][start:stop].to(device))
            common = dict(
                query=key, memory_keys=memory_keys, topk=args.topk,
                query_images=query["images"][start:stop], memory_images=memory["images"],
                query_events=query["events"][start:stop], memory_events=memory["events"],
                exclude_same_image=args.exclude_same_image, exclude_same_event=args.exclude_same_event,
            )
            result = retrieve_global_topk(memory_labels=memory_labels, **common)
            predictions.append(result["prediction"].cpu())
            direct_predictions.append((direct_logits.argmax(1) + 1).cpu())
            for repeat, shuffled_labels in enumerate(shuffled_label_sets):
                control_prediction = prediction_from_neighbours(
                    result["indices"], result["scores"], shuffled_labels
                )
                shuffled_predictions[repeat].append(control_prediction.cpu())
            margins.append(result["margin"].cpu()); agreements.append(result["agreement"].cpu())
            similarities.append(result["max_similarity"].cpu())
    prediction = torch.cat(predictions)
    direct_prediction = torch.cat(direct_predictions)
    target = query["labels"]
    correct = report(
        target, prediction, query["events"], include_event_class_metrics=True
    )
    direct = report(
        target, direct_prediction, query["events"], include_event_class_metrics=True
    )
    shuffled_reports = [
        report(target, torch.cat(values), query["events"])
        for values in shuffled_predictions
    ]
    shuffled_macro_f1 = torch.tensor([value["macro_f1"] for value in shuffled_reports])
    random_mean = float(shuffled_macro_f1.mean().item())
    random_std = float(shuffled_macro_f1.std(unbiased=False).item())
    payload = {
        "protocol": PROTOCOL + "_BARE_RETRIEVAL_AUDIT", "correct_memory": correct,
        "direct_metric_classifier": direct,
        "memory_minus_direct_classifier_macro_f1_pp": 100.0 * (
            correct["macro_f1"] - direct["macro_f1"]
        ),
        # Kept for compatibility: this is now the first genuinely random,
        # class-count-preserving permutation rather than a cyclic label shift.
        "shuffled_label_memory": shuffled_reports[0],
        "random_label_controls": {
            "method": "global class-count-preserving random permutation",
            "seed": args.seed,
            "repeats": args.shuffle_repeats,
            "macro_f1_values": [float(value) for value in shuffled_macro_f1.tolist()],
            "macro_f1_mean": random_mean,
            "macro_f1_std": random_std,
        },
        "correct_minus_shuffled_macro_f1_pp": 100.0 * (
            correct["macro_f1"] - shuffled_reports[0]["macro_f1"]
        ),
        "correct_minus_random_mean_macro_f1_pp": 100.0 * (
            correct["macro_f1"] - random_mean
        ),
        "mean_margin": float(torch.cat(margins).mean()), "mean_agreement": float(torch.cat(agreements).mean()),
        "mean_max_similarity": float(torch.cat(similarities).mean()),
        "exclude_same_image": args.exclude_same_image, "exclude_same_event": args.exclude_same_event,
        "query_count": int(target.numel()), "test_accessed": False,
    }
    write_json(args.output_json, payload); print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
