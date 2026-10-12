#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_building_relational_memory_v1.core import (
    PROTOCOL, BuildingRelationalEncoder, building_groups,
)
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import report


def load_model(checkpoint_path: str, device: torch.device) -> tuple[BuildingRelationalEncoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BuildingRelationalEncoder(
        checkpoint["group_dims"], checkpoint["variant"], checkpoint["node_dim"], checkpoint["key_dim"]
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


@torch.inference_mode()
def encode(model, records: dict, device: torch.device, batch_size: int):
    keys, logits, nodes = [], [], []
    for start in range(0, records["labels"].numel(), batch_size):
        stop = min(start + batch_size, records["labels"].numel())
        index = torch.arange(start, stop)
        key, logit, node = model(
            building_groups(records, index, device), records["mask"][index].to(device)
        )
        keys.append(key.cpu()); logits.append(logit.cpu()); nodes.append(node.cpu())
    return torch.cat(keys), torch.cat(logits), torch.cat(nodes)


def balanced_memory_indices(records: dict, per_bucket: int, seed: int) -> torch.Tensor:
    buckets = defaultdict(list)
    for index, (label, event, size) in enumerate(zip(
        records["labels"].tolist(), records["events"], records["size_bins"].tolist()
    )):
        buckets[(int(label), str(event), int(size))].append(index)
    rng = random.Random(seed); output = []
    for name in sorted(buckets):
        values = buckets[name][:]; rng.shuffle(values); output.extend(values[:per_bucket])
    return torch.tensor(sorted(output), dtype=torch.long)


def fine_similarity(query_nodes, query_mask, memory_nodes, memory_mask) -> float:
    left = query_nodes[query_mask].float()
    right = memory_nodes[memory_mask].float()
    if not left.numel() or not right.numel():
        return -1.0
    left = torch.nn.functional.normalize(left, dim=1)
    right = torch.nn.functional.normalize(right, dim=1)
    value = left @ right.t()
    return float(0.5 * (value.max(1).values.mean() + value.max(0).values.mean()))


def retrieve(
    query_keys: torch.Tensor,
    query_nodes: torch.Tensor,
    query_records: dict,
    memory_keys: torch.Tensor,
    memory_nodes: torch.Tensor,
    memory_records: dict,
    memory_index: torch.Tensor,
    topk: int,
    coarse_topr: int,
    max_per_event: int,
    exclude_same_event: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    bank_key = memory_keys[memory_index].float()
    bank_node = memory_nodes[memory_index]
    bank_mask = memory_records["mask"][memory_index]
    bank_label = memory_records["labels"][memory_index]
    bank_event = [memory_records["events"][value] for value in memory_index.tolist()]
    bank_image = [memory_records["images"][value] for value in memory_index.tolist()]
    predictions, margins, agreements, source_counts = [], [], [], []
    for row in range(query_keys.shape[0]):
        coarse = query_keys[row].float() @ bank_key.t()
        valid = torch.ones_like(coarse, dtype=torch.bool)
        for column in range(bank_key.shape[0]):
            valid[column] &= bank_image[column] != query_records["images"][row]
            if exclude_same_event:
                valid[column] &= bank_event[column] != query_records["events"][row]
        eligible = torch.nonzero(valid, as_tuple=False).flatten()
        if not eligible.numel():
            raise RuntimeError("no eligible cross-event Memory neighbour")
        take = min(coarse_topr, eligible.numel())
        candidate = eligible[coarse[eligible].topk(take).indices]
        scores = []
        for column in candidate.tolist():
            fine = fine_similarity(
                query_nodes[row], query_records["mask"][row], bank_node[column], bank_mask[column]
            )
            scores.append(0.50 * float(coarse[column]) + 0.50 * fine)
        order = torch.tensor(scores).argsort(descending=True).tolist()
        chosen, chosen_scores, per_event = [], [], defaultdict(int)
        for offset in order:
            column = int(candidate[offset])
            event = bank_event[column]
            if per_event[event] >= max_per_event:
                continue
            chosen.append(column); chosen_scores.append(scores[offset]); per_event[event] += 1
            if len(chosen) >= topk:
                break
        if not chosen:
            raise RuntimeError("diverse Memory selection produced no neighbours")
        weight = torch.softmax(torch.tensor(chosen_scores) / 0.10, dim=0)
        posterior = torch.zeros(3)
        labels = bank_label[torch.tensor(chosen)] - 1
        posterior.scatter_add_(0, labels, weight)
        top2 = posterior.topk(2).values
        prediction = int(posterior.argmax().item()) + 1
        predictions.append(prediction); margins.append(float(top2[0] - top2[1]))
        agreements.append(float((labels == prediction - 1).float().mean()))
        source_counts.append(len(per_event))
    return torch.tensor(predictions), {
        "mean_margin": sum(margins) / len(margins),
        "mean_agreement": sum(agreements) / len(agreements),
        "mean_source_event_count": sum(source_counts) / len(source_counts),
    }


def recovery(target: torch.Tensor, p3: torch.Tensor, prediction: torch.Tensor) -> dict[str, float | int]:
    wrong = p3 != target; correct = ~wrong
    rescued = wrong & (prediction == target)
    harmed = correct & (prediction != target)
    return {
        "p3_wrong_count": int(wrong.sum()),
        "recovered_count": int(rescued.sum()),
        "recovery_rate": float(rescued.sum().float() / wrong.sum().clamp_min(1)),
        "p3_correct_count": int(correct.sum()),
        "harmed_count": int(harmed.sum()),
        "harm_rate": float(harmed.sum().float() / correct.sum().clamp_min(1)),
        "net_correction_count": int(rescued.sum() - harmed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Cross-event building relation Memory audit")
    parser.add_argument("--train_records", required=True)
    parser.add_argument("--val_records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--per_class_event_size", type=int, default=64)
    parser.add_argument("--coarse_topr", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--shuffle_repeats", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for value in (args.train_records, args.val_records):
        if "test" in Path(value).name.lower():
            raise RuntimeError("Test access is forbidden in V1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_records = torch.load(args.train_records, map_location="cpu", weights_only=False)
    val_records = torch.load(args.val_records, map_location="cpu", weights_only=False)
    model, checkpoint = load_model(args.checkpoint, device)
    train_key, _, train_node = encode(model, train_records, device, args.batch_size)
    val_key, val_logits, val_node = encode(model, val_records, device, args.batch_size)
    direct_prediction = val_logits.argmax(1) + 1
    memory_index = balanced_memory_indices(train_records, args.per_class_event_size, args.seed)
    memory_prediction, diagnostics = retrieve(
        val_key, val_node, val_records, train_key, train_node, train_records, memory_index,
        args.topk, args.coarse_topr, args.max_per_event, args.exclude_same_event,
    )
    random_values = []
    generator = torch.Generator().manual_seed(args.seed)
    original = train_records["labels"].clone()
    for _ in range(args.shuffle_repeats):
        shuffled = original[torch.randperm(original.numel(), generator=generator)]
        proxy = dict(train_records); proxy["labels"] = shuffled
        prediction, _ = retrieve(
            val_key, val_node, val_records, train_key, train_node, proxy, memory_index,
            args.topk, args.coarse_topr, args.max_per_event, args.exclude_same_event,
        )
        random_values.append(report(val_records["labels"], prediction, val_records["events"])["macro_f1"])
    direct = report(val_records["labels"], direct_prediction, val_records["events"], include_event_class_metrics=True)
    memory = report(val_records["labels"], memory_prediction, val_records["events"], include_event_class_metrics=True)
    random_mean = sum(random_values) / len(random_values)
    payload = {
        "protocol": PROTOCOL + "_AUDIT",
        "variant": checkpoint["variant"],
        "direct": direct,
        "memory": memory,
        "memory_minus_direct_pp": 100.0 * (memory["macro_f1"] - direct["macro_f1"]),
        "correct_minus_random_pp": 100.0 * (memory["macro_f1"] - random_mean),
        "random_label_controls": {"values": random_values, "mean": random_mean},
        "p3_to_memory": recovery(val_records["labels"], val_records["p3_building_labels"], memory_prediction),
        "p3_to_direct": recovery(val_records["labels"], val_records["p3_building_labels"], direct_prediction),
        "retrieval": {
            **diagnostics,
            "memory_size": int(memory_index.numel()),
            "topk": args.topk,
            "coarse_topr": args.coarse_topr,
            "exclude_same_image": True,
            "exclude_same_event": args.exclude_same_event,
            "fine_score": "symmetric node-set cosine after relation encoding",
        },
        "test_accessed": False,
    }
    destination = Path(args.output_json); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2)); print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
