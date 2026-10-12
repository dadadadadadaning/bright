#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file

from . import PROTOCOL
from .build_memory import encode, load_encoder
from .core import retrieval_diverse_topk, retrieval_relative_gap


def main() -> None:
    parser = argparse.ArgumentParser("Freeze Direct and Top-k evidence for RAG training/evaluation")
    parser.add_argument("--records", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--shuffle_labels", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    if memory["metadata"]["encoder_sha256"] != sha256_file(args.encoder):
        raise RuntimeError("Memory and encoder do not match")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    query_keys, direct_logits = encode(records, checkpoint, device, args.batch_size)
    memory_keys = memory["keys"].to(device).float()
    memory_labels = memory["labels"].to(device).clone()
    if args.shuffle_labels:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        permutation = torch.randperm(memory_labels.numel(), generator=generator)
        memory_labels = memory_labels.cpu()[permutation].to(device)
    event_vocabulary = {event: index + 1 for index, event in enumerate(sorted(set(memory["events"])))}
    neighbour_keys, neighbour_labels, neighbour_geometry, neighbour_events = [], [], [], []
    posterior, predictions, margins, agreements, max_similarity = [], [], [], [], []
    relative_gap, density, source_count, source_entropy = [], [], [], []
    total = records["labels"].numel()
    for start in range(0, total, args.batch_size):
        stop = min(start + args.batch_size, total)
        query = query_keys[start:stop].to(device).float()
        result = retrieval_diverse_topk(
            query, memory_keys, memory_labels,
            query_images=records["images"][start:stop], memory_images=memory["images"],
            query_events=records["events"][start:stop], memory_events=memory["events"],
            exclude_same_image=args.exclude_same_image,
            exclude_same_event=args.exclude_same_event,
            topk=args.topk, max_per_event=args.max_per_event,
        )
        indices = result["indices"].cpu()
        neighbour_keys.append(memory["keys"][indices].half())
        neighbour_labels.append(memory_labels[result["indices"]].cpu())
        neighbour_geometry.append(memory["geometry"][indices].half())
        neighbour_events.append(torch.tensor([
            [event_vocabulary.get(str(memory["events"][int(value)]), 0) for value in row]
            for row in indices.tolist()
        ], dtype=torch.long))
        posterior.append(result["posterior"].cpu())
        predictions.append(result["prediction"].cpu())
        margins.append(result["margin"].cpu())
        agreements.append(result["agreement"].cpu())
        max_similarity.append(result["max_similarity"].cpu())
        relative_gap.append(retrieval_relative_gap(result, memory_labels).cpu())
        density.append(result["scores"].mean(1).cpu())
        source_count.append(result["source_event_count"].cpu())
        source_entropy.append(result["source_event_entropy"].cpu())
        if stop % 5000 < args.batch_size or stop == total:
            print(f"neighbours {stop}/{total}", flush=True)
    payload = {
        "protocol": PROTOCOL + "_NEIGHBOURS",
        "split": records["split"],
        "allowed_for_training": records["split"] == "train",
        "query_keys": query_keys.half(),
        "direct_logits": direct_logits.half(),
        "neighbour_keys": torch.cat(neighbour_keys),
        "neighbour_labels": torch.cat(neighbour_labels),
        "neighbour_geometry": torch.cat(neighbour_geometry),
        "neighbour_events": torch.cat(neighbour_events),
        "knn_posterior": torch.cat(posterior).half(),
        "knn_prediction": torch.cat(predictions),
        "margin": torch.cat(margins),
        "agreement": torch.cat(agreements),
        "max_similarity": torch.cat(max_similarity),
        "relative_gap": torch.cat(relative_gap),
        "density": torch.cat(density),
        "source_event_count": torch.cat(source_count),
        "source_event_entropy": torch.cat(source_entropy),
        "labels": records["labels"],
        "events": records["events"],
        "images": records["images"],
        "event_vocabulary": event_vocabulary,
        "metadata": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "encoder": str(Path(args.encoder).resolve()),
            "encoder_sha256": sha256_file(args.encoder),
            "memory": str(Path(args.memory).resolve()),
            "memory_sha256": sha256_file(args.memory),
            "topk": args.topk,
            "max_per_event": args.max_per_event,
            "exclude_same_image": args.exclude_same_image,
            "exclude_same_event": args.exclude_same_event,
            "shuffle_labels": args.shuffle_labels,
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

