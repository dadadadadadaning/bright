#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    PROTOCOL,
    FactorizedKeyEncoder,
    groups_from_payload,
)


def encode_all(records: dict, checkpoint: dict, device: torch.device, batch_size: int) -> torch.Tensor:
    model = FactorizedKeyEncoder(
        checkpoint["group_dims"],
        checkpoint["variant"],
        key_dim=checkpoint["key_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    chunks = []
    with torch.inference_mode():
        for start in range(0, records["labels"].numel(), batch_size):
            stop = min(start + batch_size, records["labels"].numel())
            index = torch.arange(start, stop)
            key, _ = model(groups_from_payload(records, index, device))
            chunks.append(key.cpu())
    return torch.cat(chunks)


def farthest_subset(keys: torch.Tensor, candidates: list[int], count: int, seed: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return list(candidates)
    generator = torch.Generator().manual_seed(seed)
    if len(candidates) > 2048:
        order = torch.randperm(len(candidates), generator=generator)[:2048].tolist()
        pool = [candidates[index] for index in order]
    else:
        pool = list(candidates)
    value = keys[pool].float()
    center = value.mean(0, keepdim=True)
    first = int((value - center).square().sum(1).argmin())
    chosen_local = [first]
    distance = 1.0 - value @ value[first]
    while len(chosen_local) < min(count, len(pool)):
        current = int(distance.argmax())
        chosen_local.append(current)
        distance = torch.minimum(distance, 1.0 - value @ value[current])
    return [pool[index] for index in chosen_local]


def select_bucket(
    keys: torch.Tensor,
    indices: list[int],
    labels: torch.Tensor,
    p3_labels: torch.Tensor,
    cap: int,
    hard_fraction: float,
    seed: int,
) -> list[int]:
    target = int(labels[indices[0]])
    hard = [index for index in indices if int(p3_labels[index]) != target]
    representative = [index for index in indices if int(p3_labels[index]) == target]
    hard_count = min(len(hard), int(round(cap * hard_fraction)))
    chosen = farthest_subset(keys, hard, hard_count, seed + 17)
    remaining = min(cap - len(chosen), len(indices) - len(chosen))
    chosen.extend(farthest_subset(keys, representative, remaining, seed + 31))
    if len(chosen) < min(cap, len(indices)):
        unused = [index for index in indices if index not in set(chosen)]
        chosen.extend(farthest_subset(keys, unused, min(cap, len(indices)) - len(chosen), seed + 47))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser("Build a balanced and diverse factorized exemplar Memory")
    parser.add_argument("--records", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_class_event", type=int, default=128)
    parser.add_argument("--hard_fraction", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 <= args.hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0,1]")

    records = torch.load(args.records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.metric, map_location="cpu", weights_only=False)
    if checkpoint.get("records_sha256") != sha256_file(args.records):
        raise RuntimeError("metric Key was not trained from the supplied record cache")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    keys = encode_all(records, checkpoint, device, args.batch_size)

    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, (label, event) in enumerate(zip(records["labels"].tolist(), records["events"])):
        buckets[(int(label), str(event))].append(index)
    selected: list[int] = []
    bucket_counts = {}
    for order, ((label, event), indices) in enumerate(sorted(buckets.items())):
        chosen = select_bucket(
            keys,
            indices,
            records["labels"],
            records["p3_region_labels"],
            args.per_class_event,
            args.hard_fraction,
            args.seed + order * 101,
        )
        selected.extend(chosen)
        bucket_counts[f"{label}:{event}"] = len(chosen)
    index = torch.tensor(selected, dtype=torch.long)
    memory = {
        "protocol": PROTOCOL + "_MEMORY",
        "variant": checkpoint["variant"],
        "keys": keys[index].half(),
        "labels": records["labels"][index],
        "soft_labels": records["soft_labels"][index].half(),
        "geometry": records["groups"]["geometry"][index].half(),
        "quality": records["groups"]["quality"][index].half(),
        "p3_state": records["groups"]["p3_state"][index].half(),
        "p3_region_labels": records["p3_region_labels"][index],
        "events": [records["events"][value] for value in selected],
        "images": [records["images"][value] for value in selected],
        "region_ids": records["region_ids"][index],
        "component_ids": records["component_ids"][index],
        "source_indices": index,
        "metadata": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "metric": str(Path(args.metric).resolve()),
            "metric_sha256": sha256_file(args.metric),
            "selection": "70% representative farthest-point + up to 30% P3-hard exemplars per class/event",
            "per_class_event_cap": args.per_class_event,
            "hard_fraction": args.hard_fraction,
            "bucket_counts": bucket_counts,
            "memory_size": len(selected),
            "test_accessed": False,
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory, destination)
    print(f"Saved: {destination} entries={len(selected)} variant={checkpoint['variant']}")


if __name__ == "__main__":
    main()
