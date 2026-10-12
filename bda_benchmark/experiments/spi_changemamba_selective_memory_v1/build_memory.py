#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.build_memory import select_bucket

from . import PROTOCOL
from .core import ChangeMambaTokenEncoder, groups_at


def encode_all(records, checkpoint, device, batch_size):
    model = ChangeMambaTokenEncoder(
        checkpoint["group_dims"], checkpoint["variant"],
        checkpoint["key_dim"], checkpoint["hidden_dim"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    chunks = []
    with torch.inference_mode():
        for start in range(0, records["labels"].numel(), batch_size):
            stop = min(start + batch_size, records["labels"].numel())
            key, _ = model(groups_at(records, torch.arange(start, stop), device))
            chunks.append(key.cpu())
    return torch.cat(chunks)


def main() -> None:
    parser = argparse.ArgumentParser("Build class-event balanced ChangeMamba Memory")
    parser.add_argument("--records", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_class_event", type=int, default=128)
    parser.add_argument("--hard_fraction", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.metric, map_location="cpu", weights_only=False)
    if checkpoint.get("records_sha256") != sha256_file(args.records):
        raise RuntimeError("Key checkpoint and record cache do not match")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    keys = encode_all(records, checkpoint, device, args.batch_size)
    buckets = defaultdict(list)
    for index, (label, event) in enumerate(zip(records["labels"].tolist(), records["events"])):
        buckets[(int(label), str(event))].append(index)
    selected, counts = [], {}
    for order, ((label, event), indices) in enumerate(sorted(buckets.items())):
        chosen = select_bucket(
            keys, indices, records["labels"], records["p3_region_labels"],
            args.per_class_event, args.hard_fraction, args.seed + 101 * order,
        )
        selected.extend(chosen)
        counts[f"{label}:{event}"] = len(chosen)
    index = torch.tensor(selected, dtype=torch.long)
    payload = {
        "protocol": PROTOCOL + "_MEMORY", "variant": checkpoint["variant"],
        "keys": keys[index].half(), "labels": records["labels"][index],
        "soft_labels": records["soft_labels"][index].half(),
        "events": [records["events"][value] for value in selected],
        "images": [records["images"][value] for value in selected],
        "region_ids": records["region_ids"][index], "source_indices": index,
        "metadata": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "metric": str(Path(args.metric).resolve()),
            "metric_sha256": sha256_file(args.metric),
            "selection": "class-event balanced; representative plus P3-hard exemplars",
            "per_class_event": args.per_class_event,
            "hard_fraction": args.hard_fraction, "bucket_counts": counts,
            "memory_size": len(selected), "test_accessed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} entries={len(selected)}")


if __name__ == "__main__":
    main()
