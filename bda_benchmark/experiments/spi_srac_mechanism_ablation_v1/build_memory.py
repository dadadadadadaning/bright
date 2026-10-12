#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.build_memory import select_bucket

from . import PROTOCOL
from .core import EvidencePooler, tokens_at


def load_encoder(checkpoint, device):
    model = EvidencePooler(
        checkpoint["source_dims"], checkpoint["source_variant"], checkpoint["pool_variant"],
        checkpoint["hidden_dim"], checkpoint["key_dim"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@torch.inference_mode()
def encode(records, checkpoint, device, batch_size):
    model = load_encoder(checkpoint, device)
    keys, logits = [], []
    for start in range(0, records["labels"].numel(), batch_size):
        stop = min(start + batch_size, records["labels"].numel())
        index = torch.arange(start, stop)
        token, mask = tokens_at(records, index, device)
        key, direct, _ = model(token, mask)
        keys.append(key.cpu())
        logits.append(direct.cpu())
    return torch.cat(keys), torch.cat(logits)


def main() -> None:
    parser = argparse.ArgumentParser("Build class-event balanced B2 region Memory")
    parser.add_argument("--records", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_class_event", type=int, default=128)
    parser.add_argument("--hard_fraction", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    if checkpoint.get("records_sha256") != sha256_file(args.records):
        raise RuntimeError("encoder and Train records do not match")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    keys, direct_logits = encode(records, checkpoint, device, args.batch_size)
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
        "protocol": PROTOCOL + "_MEMORY",
        "keys": keys[index].half(),
        "direct_logits": direct_logits[index].half(),
        "labels": records["labels"][index],
        "soft_labels": records["soft_labels"][index].half(),
        "geometry": records["geometry"][index].half(),
        "events": [records["events"][value] for value in selected],
        "images": [records["images"][value] for value in selected],
        "region_ids": records["region_ids"][index],
        "source_indices": index,
        "metadata": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "encoder": str(Path(args.encoder).resolve()),
            "encoder_sha256": sha256_file(args.encoder),
            "per_class_event": args.per_class_event,
            "hard_fraction": args.hard_fraction,
            "bucket_counts": counts,
            "memory_size": len(selected),
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} entries={len(selected)}")


if __name__ == "__main__":
    main()
