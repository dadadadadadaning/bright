#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import seed_everything, sha256_file

from . import PROTOCOL
from .core import BalancedEventClassSampler, RAG_VARIANTS, RetrievalAugmentedClassifier


def main() -> None:
    parser = argparse.ArgumentParser("Train retrieval-conditioned region classifier")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=RAG_VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seed_everything(args.seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("protocol") != PROTOCOL + "_NEIGHBOURS" or not cache.get("allowed_for_training"):
        raise RuntimeError("RAG training requires Train neighbour cache")
    if cache["metadata"].get("shuffle_labels"):
        raise RuntimeError("random-label cache is audit-only and cannot train the main RAG")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    key_dim = int(cache["query_keys"].shape[1])
    geometry_dim = int(cache["neighbour_geometry"].shape[2])
    model = RetrievalAugmentedClassifier(
        key_dim, args.variant, geometry_dim, len(cache["event_vocabulary"]),
        hidden_dim=args.hidden_dim, heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = BalancedEventClassSampler(cache["labels"], cache["events"], args.seed)
    running = defaultdict(float)
    for step in range(1, args.steps + 1):
        index = sampler.draw(args.batch_size)
        logits = model(
            cache["query_keys"][index].to(device).float(),
            cache["neighbour_keys"][index].to(device).float(),
            cache["neighbour_labels"][index].to(device),
            cache["neighbour_geometry"][index].to(device).float(),
            cache["neighbour_events"][index].to(device),
        )
        labels = cache["labels"][index].to(device) - 1
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running["loss"] += float(loss.detach())
        running["accuracy"] += float((logits.argmax(1) == labels).float().mean())
        if step % 100 == 0 or step == args.steps:
            count = 100 if step % 100 == 0 else step % 100
            print(json.dumps({"step": step, **{name: value / max(count, 1) for name, value in running.items()}}), flush=True)
            running.clear()
    payload = {
        "protocol": PROTOCOL + "_RAG",
        "variant": args.variant,
        "model": model.state_dict(),
        "key_dim": key_dim,
        "geometry_dim": geometry_dim,
        "event_count": len(cache["event_vocabulary"]),
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "cache": str(Path(args.cache).resolve()),
        "cache_sha256": sha256_file(args.cache),
        "steps": args.steps,
        "seed": args.seed,
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

