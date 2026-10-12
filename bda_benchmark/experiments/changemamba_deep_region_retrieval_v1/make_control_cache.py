#!/usr/bin/env python3
"""Create audit-only negative controls without retraining the classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control", choices=("shuffled_labels", "wrong_labels", "random_neighbours"), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if payload.get("test_accessed") or payload.get("split") != "validation":
        raise RuntimeError("controls require a clean validation neighbour cache")
    generator = torch.Generator().manual_seed(args.seed)
    if args.control == "shuffled_labels":
        flat = payload["neighbour_labels"].flatten()
        payload["neighbour_labels"] = flat[torch.randperm(flat.numel(), generator=generator)].reshape_as(payload["neighbour_labels"])
    elif args.control == "wrong_labels":
        labels = payload["neighbour_labels"].clone()
        payload["neighbour_labels"] = labels.remainder(3) + 1
    else:
        count = payload["query_keys"].shape[0]
        permutation = torch.randperm(count, generator=generator)
        for key in ("neighbour_keys", "neighbour_labels", "neighbour_geometry", "neighbour_events"):
            payload[key] = payload[key][permutation]
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["negative_control"] = args.control
    payload["metadata"]["control_seed"] = args.seed
    payload["metadata"]["test_accessed"] = False
    payload["test_accessed"] = False
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved {args.control}: {destination}")


if __name__ == "__main__":
    main()

