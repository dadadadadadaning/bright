#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import seed_everything, sha256_file
from bda_benchmark.experiments.spi_building_relational_memory_v1.core import (
    PROTOCOL, VARIANTS, BuildingRelationalEncoder, building_groups, cross_event_contrastive,
)


class BalancedSampler:
    """Balance damage class, event, and building-size group."""

    def __init__(self, labels: torch.Tensor, events: list[str], size_bins: torch.Tensor, seed: int) -> None:
        self.rng = random.Random(seed)
        self.buckets: dict[int, dict[tuple[str, int], list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, event, size) in enumerate(zip(labels.tolist(), events, size_bins.tolist())):
            self.buckets[int(label)][(str(event), int(size))].append(index)
        for label in (1, 2, 3):
            if not self.buckets[label]:
                raise RuntimeError(f"class {label} has no building records")

    def draw(self, count: int) -> torch.Tensor:
        output = []
        for offset in range(count):
            label = 1 + offset % 3
            bucket = self.rng.choice(sorted(self.buckets[label]))
            output.append(self.rng.choice(self.buckets[label][bucket]))
        self.rng.shuffle(output)
        return torch.tensor(output, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser("Train a building-level relational retrieval Key")
    parser.add_argument("--records", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--key_dim", type=int, default=128)
    parser.add_argument("--node_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ce_weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 6:
        raise ValueError("invalid training length or batch size")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    group_dims = {name: int(value.shape[-1]) for name, value in records["groups"].items()}
    model = BuildingRelationalEncoder(group_dims, args.variant, args.node_dim, args.key_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = BalancedSampler(records["labels"], records["events"], records["size_bins"], args.seed)
    running = defaultdict(float)
    model.train()
    for step in range(1, args.steps + 1):
        index = sampler.draw(args.batch_size)
        groups = building_groups(records, index, device)
        mask = records["mask"][index].to(device)
        labels = records["labels"][index].to(device)
        soft = records["soft_labels"][index].to(device).float()
        events = [records["events"][value] for value in index.tolist()]
        images = [records["images"][value] for value in index.tolist()]
        key, logits, _ = model(groups, mask)
        contrastive = cross_event_contrastive(key, labels, events, images, args.temperature)
        soft_ce = -(soft * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        total = contrastive + args.ce_weight * soft_ce
        optimizer.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        running["total"] += float(total.detach()); running["contrastive"] += float(contrastive.detach())
        running["soft_ce"] += float(soft_ce.detach())
        running["accuracy"] += float(((logits.argmax(1) + 1) == labels).float().mean())
        if step % 100 == 0 or step == args.steps:
            divisor = 100 if step % 100 == 0 else step % 100
            print(json.dumps({"step": step, **{name: value / divisor for name, value in running.items()}}), flush=True)
            running.clear()
    checkpoint = {
        "protocol": PROTOCOL + "_MODEL",
        "variant": args.variant,
        "model": model.state_dict(),
        "group_dims": group_dims,
        "node_dim": args.node_dim,
        "key_dim": args.key_dim,
        "steps": args.steps,
        "seed": args.seed,
        "records": str(Path(args.records).resolve()),
        "records_sha256": sha256_file(args.records),
        "training": {
            "sampling": "class x event x size balanced",
            "loss": "cross-event supervised contrastive + soft building-label CE",
            "quality_is_semantic_key_content": False,
            "p3_state_is_key_content": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination); print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
