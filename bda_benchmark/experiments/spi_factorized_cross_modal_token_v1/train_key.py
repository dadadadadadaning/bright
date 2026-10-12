#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import seed_everything, sha256_file
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    PROTOCOL,
    VARIANTS,
    FactorizedKeyEncoder,
    cross_event_supervised_contrastive,
    groups_from_payload,
)


class BalancedEventClassSampler:
    """Draw every class equally, then draw an event equally within that class."""

    def __init__(self, labels: torch.Tensor, events: list[str], seed: int) -> None:
        self.rng = random.Random(seed)
        buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, (label, event) in enumerate(zip(labels.tolist(), events)):
            buckets[(int(label), str(event))].append(index)
        self.by_class: dict[int, dict[str, list[int]]] = {}
        for label in (1, 2, 3):
            current = {event: values for (cls, event), values in buckets.items() if cls == label}
            if not current:
                raise RuntimeError(f"class {label} has no training records")
            self.by_class[label] = current

    def draw(self, batch_size: int) -> torch.Tensor:
        output: list[int] = []
        for offset in range(batch_size):
            label = 1 + (offset % 3)
            event = self.rng.choice(sorted(self.by_class[label]))
            output.append(self.rng.choice(self.by_class[label][event]))
        self.rng.shuffle(output)
        return torch.tensor(output, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser("Train factorized cross-modal retrieval Key")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--key_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--soft_ce_weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 6:
        raise ValueError("steps must be positive and batch_size must be at least 6")

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    groups = records.get("groups")
    if not isinstance(groups, dict):
        raise RuntimeError("records do not contain factorized feature groups")
    group_dims = {name: int(value.shape[1]) for name, value in groups.items()}
    model = FactorizedKeyEncoder(
        group_dims, args.variant, key_dim=args.key_dim, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sampler = BalancedEventClassSampler(records["labels"], records["events"], args.seed)

    geometry = records["groups"]["geometry"].float()
    geometry_mean = geometry.mean(0)
    geometry_std = geometry.std(0, unbiased=False).clamp_min(1e-6)
    running = defaultdict(float)
    model.train()
    for step in range(1, args.steps + 1):
        indices = sampler.draw(args.batch_size)
        batch_groups = groups_from_payload(records, indices, device)
        labels = records["labels"][indices].to(device)
        soft = records["soft_labels"][indices].to(device).float()
        events = [records["events"][index] for index in indices.tolist()]
        images = [records["images"][index] for index in indices.tolist()]

        key, logits = model(batch_groups)
        contrastive = cross_event_supervised_contrastive(
            key, labels, events, images, temperature=args.temperature
        )
        soft_ce = -(soft * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        total = contrastive + args.soft_ce_weight * soft_ce
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running["total"] += float(total.detach())
        running["contrastive"] += float(contrastive.detach())
        running["soft_ce"] += float(soft_ce.detach())
        running["accuracy"] += float(((logits.argmax(1) + 1) == labels).float().mean())
        if step % 100 == 0 or step == args.steps:
            scale = 1.0 / (100 if step % 100 == 0 else step % 100)
            print(json.dumps({
                "step": step,
                "total_loss": running["total"] * scale,
                "cross_event_supcon": running["contrastive"] * scale,
                "soft_label_ce": running["soft_ce"] * scale,
                "batch_accuracy": running["accuracy"] * scale,
            }), flush=True)
            running.clear()

    contract = {
        "t0_concat": "all fields concatenated; intentionally confounded control",
        "t1_factorized": "RGB + SAR + unaligned relation only",
        "t2_aligned": "RGB + SAR + locally aligned relation only",
        "t3_multiring": "T2 + Core/Ring-8/Ring-16 relation evidence",
    }[args.variant]
    checkpoint = {
        "protocol": PROTOCOL + "_KEY",
        "variant": args.variant,
        "model": model.state_dict(),
        "group_dims": group_dims,
        "key_dim": args.key_dim,
        "hidden_dim": args.hidden_dim,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "records": str(Path(args.records).resolve()),
        "records_sha256": sha256_file(args.records),
        "steps": args.steps,
        "seed": args.seed,
        "training": {
            "sampler": "class-balanced then event-balanced",
            "loss": "cross-event supervised contrastive + soft-label CE",
            "temperature": args.temperature,
            "soft_ce_weight": args.soft_ce_weight,
        },
        "key_contract": contract,
        "p3_state_in_key": args.variant == "t0_concat",
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
