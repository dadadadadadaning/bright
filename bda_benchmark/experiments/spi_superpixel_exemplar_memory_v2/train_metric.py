#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import (
    PROTOCOL, MetricKeyEncoder, cross_event_supervised_contrastive,
)


def main() -> None:
    parser = argparse.ArgumentParser("Train the three-class cross-event superpixel Key encoder")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--key_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.20)
    parser.add_argument("--soft_label_weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    payload = torch.load(args.records, map_location="cpu", weights_only=False)
    x, y, soft = payload["descriptors"], payload["labels"], payload["soft_labels"]
    events, images = payload["events"], payload["images"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MetricKeyEncoder(x.shape[1], args.key_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    by_class = {c: torch.nonzero(y == c, as_tuple=False)[:, 0] for c in (1, 2, 3)}
    if any(index.numel() < 2 for index in by_class.values()):
        raise RuntimeError("every class needs at least two superpixel records")
    per_class = max(2, args.batch_size // 3)
    for step in range(1, args.steps + 1):
        chosen = []
        for c in (1, 2, 3):
            source = by_class[c]
            chosen.append(source[torch.randint(0, source.numel(), (per_class,))])
        index = torch.cat(chosen)
        descriptor = x[index].to(device); target = y[index].to(device) - 1
        soft_target = soft[index].to(device)
        key, logits = model(descriptor)
        hard_ce = F.cross_entropy(logits, target)
        soft_ce = -(soft_target * F.log_softmax(logits, dim=1)).sum(1).mean()
        selected_events = [events[int(i)] for i in index]
        selected_images = [images[int(i)] for i in index]
        contrastive = cross_event_supervised_contrastive(
            key, target, selected_events, selected_images,
        )
        loss = hard_ce + args.soft_label_weight * soft_ce + args.contrastive_weight * contrastive
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            accuracy = (logits.argmax(1) == target).float().mean()
            print(json.dumps({"step": step, "loss": float(loss.item()), "ce": float(hard_ce.item()),
                              "soft_ce": float(soft_ce.item()), "supcon": float(contrastive.item()),
                              "batch_accuracy": float(accuracy.item())}), flush=True)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": PROTOCOL + "_METRIC", "model": model.state_dict(),
        "input_dim": int(x.shape[1]), "key_dim": args.key_dim, "steps": args.steps,
        "records": str(Path(args.records).resolve()), "arguments": vars(args), "test_accessed": False,
    }, destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

