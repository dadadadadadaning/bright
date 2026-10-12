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
from .core import (
    VARIANTS, BalancedEventClassSampler, ChangeMambaTokenEncoder,
    cross_event_supervised_contrastive, groups_at,
)


def main() -> None:
    parser = argparse.ArgumentParser("Train ChangeMamba region retrieval Key")
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
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    if records.get("protocol") != PROTOCOL + "_RECORDS" or records.get("split") != "train":
        raise RuntimeError("Key training requires the explicit Train record cache")
    group_dims = {name: int(value.shape[1]) for name, value in records["groups"].items()}
    model = ChangeMambaTokenEncoder(
        group_dims, args.variant, key_dim=args.key_dim, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = BalancedEventClassSampler(records["labels"], records["events"], args.seed)
    running = defaultdict(float)
    model.train()
    for step in range(1, args.steps + 1):
        indices = sampler.draw(args.batch_size)
        groups = groups_at(records, indices, device)
        labels = records["labels"][indices].to(device)
        soft = records["soft_labels"][indices].to(device).float()
        events = [records["events"][index] for index in indices.tolist()]
        images = [records["images"][index] for index in indices.tolist()]
        key, logits = model(groups)
        contrastive = cross_event_supervised_contrastive(
            key, labels, events, images, temperature=args.temperature
        )
        soft_ce = -(soft * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        loss = contrastive + args.soft_ce_weight * soft_ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running["loss"] += float(loss.detach())
        running["supcon"] += float(contrastive.detach())
        running["soft_ce"] += float(soft_ce.detach())
        running["accuracy"] += float(((logits.argmax(1) + 1) == labels).float().mean())
        if step % 100 == 0 or step == args.steps:
            denominator = 100 if step % 100 == 0 else step % 100
            print(json.dumps({"step": step, **{
                name: value / max(denominator, 1) for name, value in running.items()
            }}), flush=True)
            running.clear()
    payload = {
        "protocol": PROTOCOL + "_KEY", "variant": args.variant,
        "model": model.state_dict(), "group_dims": group_dims,
        "key_dim": args.key_dim, "hidden_dim": args.hidden_dim,
        "records": str(Path(args.records).resolve()),
        "records_sha256": sha256_file(args.records),
        "steps": args.steps, "seed": args.seed,
        "key_contract": "ChangeMamba pre-head content only; P3 state reserved for verifier",
        "p3_state_in_key": False, "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

