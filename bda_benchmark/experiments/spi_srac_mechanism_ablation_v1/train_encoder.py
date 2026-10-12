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
    BalancedEventClassSampler, EvidencePooler, POOL_VARIANTS, SOURCE_VARIANTS,
    cross_event_supervised_contrastive, tokens_at,
)


def main() -> None:
    parser = argparse.ArgumentParser("Train region evidence pooler and retrieval Key")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_variant", choices=tuple(SOURCE_VARIANTS), required=True)
    parser.add_argument("--pool_variant", choices=POOL_VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--key_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ce_weight", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    if records.get("protocol") != PROTOCOL + "_RECORDS" or not records.get("allowed_for_training"):
        raise RuntimeError("encoder training requires explicit Train records")
    source_dims = {name: int(value.shape[-1]) for name, value in records["tokens"].items()}
    source_dims["enhanced_change"] = source_dims["base_change"]
    model = EvidencePooler(
        source_dims, args.source_variant, args.pool_variant,
        hidden_dim=args.hidden_dim, key_dim=args.key_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = BalancedEventClassSampler(records["labels"], records["events"], args.seed)
    running = defaultdict(float)
    for step in range(1, args.steps + 1):
        index = sampler.draw(args.batch_size)
        tokens, mask = tokens_at(records, index, device)
        labels = records["labels"][index].to(device)
        soft = records["soft_labels"][index].to(device).float()
        events = [records["events"][value] for value in index.tolist()]
        images = [records["images"][value] for value in index.tolist()]
        key, logits, diagnostics = model(tokens, mask)
        contrastive = cross_event_supervised_contrastive(
            key, labels, events, images, temperature=args.temperature
        )
        ce = -(soft * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        loss = contrastive + args.ce_weight * ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running["loss"] += float(loss.detach())
        running["supcon"] += float(contrastive.detach())
        running["ce"] += float(ce.detach())
        running["accuracy"] += float(((logits.argmax(1) + 1) == labels).float().mean())
        running["attention_entropy"] += float(diagnostics["attention_entropy"].mean())
        running["skip_scale"] += float(diagnostics["skip_scale"][0])
        if step % 100 == 0 or step == args.steps:
            count = 100 if step % 100 == 0 else step % 100
            print(json.dumps({"step": step, **{name: value / max(count, 1) for name, value in running.items()}}), flush=True)
            running.clear()
    payload = {
        "protocol": PROTOCOL + "_ENCODER",
        "source_variant": args.source_variant,
        "pool_variant": args.pool_variant,
        "source_dims": source_dims,
        "hidden_dim": args.hidden_dim,
        "key_dim": args.key_dim,
        "model": model.state_dict(),
        "records": str(Path(args.records).resolve()),
        "records_sha256": sha256_file(args.records),
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
