#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.experiments.spi_aligned_region_memory_v1.core import (
    PROTOCOL, SoftRegionTokenizer, learned_region_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser("Train RGB-only differentiable region tokenizer")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.train_list).name.lower() or "val" in Path(args.train_list).name.lower():
        raise RuntimeError("learned region tokenizer may only be trained from train_set")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path, args.train_list, 640,
        max_iters=args.steps * args.batch_size, split="train", variant="h0",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = SoftRegionTokenizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        rgb, _, location = batch[:3]
        rgb, location = rgb.to(device).float(), location.to(device).long()
        assignment, feature = model(rgb, (160, 160))
        loss, parts = learned_region_loss(assignment, feature, location)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step % 100 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              **{k: float(v.detach()) for k, v in parts.items()}}), flush=True)
    payload = {
        "protocol": PROTOCOL + "_LEARNED_REGION",
        "model": model.state_dict(), "steps": args.steps, "seed": args.seed,
        "input_contract": "pre-disaster RGB only",
        "training_supervision": "train RGB reconstruction + train building-boundary preservation",
        "post_sar_used": False, "p3_class_probability_used": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output); print(f"Saved: {output}")


if __name__ == "__main__": main()

