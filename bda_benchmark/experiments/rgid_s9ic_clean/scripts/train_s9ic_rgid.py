#!/usr/bin/env python3
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from rgid import RGIDConfig, RGIDHead
from rgid.training import train_rgid_head


def build_train_loader(args) -> DataLoader:
    """Build the project adapter.

    Each batch must be:
    {
        "fused_roi": FloatTensor[B,256,Hh,Wh],
        "instance_ids": LongTensor[B,H,W],
    }
    """

    raise NotImplementedError(
        "Connect the frozen S8 fused_roi and GT-derived pseudo-instance loader."
    )


def main() -> None:
    parser = argparse.ArgumentParser("Train S9-Ic RGID head")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    config = RGIDConfig()
    train_rgid_head(
        head=RGIDHead(
            config.in_channels,
            config.hidden_channels,
            config.num_shared_blocks,
        ),
        train_loader=build_train_loader(args),
        config=config,
        output_dir=args.output_dir,
        max_optimizer_steps=args.steps,
        accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_steps=(1000, 2000, 3000),
        device=args.device,
        use_bf16=True,
    )


if __name__ == "__main__":
    main()
