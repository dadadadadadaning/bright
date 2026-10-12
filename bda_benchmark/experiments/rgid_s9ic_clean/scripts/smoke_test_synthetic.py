#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from rgid import RGIDConfig, RGIDHead
from rgid.evaluation import evaluate_g123, load_head_checkpoint
from rgid.training import train_rgid_head


class SyntheticRGIDDataset(Dataset):
    def __init__(self, count: int = 12) -> None:
        generator = torch.Generator().manual_seed(3407)
        self.features = torch.randn(count, 256, 16, 16, generator=generator)
        self.instances = torch.zeros(count, 64, 64, dtype=torch.long)
        self.parents = torch.zeros(count, 64, 64, dtype=torch.bool)
        for index in range(count):
            self.parents[index, 8:56, 8:56] = True
            self.instances[index, 8:56, 8:32] = 1
            self.instances[index, 8:56, 32:56] = 2
            self.features[index, 0, 5:11, 3:8] += 2.0
            self.features[index, 1, 5:11, 8:13] += 2.0

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, index):
        return {
            "fused_roi": self.features[index],
            "instance_ids": self.instances[index],
            "parent_mask": self.parents[index],
        }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RGIDConfig()
    dataset = SyntheticRGIDDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    head = RGIDHead(
        config.in_channels,
        config.hidden_channels,
        config.num_shared_blocks,
    )
    with tempfile.TemporaryDirectory(prefix="rgid_synthetic_") as directory:
        result = train_rgid_head(
            head=head,
            train_loader=loader,
            config=config,
            output_dir=directory,
            max_optimizer_steps=2,
            accumulation_steps=1,
            checkpoint_steps=(1, 2),
            device=device,
            use_bf16=device == "cuda",
            log_every=1,
        )
        if not torch.isfinite(torch.tensor(result["last_loss"]["loss"])):
            raise RuntimeError("synthetic training returned a non-finite loss")
        checkpoint_path = Path(directory) / "rgid_step_2.pth"
        if not checkpoint_path.is_file():
            raise RuntimeError("synthetic checkpoint was not created")
        loaded_head, loaded_config, checkpoint = load_head_checkpoint(
            checkpoint_path, device=device
        )
        if int(checkpoint["optimizer_step"]) != 2:
            raise RuntimeError("strictly loaded checkpoint has the wrong step")
        evaluation = evaluate_g123(
            head=loaded_head,
            dataloader=loader,
            config=loaded_config,
            device=device,
        )
        for mode in ("g1", "g2", "g3"):
            if mode not in evaluation:
                raise RuntimeError(f"{mode} decode result is missing")
        print(json.dumps(evaluation, indent=2), flush=True)
    print("Synthetic smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
