#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from rgid import RGIDConfig, RGIDHead, build_rgid_targets
from rgid.calibration import BoundaryThresholdCalibrator
from rgid.evaluation import evaluate_g123, load_head_checkpoint, save_evaluation
from rgid.training import train_rgid_head


class SavedBatchDataset(Dataset):
    def __init__(self, payload) -> None:
        required = ("fused_roi", "instance_ids", "parent_mask")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"saved batch missing {missing}")
        self.fused_roi = torch.as_tensor(payload["fused_roi"]).float()
        self.instance_ids = torch.as_tensor(payload["instance_ids"]).long()
        self.parent_mask = torch.as_tensor(payload["parent_mask"]).bool()
        if self.fused_roi.ndim != 4 or self.fused_roi.shape[1] != 256:
            raise ValueError("fused_roi must be [B,256,Hh,Wh]")
        if self.instance_ids.ndim != 3 or self.parent_mask.ndim != 3:
            raise ValueError("instance_ids and parent_mask must be [B,H,W]")
        if self.instance_ids.shape != self.parent_mask.shape:
            raise ValueError("instance and parent tensors have different shapes")
        if self.fused_roi.shape[0] != self.instance_ids.shape[0]:
            raise ValueError("saved batch sizes differ")
        if not torch.isfinite(self.fused_roi).all():
            raise ValueError("fused_roi contains NaN or Inf")
        if (self.instance_ids < 0).any() or not self.parent_mask.any():
            raise ValueError("invalid instance labels or empty parent mask")

    def __len__(self):
        return int(self.fused_roi.shape[0])

    def __getitem__(self, index):
        return {
            "fused_roi": self.fused_roi[index],
            "instance_ids": self.instance_ids[index],
            "parent_mask": self.parent_mask[index],
        }


def parse_args():
    parser = argparse.ArgumentParser("RGID real-batch integration test")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.batch, map_location="cpu")
    dataset = SavedBatchDataset(payload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = RGIDConfig()
    head = RGIDHead()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    first = next(iter(loader))
    target_check, target_stats = build_rgid_targets(
        first["instance_ids"],
        first["fused_roi"].shape[-2:],
        config,
    )
    if not torch.isfinite(target_check["center_target"]).all():
        raise RuntimeError("target construction produced non-finite values")

    train_rgid_head(
        head=head,
        train_loader=loader,
        config=config,
        output_dir=output,
        max_optimizer_steps=args.steps,
        accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_steps=(args.steps,),
        device=device,
        use_bf16=str(device).startswith("cuda"),
        log_every=1,
    )
    checkpoint_path = output / f"rgid_step_{args.steps}.pth"
    loaded_head, loaded_config, _ = load_head_checkpoint(
        checkpoint_path, device=device
    )

    calibrator = BoundaryThresholdCalibrator()
    loaded_head.eval()
    with torch.no_grad():
        for batch in loader:
            fused = batch["fused_roi"].to(device)
            predictions = loaded_head(fused)
            targets, _ = build_rgid_targets(
                batch["instance_ids"],
                predictions["boundary_logits"].shape[-2:],
                loaded_config,
                device=device,
            )
            calibrator.update(
                predictions["boundary_logits"],
                targets["boundary_target"],
                building_mask=batch["parent_mask"].to(device),
            )
    calibration = calibrator.compute()
    results = evaluate_g123(
        head=loaded_head,
        dataloader=loader,
        config=loaded_config,
        device=device,
        boundary_gate_threshold=calibration.boundary_gate_threshold,
    )
    results["boundary_calibration"] = calibration.__dict__
    results["target_stats"] = [stats.to_dict() for stats in target_stats]
    result_path = output / "real_batch_smoke_results.json"
    save_evaluation(results, result_path)
    print(json.dumps(results, indent=2), flush=True)
    print("Real-batch integration test passed.", flush=True)


if __name__ == "__main__":
    main()
