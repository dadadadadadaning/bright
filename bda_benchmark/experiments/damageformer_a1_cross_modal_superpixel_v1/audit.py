#!/usr/bin/env python3
"""Validation and synthetic-offset robustness audit. Test access is locked."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset

from .core import load_a1_parent, load_experiment_checkpoint, seed_worker, sha256_file, write_json
from .model import DamageFormerA1CrossModalSuperpixel
from .train import evaluate, format_metrics, read_list
from .variants import VARIANTS


def shift_without_wrap(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    output = torch.zeros_like(tensor)
    h, w = tensor.shape[-2:]
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = tensor[..., src_y0:src_y1, src_x0:src_x1]
    return output


class PerturbedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, dx: int, dy: int) -> None:
        super().__init__()
        self.model, self.dx, self.dy = model, dx, dy

    def forward(self, pre, post, *args, **kwargs):
        return self.model(pre, shift_without_wrap(post, self.dx, self.dy), *args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_kind", choices=("parent", "experiment"), default="experiment")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--perturbation", choices=("none", "x4", "y4", "x8", "y8", "x16", "y16"), default="none")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--region_stride", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()
    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("test access is forbidden in this research protocol")
    model = DamageFormerA1CrossModalSuperpixel(args.variant, args.region_stride)
    load_report = (load_a1_parent(model, args.checkpoint) if args.checkpoint_kind == "parent"
                   else load_experiment_checkpoint(model, args.checkpoint))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    shifts = {"none": (0, 0), "x4": (4, 0), "y4": (0, 4), "x8": (8, 0),
              "y8": (0, 8), "x16": (16, 0), "y16": (0, 16)}
    dx, dy = shifts[args.perturbation]
    wrapped = PerturbedModel(model, dx, dy).to(device)
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.val_list), 640, type="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        persistent_workers=args.num_workers > 0)
    metrics = evaluate(wrapped, loader, device, 255, args.max_batches)
    payload = {"protocol": "DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1_OFFSET_AUDIT",
               "variant": args.variant, "checkpoint": str(Path(args.checkpoint).resolve()),
               "checkpoint_sha256": sha256_file(args.checkpoint), "load_report": load_report,
               "perturbation": args.perturbation, "offset": {"dx": dx, "dy": dy},
               "metrics": metrics, "test_accessed": False}
    write_json(Path(args.output_json), payload)
    print(format_metrics(metrics))
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()

