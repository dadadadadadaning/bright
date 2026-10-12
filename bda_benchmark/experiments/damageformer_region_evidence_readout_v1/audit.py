#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from .train import read_list
from .core import load_frozen_sources, load_readout, sha256_file, verify_checkpoint_sources, write_json
from .evaluation import evaluate, format_result
from .model import DamageFormerRegionEvidenceReadout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True); p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_list", required=True); p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--stage1_ckpt", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_json", required=True); p.add_argument("--calibration_json", default="")
    p.add_argument("--crop_size", type=int, default=640); p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8); p.add_argument("--device", default="cuda:0")
    p.add_argument("--ignore_index", type=int, default=255); p.add_argument("--minimum_stage1_step", type=int, default=3000)
    args = p.parse_args()
    if "test" in Path(args.val_list).name.lower(): raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerRegionEvidenceReadout(args.variant).to(device)
    sources = load_frozen_sources(model, args.a1_ckpt, args.stage1_ckpt, args.minimum_stage1_step)
    ckpt = load_readout(model, args.checkpoint); verify_checkpoint_sources(ckpt, sources)
    model.configure_readout_training(); model.eval()
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.val_list), args.crop_size, type="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    runs = {}
    model.selective_threshold = None
    runs["global"] = evaluate(model, loader, device, args.ignore_index)
    print("global | " + format_result(runs["global"]), flush=True)
    calibration = None
    if args.calibration_json:
        calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
        if calibration["variant"] != args.variant: raise RuntimeError("calibration variant mismatch")
        if calibration["checkpoint_sha256"] != sha256_file(args.checkpoint):
            raise RuntimeError("calibration was made for a different readout checkpoint")
        for coverage, threshold in calibration["thresholds"].items():
            if coverage == "1.0": continue
            model.selective_threshold = float(threshold)
            runs[f"top_{100*float(coverage):g}pct"] = evaluate(model, loader, device, args.ignore_index)
            print(f"top {100*float(coverage):g}% | " + format_result(runs[f"top_{100*float(coverage):g}pct"]), flush=True)
    write_json(Path(args.output_json), {"protocol": "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1_AUDIT",
        "variant": args.variant, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_step": int(ckpt["step"]),
        "sources": sources, "calibration": calibration, "runs": runs, "test_accessed": False})
    print(f"Saved: {args.output_json}")


if __name__ == "__main__": main()
