#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from .core import load_frozen_sources, load_readout, parse_batch, sha256_file, verify_checkpoint_sources, write_json
from .model import DamageFormerRegionEvidenceReadout
from .train import read_list


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Train-only selective coverage calibration")
    p.add_argument("--variant", required=True, choices=("w2_id_logit", "w3_id_mass"))
    p.add_argument("--dataset_path", required=True); p.add_argument("--train_list", required=True)
    p.add_argument("--a1_ckpt", required=True); p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output_json", required=True)
    p.add_argument("--crop_size", type=int, default=640); p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8); p.add_argument("--device", default="cuda:0")
    p.add_argument("--minimum_stage1_step", type=int, default=3000); p.add_argument("--max_images", type=int, default=0)
    args = p.parse_args()
    if "test" in Path(args.train_list).name.lower(): raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerRegionEvidenceReadout(args.variant).to(device)
    sources = load_frozen_sources(model, args.a1_ckpt, args.stage1_ckpt, args.minimum_stage1_step)
    ckpt = load_readout(model, args.checkpoint); verify_checkpoint_sources(ckpt, sources)
    model.configure_readout_training(); model.eval()
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, read_list(args.train_list), args.crop_size, type="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    scores, images = [], 0
    for batch in loader:
        pre, post, _, _, _ = parse_batch(batch, device)
        _, _, aux = model(pre, post, return_aux=True)
        for bi in range(pre.shape[0]):
            ids = aux["building_ids"][bi]
            candidate = aux["candidate_gate"][bi, 0] > 0
            score = aux["id_gate_score"][bi, 0]
            for building_id in torch.unique(ids).tolist():
                if int(building_id) <= 0: continue
                mask = (ids == int(building_id)) & candidate
                if bool(mask.any()): scores.append(float(score[mask].float().mean()))
        images += int(pre.shape[0])
        if args.max_images and images >= args.max_images: break
    if not scores: raise RuntimeError("no candidate buildings found in training data")
    values = torch.tensor(scores, dtype=torch.float64)
    coverages = (0.01, 0.02, 0.05, 0.10)
    thresholds = {str(c): float(torch.quantile(values, 1.0-c)) for c in coverages}
    write_json(Path(args.output_json), {"protocol": "DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1_TRAIN_CALIBRATION",
        "variant": args.variant, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_step": int(ckpt["step"]),
        "source_list": str(Path(args.train_list).resolve()), "candidate_building_count": len(scores),
        "processed_images": images, "thresholds": thresholds, "sources": sources,
        "gt_used_for_threshold": False, "validation_used_for_threshold": False, "test_accessed": False})
    print(f"Saved {args.output_json}; buildings={len(scores)} thresholds={thresholds}")


if __name__ == "__main__": main()
