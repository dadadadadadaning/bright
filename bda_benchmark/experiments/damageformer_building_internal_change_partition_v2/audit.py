#!/usr/bin/env python3
"""Validation-only structural/evidence audit for V2 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset

from .building_ops import assert_same_building_candidates
from .core import load_a1_parent, load_v2_checkpoint, parse_batch, seed_worker, write_json
from .losses import weak_building_bag_loss
from .model import DamageFormerBuildingInternalChangePartitionV2
from .train import read_list


def q(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: 0.0 for k in ("mean", "p10", "p25", "p50", "p75", "p90")}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser("V2 region audit")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_list", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_batches", type=int, default=0)
    args = p.parse_args()

    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", "p_sc_residual"))
    stage = str(payload.get("stage", "change"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingInternalChangePartitionV2(variant).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    ckpt_report = load_v2_checkpoint(model, args.checkpoint)
    model.configure_stage(stage)
    model.eval()

    val_set = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), args.crop_size, type="val"
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    building_areas: List[float] = []
    nominal_regions: List[float] = []
    effective_1: List[float] = []
    effective_5: List[float] = []
    effective_10: List[float] = []
    region_mass_ratio: List[float] = []
    assoc_entropy: List[float] = []
    change_alpha: List[float] = []
    change_gate: List[float] = []
    confidence: List[float] = []
    align_entropy: List[float] = []
    peak_delta: List[float] = []
    bag_correct = 0.0
    bag_count = 0.0
    images = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        pre, post, _, damage, _ = parse_batch(batch, device)
        _, _, aux = model(pre, post, return_aux=True)
        assert_same_building_candidates(
            aux["building_ids"],
            aux["candidate_ids"],
            aux["candidate_valid"],
            aux["token_building_ids"],
        )
        owners = aux["token_building_ids"]
        mass = aux["region_mass"].float()
        area = aux["token_building_area"].float()
        ratio = aux["region_area_ratio"].float()
        for bi in range(owners.shape[0]):
            ids = torch.unique(owners[bi])
            ids = ids[ids > 0]
            for building_id in ids.tolist():
                idx = owners[bi] == int(building_id)
                if not bool(idx.any().item()):
                    continue
                building_areas.append(float(area[bi, idx][0]))
                nominal_regions.append(float(idx.sum()))
                r = ratio[bi, idx]
                effective_1.append(float((r >= 0.01).sum()))
                effective_5.append(float((r >= 0.05).sum()))
                effective_10.append(float((r >= 0.10).sum()))
                region_mass_ratio.extend(r.detach().cpu().tolist())
        building_mask = (aux["building_ids"] > 0).reshape(aux["building_ids"].shape[0], -1)
        if bool(building_mask.any().item()):
            assoc_entropy.extend(aux["association_entropy"][building_mask].float().cpu().tolist())
        change_alpha.append(float(aux["change_alpha"].abs().mean()))
        change_gate.append(float(aux["change_gate_probability"].mean()))
        confidence.append(float(aux["region_confidence"].mean()))
        align_entropy.append(float(aux["region_alignment_entropy"].mean()))
        peak_delta.append(
            float((aux["relation_token_peak"] - aux["relation_token_mean"]).abs().mean())
        )
        _, stats = weak_building_bag_loss(aux, damage)
        bag_correct += stats["bag_accuracy"] * stats["bag_count"]
        bag_count += stats["bag_count"]
        images += int(pre.shape[0])

    summary = {
        "protocol": "DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2_AUDIT",
        "variant": variant,
        "stage": stage,
        "parent_report": parent_report,
        "checkpoint_report": ckpt_report,
        "processed_images": images,
        "building_area_feature_pixels": q(building_areas),
        "nominal_regions_per_building": q(nominal_regions),
        "effective_regions_per_building_at_1pct": q(effective_1),
        "effective_regions_per_building_at_5pct": q(effective_5),
        "effective_regions_per_building_at_10pct": q(effective_10),
        "region_area_ratio": q(region_mass_ratio),
        "association_entropy": q(assoc_entropy),
        "change_alpha_abs_mean": float(np.mean(change_alpha)) if change_alpha else 0.0,
        "change_gate_mean": float(np.mean(change_gate)) if change_gate else 0.0,
        "alignment_confidence_mean": float(np.mean(confidence)) if confidence else 0.0,
        "alignment_entropy_mean": float(np.mean(align_entropy)) if align_entropy else 0.0,
        "soft_peak_minus_mean_abs": float(np.mean(peak_delta)) if peak_delta else 0.0,
        "weak_building_bag_count": bag_count,
        "weak_building_bag_accuracy": bag_correct / max(1.0, bag_count),
        "hard_candidate_cross_building_violations": 0,
        "local_damage_localization_claim": False,
        "note": (
            "BRIGHT building labels supervise weak building bags only; this audit does not "
            "interpret semantic class consistency as sub-building damage localization."
        ),
        "test_accessed": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
