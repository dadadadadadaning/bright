#!/usr/bin/env python3
"""Export immutable building/region evidence cache from V2 Stage-1/Stage-2.

GT is used only after forward to attach a weak building label to a predicted
building instance.  It never enters predicted-building partition generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    load_a1_parent,
    load_v2_checkpoint,
    parse_batch,
    seed_worker,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import (
    DamageFormerBuildingInternalChangePartitionV2,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import read_list

from .cache_features import building_feature_from_regions
from .core import event_name, reject_test_path, sha256_file


def main() -> None:
    p = argparse.ArgumentParser("Export V2 building/region evidence cache")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--list_file", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--dataset_mode", choices=("train", "val"), default="val", help="Use val-mode transforms by default for deterministic representation caches, even when list_file is train_set.txt.")
    p.add_argument("--output", required=True)
    p.add_argument("--variant", default="")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label_purity", type=float, default=0.70)
    p.add_argument("--min_region_mass", type=float, default=0.50)
    p.add_argument("--max_batches", type=int, default=0)
    args = p.parse_args()
    reject_test_path(args.list_file, args.output)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = args.variant or str(payload.get("variant", "p_sc_residual"))
    stage = str(payload.get("stage", "change"))
    if stage not in ("change", "feedback"):
        raise RuntimeError("representation cache requires a Stage-1(change) or Stage-2(feedback) checkpoint")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingInternalChangePartitionV2(variant).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    ckpt_report = load_v2_checkpoint(model, args.checkpoint)
    model.configure_stage(stage)
    model.eval()

    ds = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.list_file), args.crop_size, type=args.dataset_mode
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    features: List[torch.Tensor] = []
    labels: List[int] = []
    purities: List[float] = []
    names: List[str] = []
    events: List[str] = []
    pred_building_ids: List[int] = []
    base_probs: List[torch.Tensor] = []
    uncertainty: List[float] = []
    peak_mean_delta: List[float] = []
    region_count: List[int] = []
    all_region_tokens: List[torch.Tensor] = []
    all_region_critical: List[torch.Tensor] = []
    all_region_logits: List[torch.Tensor] = []
    region_offsets = [0]
    skipped_purity = 0
    skipped_empty = 0
    images = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            pre, post, _, damage, batch_names = parse_batch(batch, device)
            _, output_damage, aux = model(pre, post, return_aux=True)
            b_ids = aux["building_ids"]
            h, w = b_ids.shape[-2:]
            if damage.ndim == 4 and damage.shape[1] == 1:
                damage2 = damage[:, 0]
            else:
                damage2 = damage
            gt_small = F.interpolate(damage2.unsqueeze(1).float(), size=(h, w), mode="nearest")[:, 0].long()
            probs = torch.softmax(output_damage.float(), dim=1)
            probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
            probs3 = probs[:, 1:4]

            owners = aux["token_building_ids"]
            mass = aux["region_mass"]
            region_tok = aux["region_tokens"].float()
            rel_mean = aux["relation_token_mean"].float()
            rel_peak = aux["relation_token_peak"].float()
            rel_std = aux["relation_token_std"].float()
            region_logits = aux["region_logits"].float()

            for bi in range(pre.shape[0]):
                sample_name = str(batch_names[bi])
                for bid in torch.unique(b_ids[bi]).tolist():
                    bid = int(bid)
                    if bid <= 0:
                        continue
                    pmask = b_ids[bi] == bid
                    y = gt_small[bi][pmask]
                    y = y[(y >= 1) & (y <= 3)]
                    if y.numel() == 0:
                        skipped_purity += 1
                        continue
                    counts = torch.stack([(y == c).sum() for c in (1, 2, 3)]).float()
                    dom = int(counts.argmax())
                    purity = float(counts[dom] / counts.sum().clamp_min(1.0))
                    if purity < args.label_purity:
                        skipped_purity += 1
                        continue

                    tmask = (owners[bi] == bid) & (mass[bi] >= args.min_region_mass)
                    tidx = torch.nonzero(tmask, as_tuple=False).flatten()
                    if tidx.numel() == 0:
                        skipped_empty += 1
                        continue
                    tok = region_tok[bi, tidx]
                    critical = (
                        (rel_peak[bi, tidx] - rel_mean[bi, tidx]).abs().mean(-1)
                        + rel_std[bi, tidx].abs().mean(-1)
                    )
                    feat = building_feature_from_regions(tok, critical)
                    bp = probs3[bi, :, pmask].mean(1)
                    bp = bp / bp.sum().clamp_min(1e-8)

                    features.append(feat.cpu().half())
                    labels.append(dom)
                    purities.append(purity)
                    names.append(sample_name)
                    events.append(event_name(sample_name))
                    pred_building_ids.append(bid)
                    base_probs.append(bp.cpu().half())
                    uncertainty.append(float(1.0 - bp.max()))
                    peak_mean_delta.append(float((rel_peak[bi, tidx] - rel_mean[bi, tidx]).abs().mean()))
                    region_count.append(int(tidx.numel()))
                    all_region_tokens.append(tok.cpu().half())
                    all_region_critical.append(critical.cpu().half())
                    all_region_logits.append(region_logits[bi, tidx].cpu().half())
                    region_offsets.append(region_offsets[-1] + int(tidx.numel()))
            images += int(pre.shape[0])

    if not features:
        raise RuntimeError("no building records exported")
    cache: Dict[str, object] = {
        "protocol": "DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1_CACHE",
        "split": args.split,
        "variant": variant,
        "stage": stage,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "a1_checkpoint": str(args.a1_ckpt),
        "a1_checkpoint_sha256": sha256_file(args.a1_ckpt),
        "parent_report": parent_report,
        "checkpoint_report": ckpt_report,
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_purity": torch.tensor(purities, dtype=torch.float32),
        "names": names,
        "events": events,
        "pred_building_ids": torch.tensor(pred_building_ids, dtype=torch.long),
        "base_probs": torch.stack(base_probs),
        "base_pred": torch.stack(base_probs).argmax(1),
        "uncertainty": torch.tensor(uncertainty, dtype=torch.float32),
        "peak_mean_delta": torch.tensor(peak_mean_delta, dtype=torch.float32),
        "region_count": torch.tensor(region_count, dtype=torch.long),
        "region_offsets": torch.tensor(region_offsets, dtype=torch.long),
        "region_tokens": torch.cat(all_region_tokens, dim=0),
        "region_critical": torch.cat(all_region_critical, dim=0),
        "region_logits": torch.cat(all_region_logits, dim=0),
        "feature_definition": "concat(region_mean, region_std, top1_critical_region, top2_critical_region)",
        "critical_definition": "mean_abs(relation_peak-relation_mean)+mean_abs(relation_std)",
        "class_names": ["Intact", "Damaged", "Destroyed"],
        "processed_images": images,
        "dataset_mode": args.dataset_mode,
        "building_records": len(features),
        "skipped_label_purity": skipped_purity,
        "skipped_empty_regions": skipped_empty,
        "gt_used_in_forward_partition": False,
        "test_accessed": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(json.dumps({
        "output": str(out),
        "split": args.split,
        "variant": variant,
        "stage": stage,
        "images": images,
        "buildings": len(features),
        "regions": int(region_offsets[-1]),
        "feature_dim": int(cache["features"].shape[1]),
        "region_dim": int(cache["region_tokens"].shape[1]),
        "class_counts": [int((cache["labels"] == c).sum()) for c in range(3)],
        "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
