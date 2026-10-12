#!/usr/bin/env python3
"""Build frozen train/validation region records and partition-oracle statistics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    event_name, load_a1_parent, load_v2_checkpoint, parse_batch, seed_worker, sha256_file,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import (
    DamageFormerBuildingInternalChangePartitionV2,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import read_list

from .features import (
    a1_descriptor, hard_region_ids, matched_grid_mask, partition_edges, region_descriptor,
    resize_labels, semantic_edges,
)
from .metrics import add_hist, confusion_metrics


def _empty_stats() -> Dict[str, object]:
    return {"hist": torch.zeros(3, 3, dtype=torch.float64), "pixels": 0, "correct": 0, "purity_num": 0.0, "regions": 0}


def _update_group(group, counts: torch.Tensor, target: int) -> None:
    pixels = int(counts.sum())
    group["hist"][:, target] += counts.double()
    group["pixels"] += pixels
    group["correct"] += int(counts[target])
    group["purity_num"] += int(counts[target])
    group["regions"] += 1


def _final_group(group) -> Dict[str, object]:
    return {
        **confusion_metrics(group["hist"]),
        "pixel_weighted_region_purity": group["purity_num"] / max(1, group["pixels"]),
        "undersegmentation_error": 1.0 - group["purity_num"] / max(1, group["pixels"]),
        "region_count": int(group["regions"]),
        "valid_pixel_count": int(group["pixels"]),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--source_list", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--region_ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--boundary_tolerance", type=int, default=1)
    p.add_argument("--min_region_pixels", type=int, default=2)
    args = p.parse_args()
    if "test" in Path(args.source_list).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")

    payload = torch.load(args.region_ckpt, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", "p_sc_residual"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingInternalChangePartitionV2(variant).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    region_report = load_v2_checkpoint(model, args.region_ckpt)
    model.configure_stage("change")
    model.eval()
    rows: Dict[str, List] = defaultdict(list)
    overall = _empty_stats()
    by_event: Dict[str, Dict] = defaultdict(_empty_stats)
    by_size: Dict[str, Dict] = defaultdict(_empty_stats)
    by_ratio: Dict[str, Dict] = defaultdict(_empty_stats)
    boundary_hit = boundary_total = 0
    internal_boundary_hit = internal_boundary_total = 0
    small_preserved = small_total = 0
    processed = 0

    names = read_list(args.source_list)
    if args.max_images:
        names = names[: args.max_images]
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, 640, type="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        persistent_workers=args.num_workers > 0)
    for batch_idx, batch in enumerate(loader):
        pre, post, _, damage, sample_names = parse_batch(batch, device)
        _, _, aux = model(pre, post, return_aux=True)
        h, w = aux["building_ids"].shape[-2:]
        gt = resize_labels(damage, (h, w))
        hard_ids = hard_region_ids(aux)
        for bi, sample_name in enumerate(sample_names):
            event = event_name(str(sample_name))
            valid = (gt[bi] >= 1) & (gt[bi] <= 3)
            valid_all = (gt[bi] >= 0) & (gt[bi] <= 3)
            gt_edge = semantic_edges(gt[bi], valid_all)
            internal_gt_edge = semantic_edges(gt[bi], valid)
            reg_edge = partition_edges(hard_ids[bi], aux["building_ids"][bi] > 0)
            if args.boundary_tolerance > 0:
                k = 2 * args.boundary_tolerance + 1
                reg_near = F.max_pool2d(reg_edge[None, None].float(), k, stride=1,
                                        padding=args.boundary_tolerance)[0, 0] > 0
            else:
                reg_near = reg_edge
            boundary_hit += int((gt_edge & reg_near).sum())
            boundary_total += int(gt_edge.sum())
            internal_boundary_hit += int((internal_gt_edge & reg_near).sum())
            internal_boundary_total += int(internal_gt_edge.sum())

            # Small affected components mean small labelled affected buildings/components,
            # not sub-building physical crack masks.
            affected = ((gt[bi] == 2) | (gt[bi] == 3)).cpu().numpy()
            try:
                from scipy import ndimage
                cc, number = ndimage.label(affected)
                for cid in range(1, number + 1):
                    component = torch.from_numpy(cc == cid).to(device)
                    area = int(component.sum())
                    if area > 16:
                        continue
                    small_total += 1
                    kept = False
                    for rid in torch.unique(hard_ids[bi][component]).tolist():
                        if int(rid) <= 0:
                            continue
                        region = hard_ids[bi] == int(rid)
                        overlap = int((region & component).sum())
                        if overlap / max(1, area) >= 0.5 and overlap / max(1, int(region.sum())) >= 0.5:
                            kept = True
                            break
                    small_preserved += int(kept)
            except Exception:
                pass

            for rid in torch.unique(hard_ids[bi]).tolist():
                if int(rid) <= 0:
                    continue
                mask = hard_ids[bi] == int(rid)
                valid_mask = mask & valid
                if int(valid_mask.sum()) < args.min_region_pixels:
                    continue
                counts = torch.stack([(gt[bi][valid_mask] == c).sum() for c in (1, 2, 3)]).cpu()
                target = int(counts.argmax())
                building_values = aux["building_ids"][bi][mask]
                building_id = int(torch.mode(building_values).values)
                building_mask = aux["building_ids"][bi] == building_id
                building_area = int(building_mask.sum())
                affected_ratio = float((((gt[bi] == 2) | (gt[bi] == 3)) & building_mask).sum()) / max(1, building_area)
                size_bin = "small" if building_area <= 16 else ("medium" if building_area <= 64 else "large")
                ratio_bin = "low" if affected_ratio <= 0.10 else ("medium" if affected_ratio <= 0.50 else "high")
                _update_group(overall, counts, target)
                _update_group(by_event[event], counts, target)
                _update_group(by_size[size_bin], counts, target)
                _update_group(by_ratio[ratio_bin], counts, target)

                grid_mask = matched_grid_mask(mask, building_mask)
                region_feat = region_descriptor(model, aux, bi, mask, building_area)
                grid_feat = region_descriptor(model, aux, bi, grid_mask, building_area)
                a1_feat = a1_descriptor(aux, bi, mask)
                p = aux["parent_damage_probability"][bi, 1:4, mask].float().mean(1)
                rows["a1"].append(a1_feat.cpu())
                rows["region"].append(region_feat.cpu())
                rows["grid"].append(grid_feat.cpu())
                rows["target"].append(target)
                rows["a1_prediction"].append(int(p.argmax()))
                rows["event"].append(event)
                rows["image"].append(str(sample_name))
                rows["size_bin"].append(size_bin)
                rows["damage_ratio_bin"].append(ratio_bin)
                rows["pixel_count"].append(int(valid_mask.sum()))
        processed += len(sample_names)
        if (batch_idx + 1) % 25 == 0 or processed == len(dataset):
            print(f"records {processed}/{len(dataset)} regions={len(rows['target'])}", flush=True)

    record = {
        "protocol": "DAMAGEFORMER_REGION_INFORMATION_AUDIT_V1_RECORDS",
        "split": args.split,
        "a1": torch.stack(rows["a1"]) if rows["a1"] else torch.empty(0, 18),
        "region": torch.stack(rows["region"]) if rows["region"] else torch.empty(0, 128),
        "grid": torch.stack(rows["grid"]) if rows["grid"] else torch.empty(0, 128),
        "target": torch.tensor(rows["target"], dtype=torch.long),
        "a1_prediction": torch.tensor(rows["a1_prediction"], dtype=torch.long),
        "event": rows["event"], "image": rows["image"], "size_bin": rows["size_bin"],
        "damage_ratio_bin": rows["damage_ratio_bin"],
        "pixel_count": torch.tensor(rows["pixel_count"], dtype=torch.long),
        "source_list": str(Path(args.source_list).resolve()),
        "source_sha256": sha256_file(args.source_list),
        "test_accessed": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, args.output)
    audit = {
        "protocol": "DAMAGEFORMER_REGION_INFORMATION_AUDIT_V1_PARTITION",
        "split": args.split, "processed_images": processed,
        "region_count": len(rows["target"]),
        "boundary_recall": boundary_hit / max(1, boundary_total),
        "boundary_hit": boundary_hit, "boundary_total": boundary_total,
        "internal_class_boundary_recall": internal_boundary_hit / max(1, internal_boundary_total),
        "internal_class_boundary_hit": internal_boundary_hit,
        "internal_class_boundary_total": internal_boundary_total,
        "small_affected_component_preservation": small_preserved / max(1, small_total),
        "small_affected_component_count": small_total,
        "oracle_region_upper_bound": _final_group(overall),
        "by_event": {k: _final_group(v) for k, v in sorted(by_event.items())},
        "by_building_size": {k: _final_group(v) for k, v in sorted(by_size.items())},
        "by_damage_ratio": {k: _final_group(v) for k, v in sorted(by_ratio.items())},
        "parent": parent_report, "region_checkpoint": region_report,
        "interpretation_limit": (
            "BRIGHT damage labels are building-level raster labels. Boundary recall and small-component "
            "preservation therefore audit labelled building/class geometry, not physical sub-building crack localization."
        ),
        "test_accessed": False,
    }
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
