#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import parse_batch, read_list, sha256_file
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import (
    PROTOCOL, extract_region_records, local_tolerant_alignment, relation_map,
    rgb_sobel_edge,
)
from bda_benchmark.experiments.spi_superpixel_relation_occurrence_v1.core import make_region_ids
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


def event_name(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


def main() -> None:
    parser = argparse.ArgumentParser("Build one descriptor record per reliable train/validation superpixel")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region_mode", choices=("slic", "grid", "shuffled"), default="slic")
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--alignment_radius", type=int, default=1)
    parser.add_argument("--purity", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.data_list).name.lower():
        raise RuntimeError("V2 record builder forbids test-list access")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Keep the actual P3 inference contract.  The Memory descriptor below
    # explicitly recovers the unweighted fused view used by the original V2
    # cache instead of changing the whole model through roi_alpha=1.0.
    model = DamageFormer_S8_MICB_DSMIL().to(device).eval()
    parent = load_parent(model, args.parent_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = read_list(args.data_list)
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=640, type="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    descriptors, soft_labels, labels, purities, overlaps, areas, events, images = [], [], [], [], [], [], [], []
    region_total = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            pre, post, location, damage, sample_names = parse_batch(batch, device)
            _, _, aux = model(pre, post, return_aux=True)
            head = model.micb_dsmil
            rgb = head.rgb_proj(aux["rgb_feature"])
            sar = head.sar_proj(aux["sar_feature"])
            fused_unweighted = aux["fused_roi"] / aux["roi_weight"].clamp_min(1e-6)
            fused = head.fused_proj(fused_unweighted)
            aligned, entropy, offset = local_tolerant_alignment(rgb, sar, args.alignment_radius)
            relation = relation_map(rgb, aligned, fused, head.cross_token)
            edge = rgb_sobel_edge(pre, relation.shape[-2:])
            ids = make_region_ids(
                pre, relation.shape[-2:], mode=args.region_mode,
                n_segments=args.segments, compactness=args.compactness,
            )
            region_total += sum(int(torch.unique(row).numel()) for row in ids)
            rows = extract_region_records(
                relation, fused, post[:, :1], ids, aux["p_building"], entropy, offset, edge,
                damage, location, sample_names,
                [event_name(str(value)) for value in sample_names], min_purity=args.purity,
            )
            for row in rows:
                descriptors.append(row.descriptor.cpu())
                soft_labels.append(row.soft_label.cpu())
                labels.append(row.hard_label); purities.append(row.purity)
                overlaps.append(row.building_overlap); areas.append(row.area)
                events.append(row.event_id); images.append(row.image_id)
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"records {batch_index}/{len(loader)} kept={len(labels)}", flush=True)
    if not descriptors:
        raise RuntimeError("no reliable superpixel records were found")
    payload = {
        "protocol": PROTOCOL + "_RECORDS",
        "descriptors": torch.stack(descriptors).float(),
        "soft_labels": torch.stack(soft_labels).float(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "purities": torch.tensor(purities), "building_overlaps": torch.tensor(overlaps),
        "areas": torch.tensor(areas), "events": events, "images": images,
        "metadata": {
            "source_list": str(Path(args.data_list).resolve()),
            "source_list_sha256": sha256_file(args.data_list),
            "image_count": len(names), "raw_region_count": region_total,
            "kept_record_count": len(labels), "descriptor_dimension": int(descriptors[0].numel()),
            "region_mode": args.region_mode, "segments_at_640": args.segments,
            "compactness": args.compactness, "alignment_radius": args.alignment_radius,
            "purity_threshold": args.purity, "parent": parent,
            "p3_pixel_roi_alpha": 0.1,
            "descriptor_fused_view": "unweighted fused feature (V2-compatible)",
            "same_image_class_averaging": False, "test_accessed": False,
        },
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
