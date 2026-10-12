#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_aligned_region_memory_v1.build_records import GROUP_NAMES, load_aligned_model
from bda_benchmark.experiments.spi_aligned_region_memory_v1.core import region_ids
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import event_name, parse_batch
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    _pool_feature, extract_factorized_records, predicted_component_ids, recover_sar_display,
    rgb_orientation_maps,
)
from bda_benchmark.experiments.spi_building_relational_memory_v1.core import PROTOCOL


def split_by_component(ids: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Make every (SLICO id, predicted-building component) pair unique."""
    if ids.shape != components.shape:
        raise ValueError("region and component grids must match")
    output = torch.empty_like(ids)
    for batch in range(ids.shape[0]):
        factor = int(components[batch].max().item()) + 1
        output[batch] = ids[batch] * max(factor, 1) + components[batch]
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Build complete building-internal SLICO Token records")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--roi_variant", choices=("a2_lca3", "a3_conf3", "a4_orient3", "a5_conf5"), required=True)
    parser.add_argument("--roi_checkpoint", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--min_query_building", type=float, default=0.60)
    parser.add_argument("--min_tokens", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.data_list).name.lower():
        raise RuntimeError("Test access is forbidden in V1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = json.loads(Path(args.sar_stats_json).read_text(encoding="utf-8"))
    model, model_report = load_aligned_model(args.roi_variant, args.roi_checkpoint, device)
    dataset = HeterogeneousRGBSARDataset(args.dataset_path, args.data_list, 640, split="val", variant="h0")
    if args.max_images:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    output_group_names = tuple(GROUP_NAMES) + ("sar_aligned",)
    grouped = {name: [] for name in output_group_names}; labels = []; soft_labels = []
    purities = []; overlaps = []; areas = []; events = []; images = []; region_values = []
    component_values = []; base_labels = []; raw_regions = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            pre, post, location, damage, sample_names = parse_batch(batch, device)
            output = model(pre, post, return_dict=True)
            rgb = output["rgb_alignment_value"]; sar = output["sar_alignment_value"]
            aligned = output["aligned_sar_value"]; feature_size = rgb.shape[-2:]
            components = predicted_component_ids(output["localization_probability_low"], feature_size)
            ids = region_ids(
                pre, output["localization_probability_low"], feature_size,
                "r2_slico", args.segments, args.compactness, None,
            )
            ids = split_by_component(ids, components)
            raw_regions += sum(int(torch.unique(row).numel()) for row in ids)
            edge, orientation = rgb_orientation_maps(pre, feature_size)
            raw_sar = recover_sar_display(post, stats)
            diagnostics = {
                "entropy": output["alignment_entropy"],
                "expected_offset": output["alignment_offset"].float().square().sum(1, keepdim=True).sqrt(),
                "expected_dx": output["alignment_offset"][:, 0:1],
                "expected_dy": output["alignment_offset"][:, 1:2],
                "max_probability": output["alignment_confidence"],
                "match_margin": output["alignment_confidence"],
            }
            rows = extract_factorized_records(
                rgb=rgb, sar=sar, aligned_sar=aligned, raw_sar=raw_sar,
                ids=ids, components=components, p_building=output["localization_probability_low"],
                damage_probability=F.softmax(output["output_dam"].float(), dim=1),
                diagnostics=diagnostics, edge=edge, orientation=orientation,
                damage=damage, location=location, names=sample_names,
                events=[event_name(str(value)) for value in sample_names],
                min_query_building=args.min_query_building, min_write_building=0.01,
                min_purity=0.0, min_tokens=args.min_tokens, ring_radii=(2, 4),
                # Keep the complete predicted-building region set.  GT is used
                # later only to label the whole building for the audit; it must
                # not decide which local evidence Tokens exist.
                require_reliable_label=False,
            )
            sample_to_batch = {str(name): index for index, name in enumerate(sample_names)}
            for row in rows:
                for name in GROUP_NAMES:
                    grouped[name].append(getattr(row, name).cpu().half())
                batch_index = sample_to_batch[str(row.image_id)]
                region_mask = ids[batch_index] == int(row.region_id)
                grouped["sar_aligned"].append(
                    _pool_feature(aligned[batch_index], region_mask).cpu().half()
                )
                labels.append(row.hard_label); soft_labels.append(row.soft_label.cpu())
                purities.append(row.purity); overlaps.append(row.building_overlap); areas.append(row.area)
                events.append(row.event_id); images.append(row.image_id); region_values.append(row.region_id)
                component_values.append(row.component_id); base_labels.append(row.p3_region_label)
            if step % 25 == 0 or step == len(loader):
                print(f"flat records {step}/{len(loader)} kept={len(labels)}", flush=True)
    if not labels:
        raise RuntimeError("no building-internal SLICO records were produced")
    metadata = {
        "protocol": PROTOCOL + "_FLAT_RECORDS", "split": args.split,
        "source_list": str(Path(args.data_list).resolve()), "source_sha256": sha256_file(args.data_list),
        "image_count": len(dataset), "roi_lca": model_report, "roi_variant": args.roi_variant,
        "region_variant": "building_split_slico", "raw_region_count": raw_regions,
        "aligned_before_pooling": True, "aligned_sar_group": "sar_aligned",
        "split_by_predicted_component": True,
        "local_gt_selection": False,
        "gt_role": "audit labels only; never region construction or retrieval Key",
        "memory_write_allowed": args.split == "train", "test_accessed": False,
    }
    payload = {
        "protocol": PROTOCOL + "_FLAT_RECORDS",
        "groups": {name: torch.stack(value) for name, value in grouped.items()},
        "soft_labels": torch.stack(soft_labels), "labels": torch.tensor(labels),
        "purities": torch.tensor(purities), "building_overlaps": torch.tensor(overlaps),
        "areas": torch.tensor(areas), "events": events, "images": images,
        "region_ids": torch.tensor(region_values), "component_ids": torch.tensor(component_values),
        "p3_region_labels": torch.tensor(base_labels), "metadata": metadata, "test_accessed": False,
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination); print(json.dumps(metadata, indent=2, ensure_ascii=False)); print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
