#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    event_name,
    parse_batch,
    read_list,
    sha256_file,
)
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    PROTOCOL,
    extract_factorized_records,
    local_alignment_diagnostics,
    make_region_ids,
    predicted_component_ids,
    recover_sar_display,
    rgb_orientation_maps,
    split_regions_by_building,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


GROUP_NAMES = (
    "rgb",
    "sar",
    "relation_unaligned",
    "relation_aligned",
    "multiring",
    "geometry",
    "quality",
    "p3_state",
)


def main() -> None:
    parser = argparse.ArgumentParser("Build maximal factorized RGB-SAR region-token records")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region_mode", choices=("grid", "slic", "slico", "shuffled_slico"), default="slico")
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--alignment_radius", type=int, default=1)
    parser.add_argument("--purity", type=float, default=0.80)
    parser.add_argument("--min_query_building", type=float, default=0.60)
    parser.add_argument("--min_write_building", type=float, default=0.70)
    parser.add_argument("--min_tokens", type=int, default=4)
    parser.add_argument("--ring8_radius", type=int, default=2,
                        help="feature-grid radius corresponding to about 8 input pixels")
    parser.add_argument("--ring16_radius", type=int, default=4,
                        help="feature-grid radius corresponding to about 16 input pixels")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if "test" in Path(args.data_list).name.lower():
        raise RuntimeError("K2 record construction forbids test-list access")
    if args.ring8_radius >= args.ring16_radius:
        raise ValueError("ring8_radius must be smaller than ring16_radius")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = json.loads(Path(args.sar_stats_json).read_text(encoding="utf-8"))
    model = DamageFormer_S8_MICB_DSMIL().to(device).eval()
    parent = load_parent(model, args.parent_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    names = read_list(args.data_list)
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=640, type="val")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    grouped: dict[str, list[torch.Tensor]] = {name: [] for name in GROUP_NAMES}
    soft_labels: list[torch.Tensor] = []
    labels: list[int] = []
    purities: list[float] = []
    overlaps: list[float] = []
    areas: list[int] = []
    events: list[str] = []
    images: list[str] = []
    region_ids: list[int] = []
    component_ids: list[int] = []
    p3_region_labels: list[int] = []
    raw_region_count = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            pre, post, location, damage, sample_names = parse_batch(batch, device)
            _, damage_logits, aux = model(pre, post, return_aux=True)
            head = model.micb_dsmil

            # These are modality-specific pre-classification maps.  The fused
            # P3 feature and final probabilities are deliberately excluded from
            # the K2 content branches.
            rgb = head.rgb_proj(aux["rgb_feature"])
            sar = head.sar_proj(aux["sar_feature"])
            aligned_sar, diagnostics = local_alignment_diagnostics(
                rgb, sar, radius=args.alignment_radius
            )
            feature_size = rgb.shape[-2:]
            edge, orientation = rgb_orientation_maps(pre, feature_size)
            raw_sar = recover_sar_display(post, stats)
            base_ids = make_region_ids(
                pre,
                feature_size,
                mode=args.region_mode,
                n_segments=args.segments,
                compactness=args.compactness,
            )
            ids = split_regions_by_building(base_ids, aux["p_building"])
            components = predicted_component_ids(aux["p_building"], feature_size)
            raw_region_count += sum(int(torch.unique(row).numel()) for row in ids)

            rows = extract_factorized_records(
                rgb=rgb,
                sar=sar,
                aligned_sar=aligned_sar,
                raw_sar=raw_sar,
                ids=ids,
                components=components,
                p_building=aux["p_building"],
                damage_probability=F.softmax(damage_logits.float(), dim=1),
                diagnostics=diagnostics,
                edge=edge,
                orientation=orientation,
                damage=damage,
                location=location,
                names=sample_names,
                events=[event_name(str(value)) for value in sample_names],
                min_query_building=args.min_query_building,
                min_write_building=args.min_write_building,
                min_purity=args.purity,
                min_tokens=args.min_tokens,
                ring_radii=(args.ring8_radius, args.ring16_radius),
                require_reliable_label=True,
            )
            for row in rows:
                for name in GROUP_NAMES:
                    grouped[name].append(getattr(row, name).detach().cpu().half())
                soft_labels.append(row.soft_label.detach().cpu().float())
                labels.append(row.hard_label)
                purities.append(row.purity)
                overlaps.append(row.building_overlap)
                areas.append(row.area)
                events.append(row.event_id)
                images.append(row.image_id)
                region_ids.append(row.region_id)
                component_ids.append(row.component_id)
                p3_region_labels.append(row.p3_region_label)
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(f"records {batch_index}/{len(loader)} kept={len(labels)}", flush=True)

    if not labels:
        raise RuntimeError("no reliable building-conditioned region records were found")
    groups = {name: torch.stack(values) for name, values in grouped.items()}
    group_dims = {name: int(value.shape[1]) for name, value in groups.items()}
    payload = {
        "protocol": PROTOCOL + "_MAXIMAL_RECORDS",
        "groups": groups,
        "soft_labels": torch.stack(soft_labels),
        "labels": torch.tensor(labels, dtype=torch.long),
        "purities": torch.tensor(purities, dtype=torch.float32),
        "building_overlaps": torch.tensor(overlaps, dtype=torch.float32),
        "areas": torch.tensor(areas, dtype=torch.long),
        "events": events,
        "images": images,
        "region_ids": torch.tensor(region_ids, dtype=torch.long),
        "component_ids": torch.tensor(component_ids, dtype=torch.long),
        "p3_region_labels": torch.tensor(p3_region_labels, dtype=torch.long),
        "metadata": {
            "source_list": str(Path(args.data_list).resolve()),
            "source_list_sha256": sha256_file(args.data_list),
            "image_count": len(names),
            "raw_region_count": raw_region_count,
            "kept_record_count": len(labels),
            "group_dimensions": group_dims,
            "region_mode": args.region_mode,
            "building_condition": "predicted P3 soft building map; region split at 0.5",
            "segments_at_640": args.segments,
            "compactness": args.compactness,
            "alignment_radius_feature": args.alignment_radius,
            "alignment_radius_input_approx": args.alignment_radius * 4,
            "ring_radii_feature": [args.ring8_radius, args.ring16_radius],
            "ring_radii_input_approx": [args.ring8_radius * 4, args.ring16_radius * 4],
            "purity_threshold": args.purity,
            "min_query_building": args.min_query_building,
            "min_write_building": args.min_write_building,
            "sar_stats_json": str(Path(args.sar_stats_json).resolve()),
            "sar_stats_sha256": sha256_file(args.sar_stats_json),
            "parent": parent,
            "key_contract": {
                "content": ["rgb", "sar", "relation_unaligned/aligned", "multiring for T3"],
                "geometry": "reranking only",
                "quality": "gating only",
                "p3_state": "verifier only; forbidden in T1-T3 Key",
            },
            "test_accessed": False,
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
