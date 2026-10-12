from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels,
    predicted_building_label_map,
    selector_targets,
    utility_features,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import BASE_CHANNEL_NAMES, build_evidence_maps


PROTOCOL = "SPI_S2_EVIDENCE_SAMPLE_CACHE_v2"
VALID_GROUPS = ("direct", "relation", "matching", "grid", "slic")


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cap_binary_samples(
    labels: torch.Tensor,
    decisive: torch.Tensor,
    maximum_per_class: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = torch.zeros_like(decisive, dtype=torch.bool)
    for value in (0, 1):
        indices = torch.nonzero(decisive & (labels == value), as_tuple=False).flatten()
        if maximum_per_class > 0 and indices.numel() > maximum_per_class:
            order = torch.randperm(indices.numel(), generator=generator)[:maximum_per_class]
            indices = indices[order]
        selected[indices] = True
    return selected


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build Train/Calibration fixed-S2 evidence samples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "internal_calibration"), required=True)
    parser.add_argument("--groups", default=",".join(VALID_GROUPS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--maximum-per-class-per-image", type=int, default=2048)
    parser.add_argument("--matching-max-side", type=int, default=80)
    parser.add_argument("--matching-window", type=int, default=5)
    parser.add_argument("--cell-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)

    groups = tuple(dict.fromkeys(item.strip() for item in args.groups.split(",") if item.strip()))
    unknown = [name for name in groups if name not in VALID_GROUPS]
    if unknown:
        raise ValueError(f"unknown evidence groups: {unknown}")
    if "direct" not in groups:
        groups = ("direct",) + groups

    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    config = SelectiveContextConfig()
    generator = torch.Generator().manual_seed(args.seed)
    base_rows = []
    evidence_rows = {name: [] for name in groups}
    label_rows = []
    base_class_rows = []
    candidate_class_rows = []
    target_class_rows = []
    image_index_rows = []
    direction_rows = []
    image_names = []
    image_events = []
    base_histograms = []
    fixed_histograms = []
    channel_names = {}
    partition_records = {"grid": [], "slic": []}
    counts = {"changed": 0, "corrected": 0, "harmed": 0, "neutral": 0, "stored": 0}
    started = time.perf_counter()
    processed = 0

    for image_index, (rgb, sar, names) in enumerate(loader):
        if args.max_images > 0 and image_index >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits, auxiliary = p3(rgb, sar, return_aux=True)
        base_probability_cpu = damage_logits[0].float().softmax(0).cpu()
        base_prediction_cpu = base_probability_cpu.argmax(0)
        localization = loc_logits[0].argmax(0).cpu()
        region_map = predicted_building_label_map(localization, args.minimum_area)
        candidate = fixed_s2_candidate_from_labels(base_probability_cpu, region_map, config)
        candidate_prediction_cpu = candidate.probabilities.argmax(0)
        name = str(names[0])
        event = name.rsplit("_", 1)[0]
        target_cpu = load_semantic_target(args.dataset_path, name).long()

        base_device = base_probability_cpu[None].to(device)
        candidate_device = candidate.probabilities[None].to(device)
        support_device = candidate.support[None, None].to(device)
        region_features = candidate.region_features[None].to(device)
        base_features, changed_device = utility_features(
            base_device, candidate_device, support_device, region_features
        )
        evidence = build_evidence_maps(
            rgb,
            auxiliary["rgb_feature"],
            auxiliary["sar_feature"],
            tuple(base_prediction_cpu.shape),
            groups=groups,
            matching_max_side=args.matching_max_side,
            matching_window=args.matching_window,
            cell_size=args.cell_size,
        )
        for evidence_name, item in evidence.items():
            channel_names[evidence_name] = item.channel_names
            if evidence_name in partition_records:
                partition_records[evidence_name].append(item.metadata)

        selector_target, decisive, positive, negative = selector_targets(
            base_prediction_cpu[None], candidate_prediction_cpu[None], target_cpu[None]
        )
        changed_cpu = changed_device[0, 0].cpu() & (target_cpu != 255)
        positive_cpu = positive[0] & changed_cpu
        negative_cpu = negative[0] & changed_cpu
        decisive_cpu = positive_cpu | negative_cpu
        neutral_cpu = changed_cpu & ~decisive_cpu
        row_label_map = torch.full_like(target_cpu, -1, dtype=torch.int8)
        row_label_map[negative_cpu] = 0
        row_label_map[positive_cpu] = 1

        counts["changed"] += int(changed_cpu.sum())
        counts["corrected"] += int(positive_cpu.sum())
        counts["harmed"] += int(negative_cpu.sum())
        counts["neutral"] += int(neutral_cpu.sum())
        if args.split == "train":
            flat_labels = row_label_map.reshape(-1)
            flat_decisive = decisive_cpu.reshape(-1)
            flat_selected = cap_binary_samples(
                flat_labels,
                flat_decisive,
                args.maximum_per_class_per_image,
                generator,
            )
            selected_cpu = flat_selected.reshape_as(changed_cpu)
        else:
            selected_cpu = changed_cpu
        selected_device = selected_cpu.to(device)
        selected_count = int(selected_cpu.sum())
        counts["stored"] += selected_count
        if selected_count:
            base_rows.append(base_features[0].permute(1, 2, 0)[selected_device].cpu().half())
            for evidence_name, item in evidence.items():
                evidence_rows[evidence_name].append(
                    item.values[0].permute(1, 2, 0)[selected_device].cpu().half()
                )
            label_rows.append(row_label_map[selected_cpu].cpu())
            base_class_rows.append(base_prediction_cpu[selected_cpu].to(torch.uint8))
            candidate_class_rows.append(candidate_prediction_cpu[selected_cpu].to(torch.uint8))
            target_class_rows.append(target_cpu[selected_cpu].clamp(0, 255).to(torch.uint8))
            image_index_rows.append(torch.full((selected_count,), image_index, dtype=torch.int32))
            direction_rows.append(
                (4 * base_prediction_cpu[selected_cpu] + candidate_prediction_cpu[selected_cpu]).to(torch.uint8)
            )

        image_names.append(name)
        image_events.append(event)
        base_histograms.append(fast_histogram(base_prediction_cpu, target_cpu).long())
        fixed_histograms.append(fast_histogram(candidate_prediction_cpu, target_cpu).long())
        processed = image_index + 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Evidence cache {args.split}: {processed}/{len(dataset)} | "
                f"{processed/max(elapsed, 1e-9):.2f} img/s | stored={counts['stored']}",
                flush=True,
            )

    if not base_rows:
        raise RuntimeError("no fixed-S2 changed samples were stored")
    payload = {
        "protocol": PROTOCOL,
        "split": args.split,
        "source_list": str(Path(args.list_path).resolve()),
        "processed_images": processed,
        **provenance,
        "groups": list(groups),
        "base_channel_names": BASE_CHANNEL_NAMES,
        "evidence_channel_names": channel_names,
        "counts": counts,
        "features": torch.cat(base_rows),
        "evidence": {name: torch.cat(rows) for name, rows in evidence_rows.items()},
        "selector_label": torch.cat(label_rows),
        "base_class": torch.cat(base_class_rows),
        "candidate_class": torch.cat(candidate_class_rows),
        "target_class": torch.cat(target_class_rows),
        "image_index": torch.cat(image_index_rows),
        "direction_code": torch.cat(direction_rows),
        "image_names": image_names,
        "image_events": image_events,
        "base_histograms": torch.stack(base_histograms),
        "fixed_histograms": torch.stack(fixed_histograms),
        "partition_records": partition_records,
        "configuration": vars(args),
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output)
    manifest = {
        key: payload[key]
        for key in (
            "protocol", "split", "source_list", "processed_images", "groups", "counts",
            "validation_accessed", "official_instances_accessed", "test_accessed"
        )
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
