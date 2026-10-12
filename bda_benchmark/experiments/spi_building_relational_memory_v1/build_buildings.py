#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_building_relational_memory_v1.core import PROTOCOL, audit_contract


def _select_nodes(payload: dict, indices: list[int], maximum: int) -> list[int]:
    """Keep strong relation evidence and spatial coverage without using labels/P3."""
    if len(indices) <= maximum:
        return sorted(indices)
    index = torch.tensor(indices, dtype=torch.long)
    relation = payload["groups"]["relation_aligned"][index].float()
    position = payload["groups"]["geometry"][index, 4:6].float()
    anomaly = relation.norm(dim=1)
    strong_count = max(1, maximum // 2)
    strong_local = anomaly.topk(strong_count).indices.tolist()
    selected_local = list(strong_local)
    while len(selected_local) < maximum:
        remaining = [value for value in range(len(indices)) if value not in selected_local]
        if not remaining:
            break
        chosen_position = position[selected_local]
        candidate_position = position[remaining]
        distance = torch.cdist(candidate_position, chosen_position).min(1).values
        selected_local.append(remaining[int(distance.argmax().item())])
    return sorted(indices[value] for value in selected_local)


def _weighted_label(payload: dict, indices: list[int]) -> tuple[torch.Tensor, int, float, int]:
    index = torch.tensor(indices, dtype=torch.long)
    area = payload["areas"][index].float().clamp_min(1.0)
    # Only GT-valid pixels contribute to the building audit label.  Crucially,
    # this weight is not used for Token selection or retrieval features.
    valid_weight = area * payload["building_overlaps"][index].float().clamp(0.0, 1.0)
    soft = (
        payload["soft_labels"][index].float() * valid_weight[:, None]
    ).sum(0) / valid_weight.sum().clamp_min(1.0)
    purity, offset = soft.max(0)
    p3_labels = payload["p3_region_labels"][index].long()
    counts = torch.stack([(area * (p3_labels == label)).sum() for label in (1, 2, 3)])
    return soft, int(offset.item()) + 1, float(purity.item()), int(counts.argmax().item()) + 1


def _size_boundaries(areas: torch.Tensor) -> list[float]:
    value = areas.float()
    return [float(torch.quantile(value, q).item()) for q in (0.25, 0.75, 0.95)]


def _size_bin(area: float, boundaries: list[float]) -> int:
    return sum(float(area) > value for value in boundaries)


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate aligned superpixels into predicted-building relation sets")
    parser.add_argument("--region_records", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size_bins_json", required=True)
    parser.add_argument("--max_tokens", type=int, default=24)
    parser.add_argument("--min_tokens", type=int, default=2)
    parser.add_argument("--building_purity", type=float, default=0.80)
    args = parser.parse_args()
    if "test" in Path(args.region_records).name.lower():
        raise RuntimeError("Test access is forbidden in V1")
    if args.max_tokens < args.min_tokens or args.min_tokens < 1:
        raise ValueError("invalid token limits")

    source = torch.load(args.region_records, map_location="cpu", weights_only=False)
    groups = source.get("groups")
    if not isinstance(groups, dict):
        raise RuntimeError("region records have no factorized groups")
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (image, component) in enumerate(zip(source["images"], source["component_ids"].tolist())):
        if int(component) > 0:
            buckets[(str(image), int(component))].append(index)

    candidates = []
    for (image, component), indices in sorted(buckets.items()):
        if len(indices) < args.min_tokens:
            continue
        soft, label, purity, p3_label = _weighted_label(source, indices)
        if purity < args.building_purity:
            continue
        selected = _select_nodes(source, indices, args.max_tokens)
        area = float(source["areas"][torch.tensor(indices)].sum().item())
        candidates.append((image, component, selected, soft, label, purity, p3_label, area))
    if not candidates:
        raise RuntimeError("no reliable predicted-building sets were produced")

    bin_path = Path(args.size_bins_json)
    if args.split == "train":
        boundaries = _size_boundaries(torch.tensor([row[-1] for row in candidates]))
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        bin_path.write_text(json.dumps({"boundaries": boundaries, "unit": "feature_grid_area"}, indent=2) + "\n")
    else:
        if not bin_path.is_file():
            raise RuntimeError(f"missing Train-fitted size bins: {bin_path}")
        boundaries = list(map(float, json.loads(bin_path.read_text())["boundaries"]))

    max_tokens = args.max_tokens
    padded = {
        name: torch.zeros((len(candidates), max_tokens, value.shape[1]), dtype=value.dtype)
        for name, value in groups.items()
    }
    mask = torch.zeros((len(candidates), max_tokens), dtype=torch.bool)
    labels, soft_labels, purities, areas, size_bins, events, images, components, p3_labels = [], [], [], [], [], [], [], [], []
    event_by_image = {}
    for image, event in zip(source["images"], source["events"]):
        event_by_image[str(image)] = str(event)
    for building_index, row in enumerate(candidates):
        image, component, selected, soft, label, purity, p3_label, area = row
        take = len(selected)
        index = torch.tensor(selected, dtype=torch.long)
        for name, value in groups.items():
            padded[name][building_index, :take] = value[index]
        mask[building_index, :take] = True
        labels.append(label); soft_labels.append(soft); purities.append(purity); areas.append(area)
        size_bins.append(_size_bin(area, boundaries)); events.append(event_by_image[image])
        images.append(image); components.append(component); p3_labels.append(p3_label)

    metadata = {
        "protocol": PROTOCOL,
        "split": args.split,
        "source_records": str(Path(args.region_records).resolve()),
        "source_sha256": sha256_file(args.region_records),
        "building_key_groups": ["rgb", "sar_aligned", "relation_aligned", "multiring"],
        "rgb_role": "within-building standardized structural context; m0 keeps absolute RGB as the style control",
        "quality_role": "relation-attention reliability only; never concatenated into semantic content",
        "p3_state_role": "future Utility Verifier only; forbidden from retrieval Key",
        "region_reference_frame": "pre-event RGB",
        "sar_transfer": "ROI-LCA aligned SAR is pooled after transfer to the RGB grid",
        "relation_graph": "relative-position kNN over superpixels within each predicted building",
        "token_selection": "half strongest aligned-relation evidence plus spatial farthest coverage",
        "max_tokens": max_tokens,
        "size_boundaries": boundaries,
        "memory_write_allowed": args.split == "train",
        "test_accessed": False,
    }
    audit_contract(metadata, args.split)
    output = {
        "protocol": PROTOCOL + "_BUILDINGS",
        "groups": padded,
        "mask": mask,
        "labels": torch.tensor(labels),
        "soft_labels": torch.stack(soft_labels),
        "purities": torch.tensor(purities),
        "areas": torch.tensor(areas),
        "size_bins": torch.tensor(size_bins),
        "events": events,
        "images": images,
        "component_ids": torch.tensor(components),
        "p3_building_labels": torch.tensor(p3_labels),
        "metadata": metadata,
        "test_accessed": False,
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print(json.dumps({**metadata, "building_count": len(candidates)}, indent=2, ensure_ascii=False))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
