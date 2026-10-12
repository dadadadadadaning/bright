from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_building_utility_v1.core import (
    oracle_predictions,
    utility_counts,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels,
    predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import fast_histogram, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import (
    PROTOCOL,
    ZONE_GEOMETRY_NAMES,
    ZONE_SEMANTIC_NAMES,
    ZONE_S2_NAMES,
    deterministic_random_core,
    split_core_ring,
    update_zone_oracle,
    zone_feature_groups,
)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build Core-Ring Utility E0 cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--core-radius", type=int, default=1)
    parser.add_argument("--minimum-core-area", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    config = SelectiveContextConfig()
    feature_names = {
        "zone_geometry": ZONE_GEOMETRY_NAMES,
        "zone_semantic": ZONE_SEMANTIC_NAMES,
        "zone_s2": ZONE_S2_NAMES,
    }
    feature_rows = {name: [] for name in feature_names}
    utility_rows = {name: [] for name in ("changed", "corrected", "harmed", "neutral", "net", "utility_ratio")}
    zone_type_rows, image_index_rows, region_label_rows = [], [], []
    image_names, image_events = [], []
    histogram_names = (
        "p3", "fixed_s2", "core_only", "ring_only", "random_core_only", "random_ring_only",
        "building_oracle", "core_ring_oracle", "pixel_oracle",
    )
    histograms = {name: [] for name in histogram_names}
    zone_totals = {
        zone: {key: 0 for key in ("area", "changed", "corrected", "harmed", "neutral", "net")}
        for zone in ("core", "ring")
    }
    counts = {
        "regions_all": 0, "regions_with_core": 0, "regions_without_core": 0,
        "zone_rows": 0,
    }
    started = time.perf_counter()
    processed = 0

    for image_index, (rgb, sar, names) in enumerate(loader):
        if args.max_images > 0 and image_index >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits = p3(rgb, sar)
        base_probabilities = damage_logits[0].float().softmax(0).cpu()
        base_prediction = base_probabilities.argmax(0)
        localization = loc_logits[0].argmax(0).cpu()
        region_map = predicted_building_label_map(localization, args.minimum_area)
        candidate = fixed_s2_candidate_from_labels(base_probabilities, region_map, config)
        candidate_probabilities = candidate.probabilities.cpu()
        candidate_prediction = candidate_probabilities.argmax(0)
        sample_name = str(names[0])
        target = load_semantic_target(args.dataset_path, sample_name).long()

        building_oracle, pixel_oracle = oracle_predictions(
            base_prediction, candidate_prediction, target, region_map.labels
        )
        core_only = base_prediction.clone()
        ring_only = base_prediction.clone()
        random_core_only = base_prediction.clone()
        random_ring_only = base_prediction.clone()
        core_ring_oracle = base_prediction.clone()
        counts["regions_all"] += region_map.count

        for label in range(1, region_map.count + 1):
            building_mask = region_map.labels == label
            core, ring, core_available = split_core_ring(
                building_mask, args.core_radius, args.minimum_core_area
            )
            if core_available:
                counts["regions_with_core"] += 1
            else:
                counts["regions_without_core"] += 1
            random_core = deterministic_random_core(
                building_mask, int(core.sum()),
                args.seed + image_index * 100003 + label,
            )
            core_only[core] = candidate_prediction[core]
            ring_only[ring] = candidate_prediction[ring]
            random_core_only[random_core] = candidate_prediction[random_core]
            random_ring = building_mask & ~random_core
            random_ring_only[random_ring] = candidate_prediction[random_ring]

            for zone_name, zone_mask, is_core in (
                ("core", core, True), ("ring", ring, False)
            ):
                if not bool(zone_mask.any()):
                    continue
                zone_totals[zone_name]["area"] += int(zone_mask.sum())
                utility = update_zone_oracle(
                    core_ring_oracle, base_prediction, candidate_prediction, target, zone_mask
                )
                for key in ("changed", "corrected", "harmed", "neutral", "net"):
                    zone_totals[zone_name][key] += int(utility[key])
                if int(utility["changed"]) <= 0:
                    continue
                groups = zone_feature_groups(
                    zone_mask=zone_mask, building_mask=building_mask, is_core=is_core,
                    base_probabilities=base_probabilities,
                    candidate_probabilities=candidate_probabilities,
                    base_prediction=base_prediction,
                    candidate_prediction=candidate_prediction,
                )
                for group, values in groups.items():
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(f"non-finite {group} features")
                    feature_rows[group].append(values.float())
                for key in utility_rows:
                    utility_rows[key].append(utility[key])
                zone_type_rows.append(1 if is_core else 0)
                image_index_rows.append(image_index)
                region_label_rows.append(label)
                counts["zone_rows"] += 1

        predictions = {
            "p3": base_prediction,
            "fixed_s2": candidate_prediction,
            "core_only": core_only,
            "ring_only": ring_only,
            "random_core_only": random_core_only,
            "random_ring_only": random_ring_only,
            "building_oracle": building_oracle,
            "core_ring_oracle": core_ring_oracle,
            "pixel_oracle": pixel_oracle,
        }
        for name, prediction in predictions.items():
            histograms[name].append(fast_histogram(prediction, target).long())
        image_names.append(sample_name)
        image_events.append(sample_name.rsplit("_", 1)[0])
        processed = image_index + 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Core-Ring cache {args.split}: {processed}/{len(dataset)} | "
                f"{processed/max(elapsed, 1e-9):.2f} img/s | zones={counts['zone_rows']}",
                flush=True,
            )

    if not image_index_rows:
        raise RuntimeError("no changed core/ring zones were found")
    payload = {
        "protocol": PROTOCOL,
        "split": args.split,
        "source_list": str(Path(args.list_path).resolve()),
        "processed_images": processed,
        **provenance,
        "s2_configuration": config.__dict__,
        "configuration": vars(args),
        "feature_names": feature_names,
        "features": {name: torch.stack(rows).half() for name, rows in feature_rows.items()},
        "utility": {
            "changed": torch.tensor(utility_rows["changed"], dtype=torch.int32),
            "corrected": torch.tensor(utility_rows["corrected"], dtype=torch.int32),
            "harmed": torch.tensor(utility_rows["harmed"], dtype=torch.int32),
            "neutral": torch.tensor(utility_rows["neutral"], dtype=torch.int32),
            "net": torch.tensor(utility_rows["net"], dtype=torch.int32),
            "utility_ratio": torch.tensor(utility_rows["utility_ratio"], dtype=torch.float32),
        },
        "zone_type": torch.tensor(zone_type_rows, dtype=torch.uint8),
        "image_index": torch.tensor(image_index_rows, dtype=torch.int32),
        "region_label": torch.tensor(region_label_rows, dtype=torch.int32),
        "image_names": image_names,
        "image_events": image_events,
        "histograms": {name: torch.stack(rows) for name, rows in histograms.items()},
        "zone_totals": zone_totals,
        "counts": counts,
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, output)
    manifest = {
        "protocol": PROTOCOL, "split": args.split, "processed_images": processed,
        "source_list": payload["source_list"], "counts": counts,
        "zone_totals": zone_totals,
        "feature_dimensions": {name: int(values.shape[1]) for name, values in payload["features"].items()},
        "validation_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
