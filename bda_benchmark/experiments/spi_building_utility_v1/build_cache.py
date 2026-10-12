from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_s2_evidence_v2.core import build_evidence_maps
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    direction_policy_mask,
    fixed_s2_candidate_from_labels,
    predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import fast_histogram, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import (
    SEMANTIC_NAMES,
    S2_NAMES,
    SHAPE_NAMES,
    building_feature_groups,
    evidence_feature_names,
    oracle_predictions,
    utility_counts,
)


PROTOCOL = "SPI_BUILDING_UTILITY_CACHE_v1"
EVIDENCE_GROUPS = ("direct", "relation", "matching", "grid", "slic")


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_policy(path: str) -> dict[str, bool] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != "SPI_S2_DIRECTION_POLICY_TRAIN_ONLY_v1":
        raise RuntimeError("unexpected direction-policy protocol")
    if (
        payload.get("validation_accessed", True)
        or payload.get("official_instances_accessed", True)
        or payload.get("test_accessed", True)
    ):
        raise RuntimeError("direction policy accessed a forbidden split")
    return {str(key): bool(value) for key, value in payload["direction_policy"]["policy"].items()}


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build one BU-E0 building cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration", "meta_confirm"), required=True)
    parser.add_argument("--direction-policy-json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--matching-max-side", type=int, default=80)
    parser.add_argument("--matching-window", type=int, default=5)
    parser.add_argument("--cell-size", type=int, default=4)
    parser.add_argument("--evidence-groups", default=",".join(EVIDENCE_GROUPS))
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)

    evidence_groups = tuple(
        dict.fromkeys(value.strip() for value in args.evidence_groups.split(",") if value.strip())
    )
    unknown = [name for name in evidence_groups if name not in EVIDENCE_GROUPS]
    if unknown:
        raise ValueError(f"unknown evidence groups: {unknown}")
    policy = load_policy(args.direction_policy_json)
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
    feature_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("shape", "semantic", "s2", *evidence_groups)
    }
    feature_names: dict[str, tuple[str, ...]] = {
        "shape": SHAPE_NAMES, "semantic": SEMANTIC_NAMES, "s2": S2_NAMES,
    }
    utility_rows = {name: [] for name in ("changed", "corrected", "harmed", "neutral", "net", "utility_ratio")}
    image_index_rows, region_label_rows = [], []
    image_names, image_events = [], []
    histograms = {name: [] for name in ("p3", "fixed_s2", "building_oracle", "pixel_oracle")}
    if policy is not None:
        histograms["direction_policy"] = []
    counts = {"regions_all": 0, "regions_changed": 0, "changed": 0, "corrected": 0, "harmed": 0, "neutral": 0}
    started = time.perf_counter()
    processed = 0

    for image_index, (rgb, sar, names) in enumerate(loader):
        if args.max_images > 0 and image_index >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits, auxiliary = p3(rgb, sar, return_aux=True)
        base_probabilities = damage_logits[0].float().softmax(0).cpu()
        base_prediction = base_probabilities.argmax(0)
        localization = loc_logits[0].argmax(0).cpu()
        region_map = predicted_building_label_map(localization, args.minimum_area)
        candidate = fixed_s2_candidate_from_labels(base_probabilities, region_map, config)
        candidate_probabilities = candidate.probabilities.cpu()
        candidate_prediction = candidate_probabilities.argmax(0)
        sample_name = str(names[0])
        target = load_semantic_target(args.dataset_path, sample_name).long()
        evidence_maps = build_evidence_maps(
            rgb, auxiliary["rgb_feature"], auxiliary["sar_feature"],
            tuple(base_prediction.shape), groups=evidence_groups,
            matching_max_side=args.matching_max_side,
            matching_window=args.matching_window, cell_size=args.cell_size,
        )
        evidence_cpu = {name: item.values[0].float().cpu() for name, item in evidence_maps.items()}
        for evidence_name, item in evidence_maps.items():
            feature_names[evidence_name] = evidence_feature_names(item.channel_names)

        building_oracle, pixel_oracle = oracle_predictions(
            base_prediction, candidate_prediction, target, region_map.labels
        )
        histograms["p3"].append(fast_histogram(base_prediction, target).long())
        histograms["fixed_s2"].append(fast_histogram(candidate_prediction, target).long())
        histograms["building_oracle"].append(fast_histogram(building_oracle, target).long())
        histograms["pixel_oracle"].append(fast_histogram(pixel_oracle, target).long())
        if policy is not None:
            accept = direction_policy_mask(base_prediction, candidate_prediction, policy)
            direction_prediction = torch.where(accept, candidate_prediction, base_prediction)
            histograms["direction_policy"].append(fast_histogram(direction_prediction, target).long())

        counts["regions_all"] += region_map.count
        for label, record in enumerate(candidate.records, start=1):
            if int(record["changed_pixels"]) <= 0:
                continue
            mask = region_map.labels == label
            utility = utility_counts(base_prediction, candidate_prediction, target, mask)
            if int(utility["changed"]) <= 0:
                continue
            groups = building_feature_groups(
                mask=mask,
                base_probabilities=base_probabilities,
                candidate_probabilities=candidate_probabilities,
                base_prediction=base_prediction,
                candidate_prediction=candidate_prediction,
                record=record,
                evidence=evidence_cpu,
            )
            for group, values in groups.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"non-finite {group} building features")
                feature_rows[group].append(values.float())
            for key in utility_rows:
                utility_rows[key].append(utility[key])
            image_index_rows.append(image_index)
            region_label_rows.append(label)
            counts["regions_changed"] += 1
            for key in ("changed", "corrected", "harmed", "neutral"):
                counts[key] += int(utility[key])

        image_names.append(sample_name)
        image_events.append(sample_name.rsplit("_", 1)[0])
        processed = image_index + 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"BU cache {args.split}: {processed}/{len(dataset)} | "
                f"{processed/max(elapsed, 1e-9):.2f} img/s | buildings={counts['regions_changed']}",
                flush=True,
            )

    if not image_index_rows:
        raise RuntimeError("no changed fixed-S2 buildings were found")
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
        "image_index": torch.tensor(image_index_rows, dtype=torch.int32),
        "region_label": torch.tensor(region_label_rows, dtype=torch.int32),
        "image_names": image_names,
        "image_events": image_events,
        "histograms": {name: torch.stack(rows) for name, rows in histograms.items()},
        "counts": counts,
        "direction_policy_used": policy is not None,
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
        "feature_dimensions": {name: int(values.shape[1]) for name, values in payload["features"].items()},
        "validation_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
