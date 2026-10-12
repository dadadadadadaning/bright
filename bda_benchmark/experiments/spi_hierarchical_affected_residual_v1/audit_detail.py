from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels, predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram, file_sha256, metrics_from_histogram, p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import DETAIL_PROTOCOL
from .core import (
    affected_reachable_mask, building_core_and_ring, minimum_affected_residual,
    severity_choice,
)


def _delta_histogram(old: torch.Tensor, new: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & (target >= 0) & (target <= 3)
    truth = target[valid].long()
    before = torch.bincount(truth * 4 + old[valid].long(), minlength=16).reshape(4, 4)
    after = torch.bincount(truth * 4 + new[valid].long(), minlength=16).reshape(4, 4)
    return after.long() - before.long()


def _quantiles(values: list[torch.Tensor]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    row = torch.cat(values).float()
    levels = torch.tensor((0.25, 0.50, 0.75, 0.90, 0.95))
    result = {f"q{int(q * 100)}": float(v) for q, v in zip(levels, torch.quantile(row, levels))}
    result.update({"count": int(row.numel()), "mean": float(row.mean()), "maximum": float(row.max())})
    return result


def _concentration(counts: list[int]) -> dict[str, float | int]:
    if not counts or sum(counts) == 0:
        return {"units": 0, "pixels": 0, "largest_share": 0.0, "top_1_percent_share": 0.0}
    ordered = sorted(counts, reverse=True)
    total = sum(ordered)
    top = max(1, math.ceil(len(ordered) * 0.01))
    return {
        "units": len(ordered), "pixels": total,
        "largest_share": ordered[0] / total,
        "top_1_percent_share": sum(ordered[:top]) / total,
        "mean_pixels": total / len(ordered),
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("AO-D0 reachable-space detail audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--oracle-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--minimum-building-area", type=int, default=32)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument("--small-building-maximum", type=int, default=64)
    parser.add_argument("--medium-building-maximum", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--maximum-largest-instance-share", type=float, default=0.10)
    parser.add_argument("--maximum-top1pct-instance-share", type=float, default=0.50)
    parser.add_argument("--maximum-background-source-fraction", type=float, default=0.50)
    args = parser.parse_args(argv)

    oracle = json.loads(Path(args.oracle_json).read_text(encoding="utf-8"))
    if args.max_images <= 0 and oracle.get("decision") != "ADMIT_AO_E1_EVIDENCE_AUDIT":
        raise RuntimeError("AO-D0 refused: AO-E0 Oracle was not admitted")
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("refusing repeated AO-D0 output")
    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, got {len(dataset)}")
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    config = SelectiveContextConfig()
    fixed_hist = torch.zeros(4, 4, dtype=torch.long)
    deltas: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    residuals: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    building_counts, image_counts, event_counts = [], [], defaultdict(int)

    for index, (rgb_cpu, sar_cpu, names) in enumerate(loader):
        if index >= stop:
            break
        rgb = rgb_cpu.to(device, non_blocking=True)
        sar = sar_cpu.to(device, non_blocking=True)
        localization, damage = p3(rgb, p3_sar_view(sar))
        probabilities = damage[0].float().softmax(0).cpu()
        regions = predicted_building_label_map(
            localization[0].argmax(0).cpu(), args.minimum_building_area
        )
        fixed_probabilities = fixed_s2_candidate_from_labels(probabilities, regions, config).probabilities.float().cpu()
        fixed = fixed_probabilities.argmax(0)
        severity = severity_choice(fixed_probabilities)
        target = load_semantic_target(args.dataset_path, str(names[0])).long()
        reachable = affected_reachable_mask(fixed_probabilities, regions.labels > 0, target)
        promoted = fixed.clone(); promoted[reachable] = severity[reachable]
        fixed_hist += fast_histogram(fixed, target).long()
        required = minimum_affected_residual(fixed_probabilities)
        core, ring = building_core_and_ring(regions.labels, args.core_radius)
        region_areas = torch.bincount(regions.labels.flatten().long(), minlength=regions.count + 1)
        area_map = region_areas[regions.labels.long()]
        event = str(names[0]).rsplit("_", 1)[0]
        image_count = int(reachable.sum())
        image_counts.append(image_count); event_counts[event] += image_count
        for region_id in range(1, regions.count + 1):
            amount = int((reachable & (regions.labels == region_id)).sum())
            if amount:
                building_counts.append(amount)

        masks = {
            "source": {"background": reachable & (fixed == 0), "intact": reachable & (fixed == 1)},
            "target": {"damaged": reachable & (target == 2), "destroyed": reachable & (target == 3)},
            "direction": {
                "background_to_damaged": reachable & (fixed == 0) & (target == 2),
                "background_to_destroyed": reachable & (fixed == 0) & (target == 3),
                "intact_to_damaged": reachable & (fixed == 1) & (target == 2),
                "intact_to_destroyed": reachable & (fixed == 1) & (target == 3),
            },
            "position": {"core": reachable & core, "ring": reachable & ring},
            "building_size": {
                "small": reachable & (area_map <= args.small_building_maximum),
                "medium": reachable & (area_map > args.small_building_maximum) & (area_map <= args.medium_building_maximum),
                "large": reachable & (area_map > args.medium_building_maximum),
            },
        }
        for family, rows in masks.items():
            for name, mask in rows.items():
                count = int(mask.sum())
                counts[family][name] += count
                if count:
                    current = _delta_histogram(fixed, promoted, target, mask)
                    deltas[family][name] = deltas[family].get(name, torch.zeros(4, 4, dtype=torch.long)) + current
                    residuals[f"{family}/{name}"].append(required[mask])
        residuals["all"].append(required[reachable])
        if (index + 1) % args.progress_every == 0:
            print(f"AO-D0: {index+1}/{stop} | reachable={sum(image_counts)}", flush=True)

    def report_family(family: str) -> dict:
        rows = {}
        for name, count in counts[family].items():
            metrics = metrics_from_histogram(fixed_hist + deltas[family][name])
            base = metrics_from_histogram(fixed_hist)
            rows[name] = {
                "pixels": count, "fraction_of_reachable": count / max(sum(image_counts), 1),
                "oracle_gain_vs_fixed_s2_miou_pp": 100 * (metrics["valid_miou"] - base["valid_miou"]),
                "damaged_f1_change_pp": 100 * (metrics["damaged_f1"] - base["damaged_f1"]),
                "required_residual": _quantiles(residuals[f"{family}/{name}"]),
            }
        return rows

    building_concentration = _concentration(building_counts)
    image_concentration = _concentration(image_counts)
    event_concentration = _concentration(list(event_counts.values()))
    total = sum(image_counts)
    background_fraction = counts["source"]["background"] / max(total, 1)
    checks = {
        "reachable_pixels_match_oracle": args.max_images > 0 or total == int(oracle["reachable_pixels"]),
        "largest_instance_share_within_limit": building_concentration["largest_share"] <= args.maximum_largest_instance_share,
        "top1pct_instance_share_within_limit": building_concentration["top_1_percent_share"] <= args.maximum_top1pct_instance_share,
        "background_source_fraction_within_limit": background_fraction <= args.maximum_background_source_fraction,
        "both_target_classes_present": counts["target"]["damaged"] > 0 and counts["target"]["destroyed"] > 0,
    }
    decision = (
        "ADMIT_AO_E1_EVIDENCE_AUDIT"
        if args.max_images > 0 or all(checks.values())
        else "REVIEW_AO_D0_CONCENTRATION_OR_LOCALIZATION_RISK"
    )
    payload = {
        "protocol": DETAIL_PROTOCOL, "configuration": vars(args),
        "reachable_pixels": total,
        "breakdown": {name: report_family(name) for name in ("source", "target", "direction", "position", "building_size")},
        "required_residual_all": _quantiles(residuals["all"]),
        "concentration": {"building": building_concentration, "image": image_concentration, "event": event_concentration},
        "event_reachable_pixels": dict(event_counts), "checks": checks, "decision": decision,
        "provenance": {**provenance, "checkpoint_sha256": file_sha256(args.checkpoint), "list_sha256": file_sha256(args.list_path)},
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"AO-D0 reachable: {total}")
    print(f"AO-D0 source background/intact: {counts['source']['background']} / {counts['source']['intact']}")
    print(f"AO-D0 target damaged/destroyed: {counts['target']['damaged']} / {counts['target']['destroyed']}")
    print(f"AO-D0 residual quantiles: {payload['required_residual_all']}")
    print(f"AO-D0 concentration: {payload['concentration']}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
