from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import fixed_s2_candidate_from_labels, predicted_building_label_map
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import affected_component_area, alignment_displacement_proxy, bucket_counts, core_and_ring, modification_cluster_area, rate, robust_sar_noise


BUCKETS = ("core", "ring", "small_damage", "large_damage", "high_alignment_proxy", "low_alignment_proxy", "high_sar_noise", "low_sar_noise", "s2_modified_cluster", "s2_unmodified")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def threshold_path(root: Path, seed: int) -> Path:
    return root / f"p3_gate_miss_spatial_thresholds_seed{seed}.json"


def _sample(values: torch.Tensor, maximum: int = 512) -> torch.Tensor:
    values = values.flatten().cpu()
    if len(values) <= maximum:
        return values
    # Deterministic evenly spaced sampling is sufficient for freezing broad
    # quantile cutoffs and avoids retaining full-resolution maps.
    index = torch.linspace(0, len(values) - 1, maximum).round().long()
    return values[index]


def fit_thresholds(rows, *, core_radius: int, quantile: float) -> tuple[dict, int]:
    values = {"damage_area": [], "alignment": [], "sar_noise": []}
    images = 0
    for row in rows:
        images += 1
        affected = row["affected"]
        if bool(affected.any()):
            values["damage_area"].append(_sample(row["damage_area"][affected].float()))
        values["alignment"].append(_sample(row["alignment"][row["building"]].float()))
        values["sar_noise"].append(_sample(row["sar_noise"][row["building"]].float()))
    def q(name: str, level: float) -> float:
        packed = torch.cat(values[name])
        return float(torch.quantile(packed, level))
    output = {"protocol": PROTOCOL, "source_split": "meta_train", "core_radius": core_radius, "quantile": quantile,
            "small_damage_area_q25": q("damage_area", .25), "high_alignment_q": q("alignment", quantile),
            "high_sar_noise_q": q("sar_noise", quantile), "entry_filter_uses_gt": False,
            "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    return output, images


def classify(row: dict, thresholds: dict) -> dict[str, tuple[int, int]]:
    affected, missed = row["affected"], row["missed"]
    building = row["building"]
    masks = {
        "core": row["core"], "ring": row["ring"],
        "small_damage": row["damage_area"] <= thresholds["small_damage_area_q25"],
        "large_damage": row["damage_area"] > thresholds["small_damage_area_q25"],
        "high_alignment_proxy": building & (row["alignment"] >= thresholds["high_alignment_q"]),
        "low_alignment_proxy": building & (row["alignment"] < thresholds["high_alignment_q"]),
        "high_sar_noise": building & (row["sar_noise"] >= thresholds["high_sar_noise_q"]),
        "low_sar_noise": building & (row["sar_noise"] < thresholds["high_sar_noise_q"]),
        "s2_modified_cluster": row["s2_changed"],
        "s2_unmodified": ~row["s2_changed"],
    }
    return {name: bucket_counts(affected, missed, mask) for name, mask in masks.items()}


def aggregate(rows, thresholds: dict) -> tuple[dict, dict, int]:
    total = {name: [0, 0] for name in BUCKETS}; by_event = defaultdict(lambda: {name: [0, 0] for name in BUCKETS})
    base_den = base_num = images = 0
    s2_miss_changed = s2_miss_recovered = s2_miss_total = s2_cluster_count = s2_cluster_pixels = 0
    overlap_names = ("ring", "small_damage", "high_sar_noise", "s2_modified_cluster")
    overlap_patterns = torch.zeros(16, dtype=torch.long)
    for row in rows:
        images += 1
        for name, (denominator, numerator) in classify(row, thresholds).items():
            total[name][0] += denominator; total[name][1] += numerator
            by_event[row["event"]][name][0] += denominator; by_event[row["event"]][name][1] += numerator
        base_den += int(row["affected"].sum())
        base_num += int((row["affected"] & row["missed"]).sum())
        s2_miss_total += int(row["missed"].sum())
        s2_miss_changed += int((row["missed"] & row["s2_changed"]).sum())
        s2_miss_recovered += int((row["missed"] & (row["s2_prediction"] >= 2)).sum())
        s2_cluster_count += int(row["s2_cluster_count"])
        s2_cluster_pixels += int(row["s2_changed"].sum())
        conditions = (
            row["ring"],
            row["damage_area"] <= thresholds["small_damage_area_q25"],
            row["building"] & (row["sar_noise"] >= thresholds["high_sar_noise_q"]),
            row["s2_changed"],
        )
        code = sum(mask.to(torch.long) << bit for bit, mask in enumerate(conditions))
        overlap_patterns += torch.bincount(code[row["missed"]], minlength=16)
    base_rate = rate(base_num, base_den)
    def report(values: dict) -> dict:
        output = {}
        for name, (denominator, numerator) in values.items():
            item_rate = rate(numerator, denominator)
            output[name] = {"affected_pixels": denominator, "missed_pixels": numerator, "miss_rate": item_rate,
                            "risk_ratio_vs_all_affected": None if item_rate is None or not base_rate else item_rate / base_rate}
        return output
    patterns = []
    for code, pixels in enumerate(overlap_patterns.tolist()):
        if not pixels:
            continue
        enabled = [name for bit, name in enumerate(overlap_names) if code & (1 << bit)]
        patterns.append({"conditions": enabled or ["none"], "missed_pixels": pixels,
                         "fraction_of_all_misses": pixels / max(base_num, 1)})
    return {"all_affected_pixels": base_den, "all_missed_pixels": base_num, "overall_miss_rate": base_rate, "buckets": report(total),
            "s2_modification_clusters": {"clusters": s2_cluster_count, "pixels": s2_cluster_pixels,
                "p3_missed_affected_pixels": s2_miss_total, "missed_pixels_inside_s2_modified_clusters": s2_miss_changed,
                "s2_recovered_missed_pixels": s2_miss_recovered,
                "miss_coverage_by_s2_clusters": rate(s2_miss_changed, s2_miss_total),
                "miss_recovery_by_fixed_s2": rate(s2_miss_recovered, s2_miss_total)},
            "miss_overlap_patterns": {"axes": list(overlap_names), "patterns": sorted(patterns, key=lambda row: row["missed_pixels"], reverse=True)}}, {event: report(values) for event, values in sorted(by_event.items())}, images


def collect(args, split: str, maximum: int):
    device = torch.device(args.device); model, _ = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images}, observed {len(dataset)}")
    stop = min(len(dataset), maximum) if maximum else len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    affected_count = 0
    with torch.inference_mode():
        for index, (rgb_cpu, sar_cpu, names) in enumerate(loader):
            if index >= stop: break
            rgb, sar = rgb_cpu.to(device, non_blocking=True), sar_cpu.to(device, non_blocking=True)
            _loc, logits = model(rgb, p3_sar_view(sar))
            probabilities = logits[0].float().softmax(0).cpu()
            prediction = probabilities.argmax(0)
            target = load_semantic_target(args.dataset_path, str(names[0])).long()
            valid = (target >= 0) & (target <= 3); building = valid & (target >= 1)
            affected = valid & (target >= 2); missed = affected & (prediction <= 1)
            core, ring = core_and_ring(building, args.core_radius)
            regions = predicted_building_label_map(_loc[0].argmax(0).cpu(), args.minimum_building_area)
            fixed = fixed_s2_candidate_from_labels(probabilities, regions, SelectiveContextConfig()).probabilities.cpu().argmax(0)
            s2_changed = fixed != prediction
            _s2_area, s2_cluster_count = modification_cluster_area(s2_changed)
            affected_count += int(affected.sum())
            yield {"event": str(names[0]).rsplit("_", 1)[0], "building": building, "affected": affected, "missed": missed,
                   "core": core, "ring": ring, "damage_area": affected_component_area(target),
                   "s2_changed": s2_changed, "s2_prediction": fixed, "s2_cluster_count": s2_cluster_count,
                   "alignment": alignment_displacement_proxy(rgb[0], sar[0, 1], args.alignment_radius),
                   "sar_noise": robust_sar_noise(sar_cpu[0, 1], args.sar_mad_window)}
            if (index + 1) % args.progress_every == 0 or index + 1 == stop:
                print(f"P3 spatial audit {split}: {index + 1}/{stop} | affected={affected_count}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Read-only P3 gate-miss spatial attribution")
    parser.add_argument("--mode", choices=("fit", "evaluate"), required=True); parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True); parser.add_argument("--list-path", required=True); parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--thresholds", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=0); parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument("--minimum-building-area", type=int, default=32)
    parser.add_argument("--alignment-radius", type=int, default=2); parser.add_argument("--sar-mad-window", type=int, default=5)
    parser.add_argument("--high-risk-quantile", type=float, default=.75); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1); parser.add_argument("--progress-every", type=int, default=10); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    if not .5 < args.high_risk_quantile < 1: raise ValueError("high-risk quantile must lie in (.5, 1)")
    thresholds_file = Path(args.thresholds); output = Path(args.output)
    if args.mode == "fit":
        if args.split != "meta_train": raise RuntimeError("threshold fitting is permitted only on meta_train")
        thresholds, images = fit_thresholds(collect(args, args.split, args.max_images), core_radius=args.core_radius, quantile=args.high_risk_quantile)
        thresholds.update({"checkpoint_sha256": file_sha256(args.checkpoint), "list_sha256": file_sha256(args.list_path), "images": images})
        atomic_json(thresholds_file, thresholds); print(json.dumps(thresholds, indent=2), flush=True); return 0
    thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
    if thresholds.get("protocol") != PROTOCOL: raise RuntimeError("threshold file belongs to another protocol")
    if args.split != "meta_calibration" and args.max_images <= 0:
        raise RuntimeError("full evaluation is permitted only on meta_calibration")
    summary, event_rows, images = aggregate(collect(args, args.split, args.max_images), thresholds)
    payload = {"protocol": PROTOCOL, "stage": "P3_GATE_MISS_SPATIAL_ATTRIBUTION", "configuration": vars(args), "frozen_thresholds": thresholds,
               "images": images, "summary": summary, "events": event_rows,
               "interpretation_rule": "A risk ratio materially above 1 identifies a condition enriched for P3 affected misses; this is attribution only, not a model admission.",
               "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    atomic_json(output, payload)
    print(json.dumps({"overall_miss_rate": summary["overall_miss_rate"], "buckets": summary["buckets"]}, indent=2), flush=True)
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__": raise SystemExit(main())
