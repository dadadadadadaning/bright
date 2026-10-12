from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_metric_aligned_cluster_utility_v1.audit_oracle import bootstrap
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels, predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram, file_sha256, p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from . import ORACLE_PROTOCOL
from .core import apply_affected_oracle


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("AO-E0 affected-occurrence reachable Oracle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--minimum-building-area", type=int, default=32)
    parser.add_argument("--minimum-gain-pp", type=float, default=.30)
    parser.add_argument("--minimum-damaged-f1-gain-pp", type=float, default=.50)
    parser.add_argument("--minimum-nonnegative-events", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("refusing repeated AO-E0 output")
    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, got {len(dataset)}")
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    config = SelectiveContextConfig()
    histograms = {name: [] for name in ("p3", "fixed_s2", "affected_oracle_after_s2")}
    image_events, image_names = [], []
    reachable_pixels = 0
    for index, (rgb_cpu, sar_cpu, names) in enumerate(loader):
        if index >= stop:
            break
        rgb = rgb_cpu.to(device, non_blocking=True)
        sar = sar_cpu.to(device, non_blocking=True)
        localization, damage = p3(rgb, p3_sar_view(sar))
        probabilities = damage[0].float().softmax(0).cpu()
        p3_labels = probabilities.argmax(0)
        regions = predicted_building_label_map(
            localization[0].argmax(0).cpu(), args.minimum_building_area
        )
        fixed = fixed_s2_candidate_from_labels(probabilities, regions, config)
        fixed_probabilities = fixed.probabilities.float().cpu()
        target = load_semantic_target(args.dataset_path, str(names[0])).long()
        oracle_labels, reachable = apply_affected_oracle(
            fixed_probabilities, regions.labels > 0, target
        )
        histograms["p3"].append(fast_histogram(p3_labels, target).long())
        histograms["fixed_s2"].append(fast_histogram(fixed_probabilities.argmax(0), target).long())
        histograms["affected_oracle_after_s2"].append(fast_histogram(oracle_labels, target).long())
        reachable_pixels += int(reachable.sum())
        name = str(names[0])
        image_names.append(name); image_events.append(name.rsplit("_", 1)[0])
        if (index + 1) % args.progress_every == 0:
            print(f"AO-E0 {args.split}: {index+1}/{stop} | reachable={reachable_pixels}", flush=True)
    histograms = {name: torch.stack(rows) for name, rows in histograms.items()}
    reports = {name: metrics_from_histogram(rows.sum(0)) for name, rows in histograms.items()}
    fixed = reports["fixed_s2"]; oracle = reports["affected_oracle_after_s2"]
    gain = 100 * (oracle["valid_miou"] - fixed["valid_miou"])
    damaged_gain = 100 * (oracle["damaged_f1"] - fixed["damaged_f1"])
    event_rows = {}
    for event in sorted(set(image_events)):
        selected = torch.tensor([value == event for value in image_events])
        base = metrics_from_histogram(histograms["fixed_s2"][selected].sum(0))
        changed = metrics_from_histogram(histograms["affected_oracle_after_s2"][selected].sum(0))
        event_rows[event] = {
            "gain_vs_fixed_s2_miou_pp": 100 * (changed["valid_miou"] - base["valid_miou"]),
            "damaged_f1_change_pp": 100 * (changed["damaged_f1"] - base["damaged_f1"]),
        }
    nonnegative = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in event_rows.values())
    temporary = {"fixed_s2_histograms": histograms["fixed_s2"], "image_events": image_events}
    bootstrap_row = bootstrap(temporary, histograms["affected_oracle_after_s2"],
                              args.bootstrap_iterations, args.seed)
    checks = {
        "gain_vs_fixed_s2_ge_threshold": gain >= args.minimum_gain_pp,
        "damaged_f1_gain_ge_threshold": damaged_gain >= args.minimum_damaged_f1_gain_pp,
        "nonnegative_events_ge_threshold": nonnegative >= min(args.minimum_nonnegative_events, len(event_rows)),
        "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
        "reachable_pixels_positive": reachable_pixels > 0,
    }
    decision = "ADMIT_AO_E1_EVIDENCE_AUDIT" if all(checks.values()) else "STOP_AO_AFTER_REACHABLE_ORACLE"
    payload = {
        "protocol": ORACLE_PROTOCOL, "configuration": vars(args), "reports": reports,
        "gain_vs_fixed_s2_miou_pp": gain, "damaged_f1_gain_vs_fixed_s2_pp": damaged_gain,
        "reachable_pixels": reachable_pixels, "events": event_rows,
        "event_nonnegative": nonnegative, "bootstrap": bootstrap_row,
        "checks": checks, "decision": decision,
        "provenance": {**provenance, "checkpoint_sha256": file_sha256(args.checkpoint),
                       "list_sha256": file_sha256(args.list_path)},
        "image_names": image_names, "meta_confirm_accessed": False,
        "val450_accessed": False, "official_instances_accessed": False,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"P3 mIoU: {reports['p3']['valid_miou']*100:.4f}")
    print(f"fixed S2 mIoU: {fixed['valid_miou']*100:.4f}")
    print(f"AO reachable Oracle mIoU: {oracle['valid_miou']*100:.4f}")
    print(f"AO Oracle gain: {gain:+.4f}pp D-F1={damaged_gain:+.4f}pp events={nonnegative}/{len(event_rows)}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

