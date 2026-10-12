from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels, predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram, metrics_from_histogram, p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_region_feedback.core import stratified_paired_bootstrap
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .audit_suitability import load_cards, semantic_memory_posterior, _validate


VARIANTS = (
    "m0_p3", "m1_fixed_s2", "m2_p3_memory_severity",
    "m3_p3_memory_full", "m4_fixed_s2_memory_severity",
    "m5_fixed_s2_memory_full",
)


def severity_only_prediction(
    probabilities: torch.Tensor,
    region_labels: torch.Tensor,
    posterior: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Relabel foreground severity while preserving the source foreground mask exactly."""
    source = probabilities.argmax(0).cpu()
    output = source.clone()
    for region_id in range(1, int(region_labels.max()) + 1):
        region = region_labels == region_id
        editable = region & (source > 0)
        if not bool(editable.any()):
            continue
        base = probabilities[1:4, editable].float()
        base = base / base.sum(0, keepdim=True).clamp_min(1e-8)
        memory = posterior[region_id - 1, 1:4].float()
        memory = memory / memory.sum().clamp_min(1e-8)
        mixed = (1.0 - beta) * base + beta * memory[:, None]
        output[editable] = mixed.argmax(0).cpu() + 1
    if not torch.equal(output > 0, source > 0):
        raise AssertionError("severity-only writeback changed localization")
    return output


def full_semantic_prediction(
    probabilities: torch.Tensor,
    region_labels: torch.Tensor,
    posterior: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    output = probabilities.argmax(0).cpu()
    for region_id in range(1, int(region_labels.max()) + 1):
        region = region_labels == region_id
        mixed = (1.0 - beta) * probabilities[:, region].float()
        mixed += beta * posterior[region_id - 1, :, None].float()
        output[region] = mixed.argmax(0).cpu()
    return output


def change_counts(reference: torch.Tensor, candidate: torch.Tensor, target: torch.Tensor) -> dict:
    valid = target != 255; changed = valid & (reference != candidate)
    corrected = changed & (candidate == target) & (reference != target)
    harmed = changed & (candidate != target) & (reference == target)
    return {
        "changed": int(changed.sum()), "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()), "net": int(corrected.sum() - harmed.sum()),
    }


def _event_reports(rows: list[dict]) -> dict:
    output = {}
    for event in sorted(set(str(row["event"]) for row in rows)):
        selected = [row for row in rows if row["event"] == event]
        metrics = {
            name: metrics_from_histogram(sum(
                (row["histograms"][name] for row in selected),
                torch.zeros(4, 4, dtype=torch.float64),
            ))
            for name in VARIANTS
        }
        output[event] = {
            "metrics": metrics,
            "gains_vs_p3_miou_pp": {name: 100 * (value["valid_miou"] - metrics["m0_p3"]["valid_miou"]) for name, value in metrics.items()},
            "m4_minus_m1_miou_pp": 100 * (metrics["m4_fixed_s2_memory_severity"]["valid_miou"] - metrics["m1_fixed_s2"]["valid_miou"]),
        }
    return output


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Pixel-level complete-building Memory writeback E1")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True); parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--train-cards", required=True); parser.add_argument("--calibration-cards", required=True)
    parser.add_argument("--suitability-json", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--beta", type=float, default=.25); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--maximum-per-event-class", type=int, default=256); parser.add_argument("--maximum-per-event", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.10); parser.add_argument("--semantic-blend", type=float, default=.25)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000); parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    if args.beta != .25 or args.topk != 8 or args.maximum_per_event != 2:
        raise RuntimeError("E1 protocol freezes beta=0.25, topk=8 and maximum_per_event=2")
    suitability = json.load(open(args.suitability_json, encoding="utf-8"))
    if suitability.get("decision") != "MEMORY_SUITABLE_FOR_P3_SEMANTIC_CLASSIFICATION_ONLY":
        raise RuntimeError("semantic-only Memory E0 prerequisite did not pass")
    train = load_cards(args.train_cards); query = load_cards(args.calibration_cards)
    _validate(train, "meta_train"); _validate(query, "meta_calibration")
    if int(train.get("minimum_area", -1)) != args.minimum_area or int(query.get("minimum_area", -1)) != args.minimum_area:
        raise RuntimeError("writeback minimum_area differs from complete-card construction")
    if train.get("p3_sha256") != query.get("p3_sha256"):
        raise RuntimeError("card P3 fingerprints differ")
    posterior, diagnostics = semantic_memory_posterior(train, query, args)
    card_lookup = {}
    for row, (image_index, region_id) in enumerate(zip(query["image_index"], query["region_label"])):
        name = query["image_names"][int(image_index)]
        card_lookup[(name, int(region_id))] = row

    device = torch.device(args.device); model, provenance = load_frozen_p3(args.checkpoint, device)
    if provenance["p3_sha256"] != query.get("p3_sha256"):
        raise RuntimeError("writeback checkpoint differs from card checkpoint")
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images}, observed {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    totals = {name: torch.zeros(4, 4, dtype=torch.float64) for name in VARIANTS}
    rows, changes = [], {name: {"changed": 0, "corrected": 0, "harmed": 0, "net": 0} for name in VARIANTS[2:]}
    localization_differences = {"m2_vs_m0": 0, "m4_vs_m1": 0}
    for image_index, (rgb_cpu, sar_cpu, names) in enumerate(loader):
        name = str(names[0]); rgb = rgb_cpu.to(device); sar = sar_cpu.to(device)
        localization, damage = model(rgb, p3_sar_view(sar))
        p3_probability = damage[0].float().softmax(0).cpu(); p3_prediction = p3_probability.argmax(0)
        regions = predicted_building_label_map(localization[0].argmax(0).cpu(), args.minimum_area)
        memory_rows = []
        for region_id in range(1, regions.count + 1):
            key = (name, region_id)
            if key not in card_lookup:
                raise RuntimeError(f"complete-card replay mismatch: {key}")
            memory_rows.append(card_lookup[key])
        image_posterior = posterior[memory_rows] if memory_rows else posterior.new_zeros((0, 4))
        fixed_probability = fixed_s2_candidate_from_labels(
            p3_probability, regions, SelectiveContextConfig()
        ).probabilities.cpu()
        fixed_prediction = fixed_probability.argmax(0)
        predictions = {
            "m0_p3": p3_prediction, "m1_fixed_s2": fixed_prediction,
            "m2_p3_memory_severity": severity_only_prediction(p3_probability, regions.labels, image_posterior, args.beta),
            "m3_p3_memory_full": full_semantic_prediction(p3_probability, regions.labels, image_posterior, args.beta),
            "m4_fixed_s2_memory_severity": severity_only_prediction(fixed_probability, regions.labels, image_posterior, args.beta),
            "m5_fixed_s2_memory_full": full_semantic_prediction(fixed_probability, regions.labels, image_posterior, args.beta),
        }
        target = load_semantic_target(args.dataset_path, name).long().cpu()
        histograms = {key: fast_histogram(value, target) for key, value in predictions.items()}
        for key, value in histograms.items(): totals[key] += value
        for key in changes:
            reference = p3_prediction if key.startswith(("m2", "m3")) else fixed_prediction
            item = change_counts(reference, predictions[key], target)
            for field in changes[key]: changes[key][field] += item[field]
        localization_differences["m2_vs_m0"] += int(((predictions["m2_p3_memory_severity"] > 0) != (p3_prediction > 0)).sum())
        localization_differences["m4_vs_m1"] += int(((predictions["m4_fixed_s2_memory_severity"] > 0) != (fixed_prediction > 0)).sum())
        event = name.rsplit("_", 1)[0]; rows.append({"event": event, "histograms": histograms})
        if (image_index + 1) % args.progress_every == 0:
            print(f"Memory writeback E1: {image_index+1}/{len(dataset)}", flush=True)
    reports = {name: metrics_from_histogram(hist) for name, hist in totals.items()}
    events = _event_reports(rows)
    bootstrap = stratified_paired_bootstrap(
        rows, variants=VARIANTS, baseline="m0_p3", iterations=args.bootstrap_iterations,
        seed=args.seed, comparison_pairs=(("m4_fixed_s2_memory_severity", "m1_fixed_s2"),),
    )
    primary_gain = 100 * (reports["m4_fixed_s2_memory_severity"]["valid_miou"] - reports["m1_fixed_s2"]["valid_miou"])
    damaged_change = 100 * (reports["m4_fixed_s2_memory_severity"]["damaged_f1"] - reports["m1_fixed_s2"]["damaged_f1"])
    destroyed_change = 100 * (reports["m4_fixed_s2_memory_severity"]["destroyed_f1"] - reports["m1_fixed_s2"]["destroyed_f1"])
    nonnegative = sum(row["m4_minus_m1_miou_pp"] >= 0 for row in events.values())
    ci = bootstrap["comparisons"]["m4_fixed_s2_memory_severity_minus_m1_fixed_s2"]
    memory_alone_gain = 100 * (reports["m2_p3_memory_severity"]["valid_miou"] - reports["m0_p3"]["valid_miou"])
    checks = {
        "m4_gain_vs_fixed_s2_ge_0p30pp": primary_gain >= .30,
        "m2_gain_vs_p3_positive": memory_alone_gain > 0,
        "damaged_f1_not_below_fixed_s2": damaged_change >= 0,
        "destroyed_f1_drop_not_gt_0p10pp": destroyed_change >= -.10,
        "nonnegative_events_ge_10": nonnegative >= 10,
        "bootstrap_ci_low_gt_zero": ci["ci95_low"] > 0,
        "severity_localization_exact": all(value == 0 for value in localization_differences.values()),
    }
    decision = "FREEZE_SEMANTIC_MEMORY_FOR_FINAL_VALIDATION" if all(checks.values()) else "STOP_SEMANTIC_MEMORY_AFTER_PIXEL_WRITEBACK_E1"
    report = {
        "protocol": "SPI_COMPLETE_BUILDING_MEMORY_PIXEL_WRITEBACK_E1_v1",
        "configuration": vars(args), "reports": reports,
        "gains": {"m2_minus_m0_miou_pp": memory_alone_gain, "m4_minus_m1_miou_pp": primary_gain,
                  "m4_minus_m1_damaged_f1_pp": damaged_change, "m4_minus_m1_destroyed_f1_pp": destroyed_change},
        "events": events, "primary_nonnegative_events": f"{nonnegative}/{len(events)}",
        "changes": changes, "localization_differences": localization_differences,
        "bootstrap": bootstrap, "retrieval_coverage": float(diagnostics["coverage"].float().mean()),
        "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, value in reports.items():
        print(f"{name:34s} mIoU={100*value['valid_miou']:.4f} D-F1={100*value['damaged_f1']:.4f} X-F1={100*value['destroyed_f1']:.4f}")
    print(f"M4-M1={primary_gain:+.4f}pp D-F1={damaged_change:+.4f}pp events={nonnegative}/{len(events)} CI={ci}")
    print(f"Decision: {decision}"); print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
