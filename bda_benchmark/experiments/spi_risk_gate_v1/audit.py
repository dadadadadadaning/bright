from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_core_ring_adapter_v1.audit import gains, stratified_paired_bootstrap
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig, selective_context_refinement
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import fast_histogram, metrics_from_histogram, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_region_feedback.core import predicted_building_regions
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import ChangeCounts, RiskGate, apply_gate, boundary_depth_weight, fixed_uncertainty_gate, risk_features
from .runtime import frozen_candidate_forward, load_frozen_models, upsample_logits


STAGE_A = ("d0_frozen_p3", "d1_ungated_m4", "d2_fixed_uncertainty", "d3_fixed_boundary_safe", "d5_selective_context_s2")
ORACLE = "o1_selector_oracle"


def update_change(counts, base, candidate, target, valid_mask=None):
    valid = target != 255
    if valid_mask is not None:
        valid &= valid_mask
    changed = (candidate != base) & valid
    counts.changed += int(changed.sum())
    counts.corrected += int((changed & (base != target) & (candidate == target)).sum())
    counts.harmed += int((changed & (base == target) & (candidate != target)).sum())


def binary_ranking_metrics(scores, labels):
    if not scores:
        return {"auroc": None, "average_precision": None, "count": 0}
    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(np.bool_)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return {"auroc": None, "average_precision": None, "count": int(labels.size), "positives": positives, "negatives": negatives}
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    tp = np.cumsum(ranked)
    fp = np.cumsum(~ranked)
    tpr = np.concatenate([[0.0], tp / positives])
    fpr = np.concatenate([[0.0], fp / negatives])
    auroc = float(np.trapezoid(tpr, fpr))
    precision = tp / np.arange(1, ranked.size + 1)
    average_precision = float(precision[ranked].sum() / positives)
    return {"auroc": auroc, "average_precision": average_precision, "count": int(labels.size), "positives": positives, "negatives": negatives}


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser("D0/D1/O1/D2/D3/D4/D5 pure semantic audit")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--m4-checkpoint", required=True)
    parser.add_argument("--m4-steps", type=int, default=500)
    parser.add_argument("--gate-checkpoint", default="")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=450)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--expected-p3-miou", type=float, default=0.60502)
    parser.add_argument("--calibration-mode", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    p3, m4, manifest = load_frozen_models(args.base_checkpoint, args.m4_checkpoint, device, args.m4_steps)
    gate_model = None
    gate_manifest = None
    if args.gate_checkpoint:
        saved_gate = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
        if saved_gate.get("p3_sha256") != manifest["p3_sha256"] or saved_gate.get("m4_checkpoint") != manifest["m4_checkpoint"]:
            raise RuntimeError("D4 gate provenance does not match P3/M4")
        gate_model = RiskGate().to(device)
        gate_model.load_state_dict(saved_gate["gate_state"], strict=True)
        gate_model.eval().requires_grad_(False)
        gate_manifest = {"path": args.gate_checkpoint, "step": int(saved_gate["optimizer_step"]), "seed": int(saved_gate["seed"])}

    variants = list(STAGE_A)
    if args.calibration_mode:
        variants.remove("d5_selective_context_s2")
    if gate_model is not None:
        variants.insert(4, "d4_learned_risk_gate")
    metric_variants = variants + [ORACLE]
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.val_list, args.sar_stats_json)
    if args.max_images <= 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    totals = {name: torch.zeros(4, 4, dtype=torch.float64) for name in metric_variants}
    event_totals = defaultdict(lambda: {name: torch.zeros(4, 4, dtype=torch.float64) for name in metric_variants})
    changes = {name: ChangeCounts() for name in variants if name != "d0_frozen_p3"}
    zones = {name: {"core": ChangeCounts(), "boundary": ChangeCounts()} for name in changes}
    image_rows = []
    region_args = dict(threshold=0.5, minimum_area=2, core_radius=1, ring_radius=2)
    selective_config = SelectiveContextConfig()
    gate_sums = defaultdict(float)
    gate_pixels = 0
    selector_scores, selector_labels = [], []
    processed = 0
    started = time.perf_counter()
    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_output, base_output, base_low, candidate_low, feature, regions = frozen_candidate_forward(
            p3, m4, rgb, sar, region_args
        )
        support_low = (regions.labels > 0).unsqueeze(1).float()
        uncertainty_gate = fixed_uncertainty_gate(base_low, support_low)
        depth_low = boundary_depth_weight(support_low, radius=2, boundary_floor=0.20)
        d2_low = apply_gate(base_low, candidate_low, uncertainty_gate)
        d3_low = apply_gate(base_low, candidate_low, uncertainty_gate * depth_low)
        output_size = rgb.shape[-2:]
        logits = {
            "d0_frozen_p3": base_output,
            "d1_ungated_m4": upsample_logits(candidate_low, output_size),
            "d2_fixed_uncertainty": upsample_logits(d2_low, output_size),
            "d3_fixed_boundary_safe": upsample_logits(d3_low, output_size),
        }
        learned_gate = None
        if gate_model is not None:
            inputs, support, _ = risk_features(base_low, candidate_low, feature, regions)
            learned_gate = gate_model(inputs, support)
            logits["d4_learned_risk_gate"] = upsample_logits(
                apply_gate(base_low, candidate_low, learned_gate), output_size
            )
            gate_sums["all"] += float(learned_gate.sum())
            gate_sums["active"] += float((learned_gate >= 0.5).sum())
            gate_pixels += learned_gate.numel()

        base_prob_cpu = base_output[0].float().softmax(0).cpu()
        base_prediction_cpu = base_prob_cpu.argmax(0)
        predictions = {name: value[0].float().argmax(0).cpu() for name, value in logits.items()}
        if not args.calibration_mode:
            localization_cpu = loc_output[0].float().argmax(0).cpu()
            full_regions = predicted_building_regions(localization_cpu, minimum_area=32)
            d5_prediction, _ = selective_context_refinement(
                base_prediction_cpu, base_prob_cpu, full_regions,
                config=selective_config, mode="soft_region",
            )
            predictions["d5_selective_context_s2"] = d5_prediction

        name = str(names[0])
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        oracle = base_prediction_cpu.clone()
        use_m4 = (predictions["d1_ungated_m4"] == target) & (base_prediction_cpu != target) & (target != 255)
        oracle[use_m4] = predictions["d1_ungated_m4"][use_m4]
        predictions[ORACLE] = oracle
        if learned_gate is not None:
            target_low = F.interpolate(target[None, None].float().to(device), size=base_low.shape[-2:], mode="nearest")[0, 0].long()
            base_prediction_low = base_low.argmax(1)[0]
            candidate_prediction_low = candidate_low.argmax(1)[0]
            positive = (candidate_prediction_low == target_low) & (base_prediction_low != target_low) & (target_low != 255)
            negative = (base_prediction_low == target_low) & (candidate_prediction_low != target_low) & (target_low != 255)
            decisive = positive | negative
            selector_scores.append(learned_gate[0, 0][decisive].float().cpu().numpy())
            selector_labels.append(positive[decisive].cpu().numpy())
            base_probability_low = F.softmax(base_low.float(), dim=1)[0]
            base_confidence_low = base_probability_low.max(0).values
            safe = (base_prediction_low == target_low) & (base_confidence_low >= 0.8) & (target_low != 255)
            gate_sums["positive_sum"] += float(learned_gate[0, 0][positive].sum())
            gate_sums["positive_count"] += int(positive.sum())
            gate_sums["negative_sum"] += float(learned_gate[0, 0][negative].sum())
            gate_sums["negative_count"] += int(negative.sum())
            gate_sums["safe_sum"] += float(learned_gate[0, 0][safe].sum())
            gate_sums["safe_count"] += int(safe.sum())
        support_full = F.interpolate(support_low, size=output_size, mode="nearest")[0, 0].cpu().bool()
        depth_full = F.interpolate(depth_low, size=output_size, mode="nearest")[0, 0].cpu()
        core_mask = support_full & (depth_full >= 0.999)
        boundary_mask = support_full & ~core_mask
        histograms = {}
        for variant in metric_variants:
            histogram = fast_histogram(predictions[variant], target)
            totals[variant] += histogram
            event_totals[event][variant] += histogram
            histograms[variant] = histogram
            if variant in changes:
                update_change(changes[variant], base_prediction_cpu, predictions[variant], target)
                update_change(zones[variant]["core"], base_prediction_cpu, predictions[variant], target, core_mask)
                update_change(zones[variant]["boundary"], base_prediction_cpu, predictions[variant], target, boundary_mask)
        image_rows.append((event, histograms))
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"SPI risk audit: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)

    reports = {name: metrics_from_histogram(hist) for name, hist in totals.items()}
    for name in metric_variants:
        reports[name]["confusion"] = totals[name].long().tolist()
    event_reports = {event: {name: metrics_from_histogram(hist) for name, hist in table.items()} for event, table in sorted(event_totals.items())}
    comparisons = {}
    pairs = [("d1_minus_d0", "d1_ungated_m4", "d0_frozen_p3"),
             ("d2_minus_d1", "d2_fixed_uncertainty", "d1_ungated_m4"),
             ("d3_minus_d2", "d3_fixed_boundary_safe", "d2_fixed_uncertainty"),
             ("d3_minus_d0", "d3_fixed_boundary_safe", "d0_frozen_p3"),
             ("oracle_minus_d0", ORACLE, "d0_frozen_p3")]
    if not args.calibration_mode:
        pairs.append(("d5_minus_d0", "d5_selective_context_s2", "d0_frozen_p3"))
    if gate_model is not None:
        pairs += [("d4_minus_d3", "d4_learned_risk_gate", "d3_fixed_boundary_safe"),
                  ("d4_minus_d0", "d4_learned_risk_gate", "d0_frozen_p3")]
        if not args.calibration_mode:
            pairs.append(("d4_minus_d5", "d4_learned_risk_gate", "d5_selective_context_s2"))
    for label, candidate, reference in pairs:
        event_values = [100 * (table[candidate]["valid_miou"] - table[reference]["valid_miou"]) for table in event_reports.values()]
        comparisons[label] = {
            "gain_pp": gains(reports[candidate], reports[reference]),
            "event_nonnegative": sum(value >= 0 for value in event_values),
            "event_total": len(event_values),
            "bootstrap": stratified_paired_bootstrap(image_rows, candidate, reference, args.bootstrap_iterations, args.seed),
        }
    reproduction_error = 100 * (reports["d0_frozen_p3"]["valid_miou"] - args.expected_p3_miou)
    d1_harmed = changes["d1_ungated_m4"].harmed
    best_fixed = min(("d2_fixed_uncertainty", "d3_fixed_boundary_safe"), key=lambda key: changes[key].harmed)
    harm_reduction = (d1_harmed - changes[best_fixed].harmed) / max(d1_harmed, 1)
    stage_a_checks = {
        "p3_reproduced": abs(reproduction_error) <= 0.05,
        "oracle_gain_ge_0_5pp": comparisons["oracle_minus_d0"]["gain_pp"]["valid_miou"] >= 0.5,
        "fixed_gate_recovers_near_p3": max(reports["d2_fixed_uncertainty"]["valid_miou"], reports["d3_fixed_boundary_safe"]["valid_miou"]) >= reports["d0_frozen_p3"]["valid_miou"] - 0.001,
        "fixed_gate_harm_reduction_ge_20pct": harm_reduction >= 0.20,
    }
    if gate_model is not None:
        d4_harm_reduction = (d1_harmed - changes["d4_learned_risk_gate"].harmed) / max(d1_harmed, 1)
        high_confidence_gate = gate_sums["safe_sum"] / max(gate_sums["safe_count"], 1)
    if gate_model is None:
        decision = "PROCEED_TO_D4_GATE_TRAINING" if all(stage_a_checks.values()) else "STOP_BEFORE_D4_GATE_TRAINING"
        final_checks = None
    elif args.calibration_mode:
        final_checks = {
            "d4_miou_above_p3": reports["d4_learned_risk_gate"]["valid_miou"] > reports["d0_frozen_p3"]["valid_miou"],
            "d4_harms_fewer_than_d1": changes["d4_learned_risk_gate"].harmed < changes["d1_ungated_m4"].harmed,
        }
        decision = "PROCEED_TO_VAL450_D4_AUDIT" if all(final_checks.values()) else "STOP_D4_ON_INTERNAL_CALIBRATION"
    else:
        d4_gain = comparisons["d4_minus_d0"]["gain_pp"]
        final_checks = {
            "miou_gain_ge_0_5pp": d4_gain["valid_miou"] >= 0.5,
            "damaged_f1_gain_ge_1pp": d4_gain["damaged_f1"] >= 1.0,
            "no_foreground_drop_gt_0_5pp": min(d4_gain[key] for key in ("intact_iou", "damaged_iou", "destroyed_iou")) >= -0.5,
            "at_least_10_events_nonnegative": comparisons["d4_minus_d0"]["event_nonnegative"] >= 10,
            "bootstrap_low_gt_zero": comparisons["d4_minus_d0"]["bootstrap"]["ci95_low_pp"] > 0,
            "d4_beats_d5_by_0_2pp": comparisons["d4_minus_d5"]["gain_pp"]["valid_miou"] >= 0.2,
            "harm_reduction_vs_d1_ge_30pct": d4_harm_reduction >= 0.30,
            "high_confidence_p3_gate_lt_0_15": high_confidence_gate < 0.15,
        }
        decision = "PROCEED_TO_THREE_SEEDS" if all(final_checks.values()) else "STOP_D4_SINGLE_SEED_INSUFFICIENT"
    payload = {
        "protocol": "SPI_RISK_GATE_PURE_SEMANTIC_VAL450_v1",
        "processed_images": processed, "semantic_ground_truth": "dataset/target/*_building_damage.tif",
        "official_instances_accessed": False, "test_accessed": False,
        "manifest": manifest, "gate_manifest": gate_manifest,
        "p3_reproduction": {"expected": args.expected_p3_miou, "observed": reports["d0_frozen_p3"]["valid_miou"], "error_pp": reproduction_error, "passed": stage_a_checks["p3_reproduced"]},
        "reports": reports, "event_reports": event_reports, "comparisons": comparisons,
        "change_audit": {name: counts.as_dict() for name, counts in changes.items()},
        "zone_change_audit": {name: {zone: counts.as_dict() for zone, counts in table.items()} for name, table in zones.items()},
        "gate_statistics": None if gate_model is None else {
            "mean": gate_sums["all"] / max(gate_pixels, 1),
            "fraction_ge_0_5": gate_sums["active"] / max(gate_pixels, 1),
            "mean_on_m4_only_correct": gate_sums["positive_sum"] / max(gate_sums["positive_count"], 1),
            "mean_on_p3_only_correct": gate_sums["negative_sum"] / max(gate_sums["negative_count"], 1),
            "mean_on_high_confidence_p3_correct": gate_sums["safe_sum"] / max(gate_sums["safe_count"], 1),
            "selector_ranking": binary_ranking_metrics(selector_scores, selector_labels),
        },
        "stage_a": {"checks": stage_a_checks, "best_fixed_by_harm": best_fixed, "harm_reduction": harm_reduction},
        "final": {"checks": final_checks, "decision": decision},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 118)
    for variant in metric_variants:
        report = reports[variant]
        print(f"{variant:30s} mIoU={100*report['valid_miou']:.4f} D-F1={100*report['damaged_f1']:.4f}")
    print(f"Stage-A checks: {stage_a_checks}\nFinal checks: {final_checks}\nDecision: {decision}\nSaved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
