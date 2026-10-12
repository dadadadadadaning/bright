from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram,
    metrics_from_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    stratified_paired_bootstrap,
)
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import (
    UtilityGate,
    binary_ranking_metrics,
    compute_transition_audit,
    direction_policy_mask,
    empty_transition_audit,
    finalize_transition_audit,
    fixed_s2_candidate_from_labels,
    merge_transition_audit,
    predicted_building_label_map,
    select_prediction,
    selector_targets,
    utility_features,
)
from .runtime import load_frozen_p3


def empty_change_counts() -> dict[str, int]:
    return {"changed": 0, "corrected": 0, "harmed": 0, "neutral": 0}


def update_change_counts(counts, base, updated, target) -> None:
    valid = (target >= 0) & (target <= 3)
    changed = valid & (base != updated)
    corrected = changed & (base != target) & (updated == target)
    harmed = changed & (base == target) & (updated != target)
    counts["changed"] += int(changed.sum())
    counts["corrected"] += int(corrected.sum())
    counts["harmed"] += int(harmed.sum())
    counts["neutral"] += int((changed & ~corrected & ~harmed).sum())


def finalize_change_counts(counts):
    output = dict(counts)
    output["net"] = output["corrected"] - output["harmed"]
    output["harm_rate"] = output["harmed"] / max(output["changed"], 1)
    output["correction_rate"] = output["corrected"] / max(output["changed"], 1)
    output["decisive_precision"] = output["corrected"] / max(output["corrected"] + output["harmed"], 1)
    return output


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Audit fixed S2, direction policy and utility gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-checkpoint", default="")
    parser.add_argument("--direction-policy-json", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--expected-p3-miou", type=float, default=0.0)
    parser.add_argument("--calibration-mode", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, provenance = load_frozen_p3(args.checkpoint, device)
    policy_payload = json.loads(Path(args.direction_policy_json).read_text(encoding="utf-8"))
    if policy_payload.get("p3_sha256") != provenance["p3_sha256"]:
        raise RuntimeError("direction policy and P3 provenance do not match")
    policy = policy_payload["direction_policy"]["policy"]
    gate = None
    gate_manifest = None
    if args.gate_checkpoint:
        saved = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
        if saved.get("p3_sha256") != provenance["p3_sha256"]:
            raise RuntimeError("utility gate and P3 provenance do not match")
        gate = UtilityGate().to(device)
        gate.load_state_dict(saved["gate_state"], strict=True)
        gate.eval().requires_grad_(False)
        gate_manifest = {"path": str(Path(args.gate_checkpoint).resolve()), "optimizer_step": int(saved["optimizer_step"])}

    variants = ["u0_frozen_p3", "u1_fixed_s2", "u2_direction_policy"]
    if gate is not None:
        variants.append("u3_utility_gate")
    metric_variants = variants + ["o1_pixel_oracle"]
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if args.expected_images > 0 and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    totals = {name: torch.zeros(4, 4, dtype=torch.float64) for name in metric_variants}
    event_totals = defaultdict(lambda: {name: torch.zeros(4, 4, dtype=torch.float64) for name in metric_variants})
    change_counts = {name: empty_change_counts() for name in variants[1:]}
    transition_total = empty_transition_audit()
    transition_events = defaultdict(empty_transition_audit)
    gate_scores: list[np.ndarray] = []
    gate_labels: list[np.ndarray] = []
    gate_sums = defaultdict(float)
    image_rows = []
    config = SelectiveContextConfig()
    processed = 0
    accepted_regions = 0
    total_regions = 0
    started = time.perf_counter()
    for rgb, sar, names in loader:
        if args.max_images > 0 and processed >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits = model(rgb, sar)
        base_prob = damage_logits[0].float().softmax(0).cpu()
        base_prediction = base_prob.argmax(0)
        localization = loc_logits[0].argmax(0).cpu()
        region_map = predicted_building_label_map(localization, args.minimum_area)
        candidate = fixed_s2_candidate_from_labels(base_prob, region_map, config)
        candidate_prediction = candidate.probabilities.argmax(0)
        total_regions += len(candidate.records)
        accepted_regions += sum(bool(row["accepted"]) for row in candidate.records)
        direction_accept = direction_policy_mask(base_prediction, candidate_prediction, policy)
        predictions = {
            "u0_frozen_p3": base_prediction,
            "u1_fixed_s2": candidate_prediction,
            "u2_direction_policy": select_prediction(base_prediction, candidate_prediction, direction_accept),
        }
        gate_values_cpu = None
        if gate is not None:
            base_device = base_prob[None].to(device)
            candidate_device = candidate.probabilities[None].to(device)
            support = candidate.support[None, None].to(device)
            region_features = candidate.region_features[None].to(device)
            inputs, changed = utility_features(base_device, candidate_device, support, region_features)
            gate_values = gate(inputs, changed)
            gate_accept = gate_values >= args.gate_threshold
            predictions["u3_utility_gate"] = select_prediction(
                base_device.argmax(1), candidate_device.argmax(1), gate_accept
            )[0].cpu()
            gate_values_cpu = gate_values[0, 0].cpu()
            gate_sums["score_sum"] += float(gate_values.sum())
            gate_sums["changed"] += int(changed.sum())
            gate_sums["accepted"] += int((gate_accept & changed).sum())

        # Labels are loaded only after all deployable predictions are fixed.
        name = str(names[0])
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        oracle_accept = (candidate_prediction == target) & (base_prediction != target) & (target != 255)
        predictions["o1_pixel_oracle"] = select_prediction(base_prediction, candidate_prediction, oracle_accept)
        image_transition = compute_transition_audit(
            base_prob, candidate.probabilities, target, candidate.support
        )
        merge_transition_audit(transition_total, image_transition)
        merge_transition_audit(transition_events[event], image_transition)
        if gate_values_cpu is not None:
            selector_target, decisive, positive, negative = selector_targets(
                base_prediction[None], candidate_prediction[None], target[None]
            )
            decisive_cpu = decisive[0, 0]
            gate_scores.append(gate_values_cpu[decisive_cpu].numpy())
            gate_labels.append(selector_target[0, 0][decisive_cpu].numpy())
            gate_sums["positive_sum"] += float(gate_values_cpu[positive[0]].sum())
            gate_sums["positive_count"] += int(positive.sum())
            gate_sums["negative_sum"] += float(gate_values_cpu[negative[0]].sum())
            gate_sums["negative_count"] += int(negative.sum())
        histograms = {}
        for variant, prediction in predictions.items():
            histogram = fast_histogram(prediction, target)
            totals[variant] += histogram
            event_totals[event][variant] += histogram
            histograms[variant] = histogram.long().tolist()
            if variant in change_counts:
                update_change_counts(change_counts[variant], base_prediction, prediction, target)
        image_rows.append({"name": name, "event": event, "histograms": histograms})
        processed += 1
        if processed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"S2 utility audit: {processed}/{len(dataset)} | {processed/max(elapsed, 1e-9):.2f} img/s", flush=True)

    reports = {name: metrics_from_histogram(histogram) for name, histogram in totals.items()}
    for name in metric_variants:
        reports[name]["confusion"] = totals[name].long().tolist()
    event_reports = {
        event: {name: metrics_from_histogram(histogram) for name, histogram in table.items()}
        for event, table in sorted(event_totals.items())
    }
    event_wins = {
        variant: {
            "improved_vs_fixed_s2": sum(rows[variant]["valid_miou"] > rows["u1_fixed_s2"]["valid_miou"] for rows in event_reports.values()),
            "nonnegative_vs_fixed_s2": sum(rows[variant]["valid_miou"] >= rows["u1_fixed_s2"]["valid_miou"] for rows in event_reports.values()),
            "total": len(event_reports),
        }
        for variant in variants[2:]
    }
    full = args.max_images <= 0 and (args.expected_images <= 0 or processed == args.expected_images)
    comparisons = [(variant, "u1_fixed_s2") for variant in variants[2:]]
    bootstrap = stratified_paired_bootstrap(
        image_rows,
        variants=metric_variants,
        baseline="u0_frozen_p3",
        comparison_pairs=comparisons,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    ) if full else None
    p3_miou = reports["u0_frozen_p3"]["valid_miou"]
    reproduction = {
        "expected": args.expected_p3_miou if args.expected_p3_miou > 0 else None,
        "observed": p3_miou,
        "error_pp": None if args.expected_p3_miou <= 0 else 100.0 * (p3_miou - args.expected_p3_miou),
        "passed": True if args.expected_p3_miou <= 0 else abs(p3_miou - args.expected_p3_miou) <= 1e-5,
        "basis": "locked_checkpoint_sha256" if args.expected_p3_miou <= 0 else "metric_and_checkpoint",
    }
    finalized_changes = {name: finalize_change_counts(counts) for name, counts in change_counts.items()}
    selector_ranking = binary_ranking_metrics(gate_scores, gate_labels) if gate is not None else None
    checks = None
    decision = "AUDIT_COMPLETE"
    if args.calibration_mode and gate is not None:
        fixed = reports["u1_fixed_s2"]
        learned = reports["u3_utility_gate"]
        fixed_harm = finalized_changes["u1_fixed_s2"]["harmed"]
        learned_harm = finalized_changes["u3_utility_gate"]["harmed"]
        minimum_events = math.ceil(0.60 * len(event_reports))
        checks = {
            "p3_reproduced": reproduction["passed"],
            "gate_beats_fixed_s2_ge_0_1pp": 100.0 * (learned["valid_miou"] - fixed["valid_miou"]) >= 0.10,
            "harm_reduction_ge_20pct": learned_harm <= 0.80 * max(fixed_harm, 1),
            "selector_auroc_ge_0_60": selector_ranking["auroc"] is not None and selector_ranking["auroc"] >= 0.60,
            "no_foreground_drop_vs_s2_gt_0_5pp": min(
                100.0 * (learned[key] - fixed[key]) for key in ("intact_iou", "damaged_iou", "destroyed_iou")
            ) >= -0.50,
            "event_nonnegative_vs_s2": event_wins["u3_utility_gate"]["nonnegative_vs_fixed_s2"] >= minimum_events,
        }
        decision = "PROCEED_TO_VAL450" if all(checks.values()) else "STOP_UTILITY_GATE_AFTER_INTERNAL_CALIBRATION"
    gate_statistics = None
    if gate is not None:
        gate_statistics = {
            "threshold": args.gate_threshold,
            "mean_on_changed": gate_sums["score_sum"] / max(gate_sums["changed"], 1),
            "accepted_fraction": gate_sums["accepted"] / max(gate_sums["changed"], 1),
            "mean_on_s2_only_correct": gate_sums["positive_sum"] / max(gate_sums["positive_count"], 1),
            "mean_on_p3_only_correct": gate_sums["negative_sum"] / max(gate_sums["negative_count"], 1),
            "selector_ranking": selector_ranking,
        }
    payload = {
        "protocol": "SPI_S2_UTILITY_GATE_PURE_SEMANTIC_v1",
        "split": str(Path(args.list_path).resolve()),
        "calibration_mode": args.calibration_mode,
        "processed_images": processed,
        **provenance,
        "gate": gate_manifest,
        "direction_policy_json": str(Path(args.direction_policy_json).resolve()),
        "accepted_directions": [key for key, value in policy.items() if value],
        "s2_config": config.__dict__,
        "regions": {"total": total_regions, "accepted": accepted_regions},
        "p3_reproduction": reproduction,
        "reports": reports,
        "event_reports": event_reports,
        "event_wins": event_wins,
        "change_audit": finalized_changes,
        "s2_transition_audit": finalize_transition_audit(transition_total),
        "s2_event_transition_audit": {event: finalize_transition_audit(audit) for event, audit in sorted(transition_events.items())},
        "gate_statistics": gate_statistics,
        "event_stratified_image_bootstrap": bootstrap,
        "final": {"checks": checks, "decision": decision},
        "official_instances_accessed": False,
        "test_accessed": False,
        "image_histograms": image_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 118)
    for variant in metric_variants:
        report = reports[variant]
        print(f"{variant:25s} mIoU={100*report['valid_miou']:.4f} gain_vs_P3={100*(report['valid_miou']-p3_miou):+.4f}pp D-F1={100*report['damaged_f1']:.4f}")
    print(f"P3 reproduction: {reproduction}")
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
