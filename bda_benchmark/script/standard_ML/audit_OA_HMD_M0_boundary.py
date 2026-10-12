# -*- coding: utf-8 -*-
"""Validation-only P3 Intact/Affected boundary audit (M0).

The frozen P3 model is evaluated once.  A declared set of constant occurrence
offsets is then scored in parallel from the same logits.  Ground truth is used
only for metrics; neither the support nor the predictions consume labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (  # noqa: E402
    factorize_building_logits,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.script.standard_ML.m0_boundary_runtime import (  # noqa: E402
    M0_THRESHOLDS_PP,
    apply_occurrence_offset,
    choose_global_candidate,
    deltas_pp,
    finish_metrics,
    global_boundary_gate,
    new_occurrence_counters,
    normalize_betas,
    update_occurrence_counters,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (  # noqa: E402
    fast_hist,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_betas(text: str) -> tuple[float, ...]:
    return normalize_betas(
        float(token) for token in re.split(r"[\s,]+", text.strip()) if token
    )


def event_name(sample_name: object) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def histogram_metrics(histogram: torch.Tensor) -> Dict[str, float]:
    intersection = torch.diag(histogram)
    union = histogram.sum(1) + histogram.sum(0) - intersection
    iou = intersection / union.clamp_min(1.0)
    return {
        "miou4": float(iou.mean()),
        "valid_miou": float(iou[1:4].mean()),
        "intact_iou": float(iou[1]),
        "damaged_iou": float(iou[2]),
        "destroyed_iou": float(iou[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser("OA-HMD M0 boundary audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_baseline_json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--betas",
        default="-0.50 -0.40 -0.30 -0.20 -0.10 0.00 0.10 0.20 0.30 0.40 0.50",
    )
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("P3 checkpoint has no state dictionary")
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load unexpectedly failed: {incompatible}")
    model.to(device).eval()

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split="val",
        variant="h0",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = {
        beta: torch.zeros(4, 4, dtype=torch.float64, device=device)
        for beta in betas
    }
    counters = {beta: new_occurrence_counters() for beta in betas}
    event_hist: Dict[float, Dict[str, torch.Tensor]] = {
        beta: {} for beta in betas
    }
    event_images: Dict[str, int] = defaultdict(int)
    processed = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            rgb, sar, location, damage, names = batch
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            location = location.to(device, non_blocking=True).long()
            damage = damage.to(device, non_blocking=True).long()
            loc_logits, base_logits = model(rgb, sar)
            loc_prediction = loc_logits.argmax(1)
            loc_hist += fast_hist(
                loc_prediction, location, 2, args.ignore_index
            )

            for beta in betas:
                updated, _ = apply_occurrence_offset(
                    base_logits, loc_logits, beta
                )
                prediction = updated.argmax(1)
                occurrence, severity = factorize_building_logits(updated)
                occurrence_prediction = occurrence[:, 0] >= 0
                severity_prediction = severity[:, 0] >= 0
                damage_hist[beta] += fast_hist(
                    prediction, damage, 4, args.ignore_index
                )
                update_occurrence_counters(
                    counters[beta],
                    occurrence_prediction,
                    severity_prediction,
                    damage,
                    args.ignore_index,
                )
                for sample_index, name in enumerate(names):
                    event = event_name(name)
                    sample_hist = fast_hist(
                        prediction[sample_index : sample_index + 1],
                        damage[sample_index : sample_index + 1],
                        4,
                        args.ignore_index,
                    )
                    old = event_hist[beta].get(event)
                    event_hist[beta][event] = (
                        sample_hist if old is None else old + sample_hist
                    )
            for name in names:
                event_images[event_name(name)] += 1
            processed += int(rgb.shape[0])
            if processed % 50 == 0:
                print(f"OA-HMD M0 boundary audit: {processed}", flush=True)

    raw_records = []
    for beta in betas:
        metrics = finish_metrics(
            loc_hist,
            damage_hist[beta],
            counters[beta],
            event_hist[beta],
            processed,
        )
        events = {
            event: {
                "event": event,
                "processed_images": int(event_images[event]),
                **histogram_metrics(histogram),
            }
            for event, histogram in sorted(event_hist[beta].items())
        }
        raw_records.append({"beta": beta, "metrics": metrics, "event_metrics": events})

    baseline_record = next(
        record for record in raw_records if float(record["beta"]) == 0.0
    )
    baseline = baseline_record["metrics"]
    baseline_events = baseline_record["event_metrics"]
    baseline_guard = {
        "checked": False,
        "path": "",
        "tolerance": 2e-6,
        "deltas": {},
    }
    if args.expected_baseline_json:
        expected_payload = json.loads(
            Path(args.expected_baseline_json).read_text(encoding="utf-8")
        )
        expected = None
        for key in ("observed_metrics", "metrics", "best_metrics"):
            value = expected_payload.get(key)
            if isinstance(value, Mapping):
                expected = value
                break
        if expected is None:
            raise KeyError("expected baseline JSON has no metric mapping")
        required = (
            "valid_miou",
            "miou4",
            "damaged_iou",
            "destroyed_iou",
            "loc_f1",
        )
        guard_deltas = {
            key: abs(float(baseline[key]) - float(expected[key]))
            for key in required
        }
        failed = {
            key: value for key, value in guard_deltas.items() if value > 2e-6
        }
        if failed:
            raise RuntimeError(f"M0 beta=0 failed exact P3 baseline guard: {failed}")
        baseline_guard = {
            "checked": True,
            "path": str(Path(args.expected_baseline_json).resolve()),
            "tolerance": 2e-6,
            "deltas": guard_deltas,
        }
    records = []
    for record in raw_records:
        delta = deltas_pp(record["metrics"], baseline)
        event_delta = {
            event: 100.0 * (
                float(values["miou4"]) - float(baseline_events[event]["miou4"])
            )
            for event, values in record["event_metrics"].items()
        }
        nonnegative = sum(value >= 0.0 for value in event_delta.values())
        gate = global_boundary_gate(delta, nonnegative)
        records.append(
            {
                **record,
                "delta_pp": delta,
                "event_miou4_delta_pp": event_delta,
                "nonnegative_events": nonnegative,
                "gate": gate,
            }
        )

    candidate = choose_global_candidate(records)
    passed = [
        record for record in records
        if float(record["beta"]) != 0.0 and bool(record["gate"]["passed"])
    ]

    per_event_best = {}
    direction_counts = {"negative": 0, "zero": 0, "positive": 0}
    useful_event_count = 0
    for event in sorted(baseline_events):
        choices = [
            (
                float(record["event_metrics"][event]["damaged_iou"]),
                float(record["event_metrics"][event]["miou4"]),
                -abs(float(record["beta"])),
                record,
            )
            for record in records
        ]
        best = max(choices, key=lambda item: item[:3])[3]
        beta = float(best["beta"])
        direction = "positive" if beta > 0 else "negative" if beta < 0 else "zero"
        direction_counts[direction] += 1
        damage_gain = 100.0 * (
            float(best["event_metrics"][event]["damaged_iou"])
            - float(baseline_events[event]["damaged_iou"])
        )
        miou_gain = 100.0 * (
            float(best["event_metrics"][event]["miou4"])
            - float(baseline_events[event]["miou4"])
        )
        if damage_gain >= 0.30 and miou_gain >= 0.0:
            useful_event_count += 1
        per_event_best[event] = {
            "beta": beta,
            "direction": direction,
            "damaged_iou_delta_pp": damage_gain,
            "miou4_delta_pp": miou_gain,
        }

    direction_conflict = (
        direction_counts["negative"] >= 2 and direction_counts["positive"] >= 2
    )
    if passed:
        decision = "PROCEED_M1_GLOBAL_ORDINAL_CALIBRATION"
    elif direction_conflict and useful_event_count >= 6:
        decision = "PROCEED_EVENT_CONDITIONED_CALIBRATION_AUDIT"
    else:
        decision = "STOP_GLOBAL_CALIBRATION_REPRESENTATION_REQUIRED"

    payload = {
        "protocol": "OA_HMD_M0_boundary_audit_v1",
        "split": "val",
        "training": False,
        "test_accessed": False,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
            "strict_load": True,
            "source_step": checkpoint.get("step"),
        },
        "offset_contract": {
            "target": "Intact-vs-Affected only",
            "support": "frozen hard P3 building prediction",
            "severity_odds_preserved": True,
            "ground_truth_used_for_prediction": False,
            "betas": list(betas),
        },
        "exact_baseline_guard": baseline_guard,
        "thresholds_pp": M0_THRESHOLDS_PP,
        "baseline": baseline_record,
        "records": records,
        "selected_global_candidate_beta": float(candidate["beta"]),
        "global_candidate_passed": bool(candidate["gate"]["passed"]),
        "per_event_best": per_event_best,
        "per_event_direction_counts": direction_counts,
        "direction_conflict": direction_conflict,
        "useful_event_count": useful_event_count,
        "decision": decision,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    beta_tsv = output.with_suffix(".betas.tsv")
    with beta_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "beta\tvalid_miou\tdamaged_iou\taffected_recall\tdamaged_gate_miss"
            "\tintact_to_affected\tevent_mean\tevent_worst\tnonnegative_events\tgate\n"
        )
        for record in records:
            metrics = record["metrics"]
            handle.write(
                f"{record['beta']:.2f}\t{metrics['valid_miou']:.8f}"
                f"\t{metrics['damaged_iou']:.8f}\t{metrics['affected_recall']:.8f}"
                f"\t{metrics['damaged_gate_miss']:.8f}"
                f"\t{metrics['intact_to_affected']:.8f}"
                f"\t{metrics['event_mean_official_style_4class_miou']:.8f}"
                f"\t{metrics['event_worst_official_style_4class_miou']:.8f}"
                f"\t{record['nonnegative_events']}\t{int(record['gate']['passed'])}\n"
            )

    event_tsv = output.with_suffix(".events.tsv")
    with event_tsv.open("w", encoding="utf-8") as handle:
        handle.write("event\tbest_beta\tdirection\tdamaged_iou_delta_pp\tmiou4_delta_pp\n")
        for event, values in per_event_best.items():
            handle.write(
                f"{event}\t{values['beta']:.2f}\t{values['direction']}"
                f"\t{values['damaged_iou_delta_pp']:.6f}"
                f"\t{values['miou4_delta_pp']:.6f}\n"
            )

    print("=" * 104)
    print("OA-HMD M0 | frozen P3 Intact/Affected boundary audit | validation only")
    print(" beta    Valid      Dmg   AffRec  GateMiss  Int->Aff    Event    Worst  nonneg pass")
    print("-" * 104)
    for record in records:
        metrics = record["metrics"]
        print(
            f"{record['beta']:+5.2f}  {100*metrics['valid_miou']:8.4f}"
            f" {100*metrics['damaged_iou']:8.4f} {100*metrics['affected_recall']:8.4f}"
            f" {100*metrics['damaged_gate_miss']:9.4f}"
            f" {100*metrics['intact_to_affected']:9.4f}"
            f" {100*metrics['event_mean_official_style_4class_miou']:8.4f}"
            f" {100*metrics['event_worst_official_style_4class_miou']:8.4f}"
            f" {record['nonnegative_events']:7d} {str(record['gate']['passed']):>5}"
        )
    print(f"Selected global candidate : beta={float(candidate['beta']):+.2f}")
    print(f"Per-event beta directions : {direction_counts}")
    print(f"Decision                  : {decision}")
    print(f"Saved                     : {output}")
    print(f"Beta TSV                  : {beta_tsv}")
    print(f"Event TSV                 : {event_tsv}")
    print("Training/Test accessed    : NONE / False")


if __name__ == "__main__":
    main()
