#!/usr/bin/env python3
"""Validation-only audit of where H6 loses native-SAR damage evidence.

This script never trains a model and deliberately accepts a validation list
only.  It compares four logits that already exist in H6-A2:

``sar_aux`` -> ``base_damage`` -> ``native_damage`` -> ``final_damage``.

The main report follows true Damaged pixels through that chain and counts
where a correct Damaged decision is changed into Background/Intact.  Results
are also split by event so a global average cannot hide event-dependent harm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    VALID_H6_VARIANTS,
    build_h6_sdcr_damageformer,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    configure_freeze_policy,
    keep_frozen_modules_in_eval,
)


VALID_PROTOCOLS = ("H6_SDCR_best_weights_v1", "H6_SDCR_resume_v1")
EVENT_RE = re.compile(r"_\d+$")
STAGES = ("sar_aux", "base", "native", "final")
OUTPUT_KEYS = {
    "sar_aux": "sar_aux_logits",
    "base": "base_damage_logits",
    "native": "native_output_dam",
    "final": "output_dam",
}
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


def event_name(sample: str) -> str:
    return EVENT_RE.sub("", str(sample))


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_counts() -> Dict[str, object]:
    return {
        "valid_pixels": 0,
        "damaged_pixels": 0,
        "affected_pixels": 0,
        "histograms": {name: torch.zeros(4, 4, dtype=torch.int64) for name in STAGES},
        "damaged_predictions": {
            name: torch.zeros(4, dtype=torch.int64) for name in STAGES
        },
        "flow": defaultdict(int),
    }


def _prediction(logits: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
    if logits.shape[-2:] != tuple(target_size):
        logits = F.interpolate(
            logits.float(), size=tuple(target_size), mode="bilinear", align_corners=False
        )
    return logits.argmax(1)


def _histogram(
    target: torch.Tensor, prediction: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    valid = (target != ignore_index) & (target >= 0) & (target < 4)
    encoded = target[valid].to(torch.int64) * 4 + prediction[valid].to(torch.int64)
    return torch.bincount(encoded, minlength=16).reshape(4, 4).cpu()


def update_counts(
    counts: MutableMapping[str, object],
    target: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    ignore_index: int = 255,
) -> None:
    """Accumulate confusion and exact Damaged evidence transitions."""
    missing = [name for name in STAGES if name not in predictions]
    if missing:
        raise KeyError(f"missing funnel predictions: {missing}")
    valid = (target != ignore_index) & (target >= 0) & (target < 4)
    damaged = valid & (target == 2)
    affected = valid & (target >= 2)
    counts["valid_pixels"] = int(counts["valid_pixels"]) + int(valid.sum())
    counts["damaged_pixels"] = int(counts["damaged_pixels"]) + int(damaged.sum())
    counts["affected_pixels"] = int(counts["affected_pixels"]) + int(affected.sum())

    histograms = counts["histograms"]
    damaged_predictions = counts["damaged_predictions"]
    for name in STAGES:
        pred = predictions[name]
        histograms[name] += _histogram(target, pred, ignore_index)
        if damaged.any():
            damaged_predictions[name] += torch.bincount(
                pred[damaged].to(torch.int64).cpu(), minlength=4
            )

    if not damaged.any():
        return
    sar = predictions["sar_aux"][damaged]
    base = predictions["base"][damaged]
    native = predictions["native"][damaged]
    final = predictions["final"][damaged]
    flow = counts["flow"]
    flow["sar_correct"] += int((sar == 2).sum())
    flow["base_correct"] += int((base == 2).sum())
    flow["native_correct"] += int((native == 2).sum())
    flow["final_correct"] += int((final == 2).sum())
    flow["sar_correct_base_correct"] += int(((sar == 2) & (base == 2)).sum())
    flow["sar_correct_base_to_gate_miss"] += int(
        ((sar == 2) & (base <= 1)).sum()
    )
    flow["sar_wrong_base_correct"] += int(((sar != 2) & (base == 2)).sum())
    flow["base_correct_native_correct"] += int(((base == 2) & (native == 2)).sum())
    flow["base_correct_native_to_gate_miss"] += int(
        ((base == 2) & (native <= 1)).sum()
    )
    flow["base_wrong_native_correct"] += int(((base != 2) & (native == 2)).sum())
    flow["native_correct_final_to_gate_miss"] += int(
        ((native == 2) & (final <= 1)).sum()
    )
    flow["all_stages_gate_miss"] += int(
        ((sar <= 1) & (base <= 1) & (native <= 1) & (final <= 1)).sum()
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _stage_metrics(histogram: torch.Tensor) -> Dict[str, object]:
    hist = histogram.to(torch.float64)
    damaged_total = float(hist[2].sum())
    affected_total = float(hist[2:4].sum())
    damaged_prediction = hist[2]
    affected_tp = float(hist[2:4, 2:4].sum())
    return {
        "confusion": hist.to(torch.int64).tolist(),
        "damaged_recall": _safe_ratio(hist[2, 2], damaged_total),
        "damaged_gate_miss": _safe_ratio(hist[2, 0:2].sum(), damaged_total),
        "damaged_to_destroyed": _safe_ratio(hist[2, 3], damaged_total),
        "affected_recall": _safe_ratio(affected_tp, affected_total),
        "damaged_prediction_fraction": {
            CLASS_NAMES[index]: _safe_ratio(damaged_prediction[index], damaged_total)
            for index in range(4)
        },
    }


def finalize_counts(counts: Mapping[str, object]) -> Dict[str, object]:
    damaged_total = int(counts["damaged_pixels"])
    flow = {key: int(value) for key, value in counts["flow"].items()}
    flow_rates = {
        "sar_correct_base_to_gate_miss_given_sar_correct": _safe_ratio(
            flow.get("sar_correct_base_to_gate_miss", 0), flow.get("sar_correct", 0)
        ),
        "base_correct_native_to_gate_miss_given_base_correct": _safe_ratio(
            flow.get("base_correct_native_to_gate_miss", 0), flow.get("base_correct", 0)
        ),
        "native_correct_final_to_gate_miss_given_native_correct": _safe_ratio(
            flow.get("native_correct_final_to_gate_miss", 0), flow.get("native_correct", 0)
        ),
        "all_stages_gate_miss_fraction_of_damaged": _safe_ratio(
            flow.get("all_stages_gate_miss", 0), damaged_total
        ),
        "sar_wrong_base_correct_fraction_of_damaged": _safe_ratio(
            flow.get("sar_wrong_base_correct", 0), damaged_total
        ),
        "base_wrong_native_correct_fraction_of_damaged": _safe_ratio(
            flow.get("base_wrong_native_correct", 0), damaged_total
        ),
    }
    return {
        "valid_pixels": int(counts["valid_pixels"]),
        "damaged_pixels": damaged_total,
        "affected_pixels": int(counts["affected_pixels"]),
        "stages": {
            name: _stage_metrics(counts["histograms"][name]) for name in STAGES
        },
        "damaged_evidence_flow_counts": flow,
        "damaged_evidence_flow_rates": flow_rates,
    }


def diagnosis(report: Mapping[str, object]) -> Dict[str, object]:
    rates = report["damaged_evidence_flow_rates"]
    stages = report["stages"]
    sar_recall = float(stages["sar_aux"]["damaged_recall"])
    base_recall = float(stages["base"]["damaged_recall"])
    native_recall = float(stages["native"]["damaged_recall"])
    fusion_loss = float(rates["sar_correct_base_to_gate_miss_given_sar_correct"])
    hierarchy_loss = float(
        rates["base_correct_native_to_gate_miss_given_base_correct"]
    )
    all_missed = float(rates["all_stages_gate_miss_fraction_of_damaged"])

    findings = []
    if fusion_loss >= 0.10:
        findings.append("FUSION_OR_DECODER_DISCARDS_CORRECT_SAR_DAMAGED_EVIDENCE")
    if hierarchy_loss >= 0.05:
        findings.append("HIERARCHY_TURNS_CORRECT_DAMAGED_INTO_NON_AFFECTED")
    if all_missed >= 0.20:
        findings.append("DAMAGED_EVIDENCE_IS_MISSING_ACROSS_ALL_EXISTING_HEADS")
    if sar_recall < base_recall:
        findings.append("FUSION_RESCUES_MORE_DAMAGED_THAN_THE_CURRENT_SAR_AUX_HEAD")
    if native_recall > base_recall:
        findings.append("HIERARCHY_HAS_NET_DAMAGED_RECALL_BENEFIT")
    if not findings:
        findings.append("NO_SINGLE_DOMINANT_LOGIT_FUNNEL_FAILURE")
    return {
        "thresholds_are_diagnostic_not_selection_rules": True,
        "findings": findings,
        "sar_aux_damaged_recall": sar_recall,
        "base_damaged_recall": base_recall,
        "native_damaged_recall": native_recall,
    }


@torch.inference_mode()
def audit(model, loader, device, ignore_index: int, max_images: int):
    overall = empty_counts()
    events: Dict[str, Dict[str, object]] = defaultdict(empty_counts)
    processed = 0
    for batch in loader:
        rgb, sar, _, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar, return_dict=True)
        missing = [key for key in OUTPUT_KEYS.values() if key not in outputs]
        if missing:
            raise RuntimeError(f"model does not expose required funnel outputs: {missing}")
        predictions = {
            name: _prediction(outputs[key], damage.shape[-2:])
            for name, key in OUTPUT_KEYS.items()
        }
        update_counts(overall, damage, predictions, ignore_index)
        for batch_index, name in enumerate(names):
            event = event_name(name)
            update_counts(
                events[event],
                damage[batch_index : batch_index + 1],
                {
                    key: value[batch_index : batch_index + 1]
                    for key, value in predictions.items()
                },
                ignore_index,
            )
        processed += len(names)
        if processed % 50 == 0:
            print(f"H6 information funnel audit: {processed}", flush=True)
        if max_images and processed >= max_images:
            break
    overall_report = finalize_counts(overall)
    event_reports = {name: finalize_counts(value) for name, value in sorted(events.items())}
    return overall_report, event_reports, processed


def main() -> None:
    parser = argparse.ArgumentParser("H6 validation-only information funnel audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol not in VALID_PROTOCOLS:
        raise RuntimeError(f"unsupported H6 checkpoint protocol: {protocol!r}")
    if variant not in VALID_H6_VARIANTS:
        raise RuntimeError(f"unsupported H6 variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("checkpoint has no model state")
    provenance = checkpoint.get("provenance", {})
    freeze_mode = (
        provenance.get("sar_freeze_mode", "f0_shallow_frozen")
        if isinstance(provenance, Mapping)
        else "f0_shallow_frozen"
    )
    model = build_h6_sdcr_damageformer(variant, pretrained_rgb=False)
    model.load_state_dict(state, strict=True)
    configure_freeze_policy(model, mode=str(freeze_mode))
    device = torch.device(args.device)
    model.to(device).eval()
    keep_frozen_modules_in_eval(model)

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.val_list,
        args.crop_size,
        split="val",
        variant=("h4" if variant in ("b0", "b1", "b2", "b1r") else "h3"),
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    overall, events, processed = audit(
        model, loader, device, args.ignore_index, args.max_images
    )
    payload = {
        "protocol": "H6_information_funnel_audit_v1",
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": file_sha256(args.checkpoint),
            "variant": variant,
            "step": int(checkpoint.get("step", 0)),
            "sar_freeze_mode": str(freeze_mode),
            "strict_load": True,
        },
        "data": {
            "validation_list": str(Path(args.val_list).resolve()),
            "processed_images": processed,
        },
        "training": "NONE",
        "test_accessed": False,
        "overall": overall,
        "diagnosis": diagnosis(overall),
        "events": events,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=" * 102)
    print("H6 validation-only SAR information funnel")
    print(
        f"{'Stage':<10} {'DmgRecall':>10} {'GateMiss':>10} "
        f"{'Dmg->Des':>10} {'AffRecall':>10}"
    )
    print("-" * 58)
    for name in STAGES:
        item = overall["stages"][name]
        print(
            f"{name:<10} {100*item['damaged_recall']:10.3f} "
            f"{100*item['damaged_gate_miss']:10.3f} "
            f"{100*item['damaged_to_destroyed']:10.3f} "
            f"{100*item['affected_recall']:10.3f}"
        )
    rates = overall["damaged_evidence_flow_rates"]
    print("-" * 102)
    print(
        "SAR-correct -> base gate-miss: "
        f"{100*rates['sar_correct_base_to_gate_miss_given_sar_correct']:.3f}%"
    )
    print(
        "Base-correct -> hierarchy gate-miss: "
        f"{100*rates['base_correct_native_to_gate_miss_given_base_correct']:.3f}%"
    )
    print(
        "All stages gate-miss / true Damaged: "
        f"{100*rates['all_stages_gate_miss_fraction_of_damaged']:.3f}%"
    )
    ranked_events = sorted(
        events.items(),
        key=lambda item: item[1]["damaged_evidence_flow_rates"][
            "sar_correct_base_to_gate_miss_given_sar_correct"
        ],
        reverse=True,
    )
    print("-" * 102)
    print(
        f"{'Event':<30} {'Damaged':>10} {'SARrec':>9} "
        f"{'Baserec':>9} {'FusionLoss':>11} {'AllMiss':>9}"
    )
    for event, item in ranked_events:
        event_rates = item["damaged_evidence_flow_rates"]
        event_stages = item["stages"]
        print(
            f"{event:<30} {item['damaged_pixels']:10d} "
            f"{100*event_stages['sar_aux']['damaged_recall']:9.3f} "
            f"{100*event_stages['base']['damaged_recall']:9.3f} "
            f"{100*event_rates['sar_correct_base_to_gate_miss_given_sar_correct']:11.3f} "
            f"{100*event_rates['all_stages_gate_miss_fraction_of_damaged']:9.3f}"
        )
    print("Findings: " + ", ".join(payload["diagnosis"]["findings"]))
    print(f"Saved: {output}")
    print("Training/test: NONE / NEVER")
    print("=" * 102)


if __name__ == "__main__":
    main()
