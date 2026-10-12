# -*- coding: utf-8 -*-
"""Validation-only complementarity audit for the frozen P3 and H6-A2 experts.

This is deliberately an analysis program, not a fusion model.  It answers the
question that must precede H7-CER: does the native-SAR expert contain enough
correct, complementary evidence to justify learning a router?

The two experts are evaluated on the same samples and geometry in one pass:

* P3 receives the historical repeated display-SAR/ImageNet-normalised view.
* H6-A2 receives the train-statistics-normalised native one-channel SAR view.

Ground truth is used only to compute a correctness oracle on the validation
split.  The oracle is an upper-bound diagnostic and must never be deployed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    build_h6_sdcr_damageformer,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
    connected_components_4,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
EXPECTED_P3_SHA256 = (
    "45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7"
)
VALID_A2_PROTOCOLS = ("H6_SDCR_best_weights_v1", "H6_SDCR_resume_v1")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("checkpoint does not contain a model state dictionary")
    return state  # type: ignore[return-value]


def fast_histogram(
    prediction: torch.Tensor,
    target: torch.Tensor,
    classes: int = 4,
    ignore_index: int = 255,
) -> torch.Tensor:
    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target != ignore_index) & (target >= 0) & (target < classes)
    encoded = classes * target[valid] + prediction[valid]
    return torch.bincount(encoded, minlength=classes * classes).reshape(
        classes, classes
    ).to(torch.float64)


def metrics_from_histogram(histogram: torch.Tensor) -> Dict[str, float]:
    hist = histogram.to(torch.float64)
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / union.clamp_min(1.0)
    f1 = 2.0 * intersection / (hist.sum(1) + hist.sum(0)).clamp_min(1.0)
    total = hist.sum().clamp_min(1.0)
    output: Dict[str, float] = {
        "oa": float((intersection.sum() / total).item()),
        "miou4": float(iou.mean().item()),
        "valid_miou": float(iou[1:4].mean().item()),
    }
    for index, name in enumerate(CLASS_NAMES):
        output[f"{name}_iou"] = float(iou[index].item())
        output[f"{name}_f1"] = float(f1[index].item())
    output["damaged_to_intact"] = float(
        (hist[2, 1] / hist[2].sum().clamp_min(1.0)).item()
    )
    output["destroyed_to_intact"] = float(
        (hist[3, 1] / hist[3].sum().clamp_min(1.0)).item()
    )
    return output


def complementarity_counts(
    p3_prediction: torch.Tensor,
    a2_prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> Dict[str, int]:
    valid = target != ignore_index
    p3_correct = (p3_prediction == target) & valid
    a2_correct = (a2_prediction == target) & valid
    disagreement = (p3_prediction != a2_prediction) & valid
    return {
        "valid": int(valid.sum().item()),
        "both_correct": int((p3_correct & a2_correct).sum().item()),
        "p3_only_correct": int((p3_correct & ~a2_correct).sum().item()),
        "a2_only_correct": int((~p3_correct & a2_correct).sum().item()),
        "neither_correct": int((~p3_correct & ~a2_correct & valid).sum().item()),
        "disagreement": int(disagreement.sum().item()),
        "disagreement_neither_correct": int(
            (disagreement & ~p3_correct & ~a2_correct).sum().item()
        ),
    }


def correctness_oracle_prediction(
    p3_prediction: torch.Tensor,
    a2_prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Select A2 exactly where it is correct and P3 is wrong.

    This is a label-aware diagnostic upper bound, not an inference algorithm.
    """

    use_a2 = (
        (target != ignore_index)
        & (a2_prediction == target)
        & (p3_prediction != target)
    )
    return torch.where(use_a2, a2_prediction, p3_prediction)


def expert_transition_cube(
    p3_prediction: torch.Tensor,
    a2_prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Return counts indexed as ``[ground_truth, P3, A2]``."""

    p3_prediction = p3_prediction.reshape(-1).long()
    a2_prediction = a2_prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target != ignore_index) & (target >= 0) & (target < 4)
    encoded = (
        16 * target[valid] + 4 * p3_prediction[valid] + a2_prediction[valid]
    )
    return torch.bincount(encoded, minlength=64).reshape(4, 4, 4).to(torch.int64)


def transition_records(cube: torch.Tensor) -> list[Dict[str, object]]:
    """Summarise every P3-to-A2 class proposal and its correctness balance."""

    records: list[Dict[str, object]] = []
    for p3_class, p3_name in enumerate(CLASS_NAMES):
        for a2_class, a2_name in enumerate(CLASS_NAMES):
            if p3_class == a2_class:
                continue
            by_target = cube[:, p3_class, a2_class]
            total = int(by_target.sum().item())
            p3_correct = int(by_target[p3_class].item())
            a2_correct = int(by_target[a2_class].item())
            neither_correct = total - p3_correct - a2_correct
            records.append(
                {
                    "p3_class": p3_name,
                    "a2_class": a2_name,
                    "count": total,
                    "p3_correct": p3_correct,
                    "a2_correct": a2_correct,
                    "neither_correct": neither_correct,
                    "a2_precision_within_transition": a2_correct / max(total, 1),
                    "net_a2_correct": a2_correct - p3_correct,
                    "ground_truth_counts": {
                        CLASS_NAMES[index]: int(by_target[index].item())
                        for index in range(4)
                    },
                }
            )
    return sorted(records, key=lambda value: int(value["net_a2_correct"]), reverse=True)


def proposal_route_audit(
    cube: torch.Tensor,
    a2_proposal_class: int = 2,
) -> Dict[str, object]:
    """Audit a route selected by A2's prediction rather than by GT class."""

    if not 0 <= a2_proposal_class < len(CLASS_NAMES):
        raise ValueError("a2_proposal_class is outside the class range")
    total = p3_correct = a2_correct = neither_correct = 0
    by_p3_class: Dict[str, Dict[str, int]] = {}
    for p3_class, p3_name in enumerate(CLASS_NAMES):
        if p3_class == a2_proposal_class:
            continue
        by_target = cube[:, p3_class, a2_proposal_class]
        local_total = int(by_target.sum().item())
        local_p3 = int(by_target[p3_class].item())
        local_a2 = int(by_target[a2_proposal_class].item())
        local_neither = local_total - local_p3 - local_a2
        by_p3_class[p3_name] = {
            "count": local_total,
            "p3_correct": local_p3,
            "a2_correct": local_a2,
            "neither_correct": local_neither,
            "net_a2_correct": local_a2 - local_p3,
        }
        total += local_total
        p3_correct += local_p3
        a2_correct += local_a2
        neither_correct += local_neither
    one_expert_correct = p3_correct + a2_correct
    passed = a2_correct > p3_correct and one_expert_correct > 0
    return {
        "selection": (
            f"P3_prediction != A2_prediction and "
            f"A2_prediction == {CLASS_NAMES[a2_proposal_class]}"
        ),
        "a2_proposal_class": CLASS_NAMES[a2_proposal_class],
        "count": total,
        "p3_correct": p3_correct,
        "a2_correct": a2_correct,
        "neither_correct": neither_correct,
        "net_a2_correct": a2_correct - p3_correct,
        "a2_win_rate_when_one_expert_correct": a2_correct
        / max(one_expert_correct, 1),
        "by_p3_class": by_p3_class,
        "passed": passed,
        "recommendation": (
            "TRAIN_DAMAGED_COUNTERFACTUAL_ROUTER"
            if passed
            else "DO_NOT_TRAIN_DAMAGED_COUNTERFACTUAL_ROUTER"
        ),
    }


class CalibrationBins:
    def __init__(self, bin_count: int = 15) -> None:
        if bin_count <= 1:
            raise ValueError("bin_count must be greater than one")
        self.bin_count = int(bin_count)
        self.count = np.zeros(self.bin_count, dtype=np.int64)
        self.confidence_sum = np.zeros(self.bin_count, dtype=np.float64)
        self.correct_sum = np.zeros(self.bin_count, dtype=np.float64)

    def update(
        self,
        confidence: torch.Tensor,
        correct: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        confidence = confidence[valid].detach().float().cpu().numpy()
        correct = correct[valid].detach().float().cpu().numpy()
        if not confidence.size:
            return
        indices = np.minimum(
            (confidence * self.bin_count).astype(np.int64), self.bin_count - 1
        )
        self.count += np.bincount(indices, minlength=self.bin_count)
        self.confidence_sum += np.bincount(
            indices, weights=confidence, minlength=self.bin_count
        )
        self.correct_sum += np.bincount(
            indices, weights=correct, minlength=self.bin_count
        )

    def result(self) -> Dict[str, object]:
        total = max(int(self.count.sum()), 1)
        bins = []
        ece = 0.0
        for index in range(self.bin_count):
            count = int(self.count[index])
            mean_confidence = (
                float(self.confidence_sum[index] / count) if count else 0.0
            )
            accuracy = float(self.correct_sum[index] / count) if count else 0.0
            ece += count / total * abs(accuracy - mean_confidence)
            bins.append(
                {
                    "lower": index / self.bin_count,
                    "upper": (index + 1) / self.bin_count,
                    "count": count,
                    "mean_confidence": mean_confidence,
                    "accuracy": accuracy,
                }
            )
        return {"ece": float(ece), "sample_count": int(self.count.sum()), "bins": bins}


def merge_counts(destination: MutableMapping[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        destination[key] = int(destination.get(key, 0)) + int(value)


def rates(counts: Mapping[str, int]) -> Dict[str, float]:
    valid = max(int(counts.get("valid", 0)), 1)
    disagreement = max(int(counts.get("disagreement", 0)), 1)
    a2_gain = int(counts.get("a2_only_correct", 0))
    p3_harm = int(counts.get("p3_only_correct", 0))
    return {
        "both_correct_rate": counts.get("both_correct", 0) / valid,
        "p3_only_correct_rate": p3_harm / valid,
        "a2_only_correct_rate": a2_gain / valid,
        "neither_correct_rate": counts.get("neither_correct", 0) / valid,
        "disagreement_rate": counts.get("disagreement", 0) / valid,
        "a2_advantage_among_disagreements": a2_gain / disagreement,
        "p3_advantage_among_disagreements": p3_harm / disagreement,
        "net_a2_correct_pixels": a2_gain - p3_harm,
    }


def majority_label(prediction: np.ndarray, mask: np.ndarray) -> int:
    values = prediction[mask]
    if values.size == 0:
        raise ValueError("cannot compute a majority label over an empty mask")
    return int(np.bincount(values.astype(np.int64), minlength=4).argmax())


def update_building_audit(
    totals: MutableMapping[str, int],
    per_class: MutableMapping[str, MutableMapping[str, int]],
    p3_prediction: np.ndarray,
    a2_prediction: np.ndarray,
    target: np.ndarray,
    min_area: int,
) -> None:
    valid_building = (target >= 1) & (target <= 3)
    for class_index, class_name in enumerate(CLASS_NAMES[1:], start=1):
        class_mask = valid_building & (target == class_index)
        for component in connected_components_4(class_mask):
            area = int(component.sum())
            if area < min_area:
                continue
            p3_label = majority_label(p3_prediction, component)
            a2_label = majority_label(a2_prediction, component)
            p3_correct = p3_label == class_index
            a2_correct = a2_label == class_index
            category = (
                "both_correct"
                if p3_correct and a2_correct
                else "p3_only_correct"
                if p3_correct
                else "a2_only_correct"
                if a2_correct
                else "neither_correct"
            )
            totals["valid"] = int(totals.get("valid", 0)) + 1
            totals[category] = int(totals.get(category, 0)) + 1
            totals["disagreement"] = int(totals.get("disagreement", 0)) + int(
                p3_label != a2_label
            )
            class_counts = per_class.setdefault(class_name, defaultdict(int))
            class_counts["valid"] += 1
            class_counts[category] += 1
            class_counts["disagreement"] += int(p3_label != a2_label)


def p3_sar_view(post_sar: torch.Tensor) -> torch.Tensor:
    if post_sar.ndim != 4 or post_sar.shape[1] != 2:
        raise RuntimeError(
            "oracle analysis requires the H4 dual-SAR view: native channel + "
            "unclipped display channel"
        )
    display = post_sar[:, 1:2].float()
    mean = display.new_tensor([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = display.new_tensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    return ((display.repeat(1, 3, 1, 1) - mean) / std).to(post_sar.dtype)


def event_name(sample_name: str) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def event_rows(
    histograms: Mapping[str, Mapping[str, torch.Tensor]],
    event_counts: Mapping[str, Mapping[str, int]],
    event_transition_cubes: Mapping[str, torch.Tensor] | None = None,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for event in sorted(histograms):
        p3 = metrics_from_histogram(histograms[event]["p3"])
        a2 = metrics_from_histogram(histograms[event]["a2"])
        oracle = metrics_from_histogram(histograms[event]["oracle"])
        counts = dict(event_counts[event])
        row: Dict[str, object] = {
            "event": event,
            "p3": p3,
            "a2": a2,
            "oracle": oracle,
            "counts": counts,
            "rates": rates(counts),
            "oracle_minus_p3_pp": {
                key: 100.0 * (oracle[key] - p3[key])
                for key in ("miou4", "valid_miou", "damaged_iou", "destroyed_iou")
            },
            "a2_minus_p3_pp": {
                key: 100.0 * (a2[key] - p3[key])
                for key in ("miou4", "valid_miou", "damaged_iou", "destroyed_iou")
            },
            "confusion": {
                key: histograms[event][key].to(torch.int64).tolist()
                for key in ("p3", "a2", "oracle")
            },
        }
        if event_transition_cubes is not None:
            row["expert_transition_cube"] = event_transition_cubes[event].tolist()
            row["expert_transition_records"] = transition_records(
                event_transition_cubes[event]
            )
        rows.append(row)
    return rows


def write_event_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    header = (
        "event\tp3_valid\ta2_valid\toracle_valid\toracle_gain_pp\t"
        "p3_damaged\ta2_damaged\toracle_damaged\ta2_only_correct\t"
        "p3_only_correct\tdisagreement\n"
    )
    lines = [header]
    for row in rows:
        p3 = row["p3"]
        a2 = row["a2"]
        oracle = row["oracle"]
        counts = row["counts"]
        delta = row["oracle_minus_p3_pp"]
        assert isinstance(p3, Mapping) and isinstance(a2, Mapping)
        assert isinstance(oracle, Mapping) and isinstance(counts, Mapping)
        assert isinstance(delta, Mapping)
        lines.append(
            f"{row['event']}\t{100*float(p3['valid_miou']):.6f}\t"
            f"{100*float(a2['valid_miou']):.6f}\t"
            f"{100*float(oracle['valid_miou']):.6f}\t"
            f"{float(delta['valid_miou']):.6f}\t"
            f"{100*float(p3['damaged_iou']):.6f}\t"
            f"{100*float(a2['damaged_iou']):.6f}\t"
            f"{100*float(oracle['damaged_iou']):.6f}\t"
            f"{int(counts.get('a2_only_correct', 0))}\t"
            f"{int(counts.get('p3_only_correct', 0))}\t"
            f"{int(counts.get('disagreement', 0))}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def make_decision(
    p3_metrics: Mapping[str, float],
    oracle_metrics: Mapping[str, float],
    events: Sequence[Mapping[str, object]],
    minimum_valid_gain_pp: float,
    minimum_damaged_gain_pp: float,
    minimum_positive_event_fraction: float,
) -> Dict[str, object]:
    valid_gain = 100.0 * (
        float(oracle_metrics["valid_miou"]) - float(p3_metrics["valid_miou"])
    )
    damaged_gain = 100.0 * (
        float(oracle_metrics["damaged_iou"]) - float(p3_metrics["damaged_iou"])
    )
    positive_events = 0
    for row in events:
        delta = row["oracle_minus_p3_pp"]
        assert isinstance(delta, Mapping)
        positive_events += int(float(delta["valid_miou"]) > 1e-9)
    fraction = positive_events / max(len(events), 1)
    checks = {
        "oracle_valid_gain": valid_gain >= minimum_valid_gain_pp,
        "oracle_damaged_gain": damaged_gain >= minimum_damaged_gain_pp,
        "positive_event_fraction": fraction >= minimum_positive_event_fraction,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "recommendation": (
            "COMPLEMENTARITY_CONFIRMED_DESIGN_LEARNED_SELECTOR"
            if passed
            else "UPGRADE_SAR_EXPERT_BEFORE_ROUTING"
        ),
        "observed": {
            "oracle_valid_gain_pp": valid_gain,
            "oracle_damaged_gain_pp": damaged_gain,
            "positive_event_count": positive_events,
            "event_count": len(events),
            "positive_event_fraction": fraction,
        },
        "thresholds": {
            "minimum_valid_gain_pp": minimum_valid_gain_pp,
            "minimum_damaged_gain_pp": minimum_damaged_gain_pp,
            "minimum_positive_event_fraction": minimum_positive_event_fraction,
        },
        "checks": checks,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("H7 validation-only P3/A2 oracle audit")
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--a2_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_event_tsv", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--calibration_bins", type=int, default=15)
    parser.add_argument("--minimum_building_area", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--expected_p3_sha256", default=EXPECTED_P3_SHA256)
    parser.add_argument("--minimum_valid_gain_pp", type=float, default=0.5)
    parser.add_argument("--minimum_damaged_gain_pp", type=float, default=1.0)
    parser.add_argument(
        "--minimum_positive_event_fraction", type=float, default=0.5
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    p3_hash = file_sha256(args.p3_checkpoint)
    if args.expected_p3_sha256 and p3_hash != args.expected_p3_sha256:
        raise RuntimeError(
            f"P3 SHA256 mismatch: expected={args.expected_p3_sha256}, observed={p3_hash}"
        )
    p3_checkpoint = torch.load(
        args.p3_checkpoint, map_location="cpu", weights_only=False
    )
    p3 = DamageFormer_S8_MICB_DSMIL()
    incompatible = p3.load_state_dict(checkpoint_state(p3_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")

    a2_checkpoint = torch.load(
        args.a2_checkpoint, map_location="cpu", weights_only=False
    )
    if str(a2_checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError(
            f"native expert must be H6-A2, got {a2_checkpoint.get('variant')!r}"
        )
    if a2_checkpoint.get("protocol") not in VALID_A2_PROTOCOLS:
        raise RuntimeError(
            f"unexpected H6-A2 protocol: {a2_checkpoint.get('protocol')!r}"
        )
    a2 = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = a2.load_state_dict(checkpoint_state(a2_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H6-A2 load failed: {incompatible}")

    p3.to(device).eval()
    a2.to(device).eval()
    # Keep the two experts resident on one GPU so every prediction is produced
    # from the exact same batch.  This avoids writing hundreds of gigabytes of
    # intermediate logits and makes sample alignment auditable.
    for model_name, model in (("P3", p3), ("H6-A2", a2)):
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{model_name} parameters: {parameter_count:,}", flush=True)
    for model in (p3, a2):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.val_data_list_path,
        args.crop_size,
        split="val",
        variant="h4",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    histograms = {
        key: torch.zeros(4, 4, dtype=torch.float64)
        for key in ("p3", "a2", "oracle")
    }
    event_histograms: Dict[str, Dict[str, torch.Tensor]] = {}
    event_counts: Dict[str, MutableMapping[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    overall_counts: MutableMapping[str, int] = defaultdict(int)
    per_class_counts: Dict[str, MutableMapping[str, int]] = {
        name: defaultdict(int) for name in CLASS_NAMES
    }
    building_counts: MutableMapping[str, int] = defaultdict(int)
    building_per_class: Dict[str, MutableMapping[str, int]] = {
        name: defaultdict(int) for name in CLASS_NAMES[1:]
    }
    p3_calibration = CalibrationBins(args.calibration_bins)
    a2_calibration = CalibrationBins(args.calibration_bins)
    overall_transition_cube = torch.zeros(4, 4, 4, dtype=torch.int64)
    event_transition_cubes: Dict[str, torch.Tensor] = {}
    processed_images = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb, sar, _location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()

        _, p3_logits = p3(rgb, p3_sar_view(sar))
        a2_outputs = a2(rgb, sar, return_dict=True)
        a2_logits = a2_outputs["native_output_dam"]
        p3_probability = p3_logits.float().softmax(1)
        a2_probability = a2_logits.float().softmax(1)
        p3_confidence, p3_prediction = p3_probability.max(1)
        a2_confidence, a2_prediction = a2_probability.max(1)
        oracle_prediction = correctness_oracle_prediction(
            p3_prediction, a2_prediction, damage, args.ignore_index
        )

        for key, prediction in (
            ("p3", p3_prediction),
            ("a2", a2_prediction),
            ("oracle", oracle_prediction),
        ):
            histograms[key] += fast_histogram(
                prediction.cpu(), damage.cpu(), ignore_index=args.ignore_index
            )

        valid = damage != args.ignore_index
        p3_calibration.update(p3_confidence, p3_prediction == damage, valid)
        a2_calibration.update(a2_confidence, a2_prediction == damage, valid)
        merge_counts(
            overall_counts,
            complementarity_counts(
                p3_prediction, a2_prediction, damage, args.ignore_index
            ),
        )
        overall_transition_cube += expert_transition_cube(
            p3_prediction.cpu(), a2_prediction.cpu(), damage.cpu(), args.ignore_index
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_target = torch.where(
                damage == class_index,
                damage,
                damage.new_full(damage.shape, args.ignore_index),
            )
            merge_counts(
                per_class_counts[class_name],
                complementarity_counts(
                    p3_prediction, a2_prediction, class_target, args.ignore_index
                ),
            )

        p3_cpu = p3_prediction.cpu()
        a2_cpu = a2_prediction.cpu()
        oracle_cpu = oracle_prediction.cpu()
        damage_cpu = damage.cpu()
        for sample_index, name in enumerate(names):
            event = event_name(str(name))
            if event not in event_histograms:
                event_histograms[event] = {
                    key: torch.zeros(4, 4, dtype=torch.float64)
                    for key in ("p3", "a2", "oracle")
                }
                event_transition_cubes[event] = torch.zeros(
                    4, 4, 4, dtype=torch.int64
                )
            for key, prediction in (
                ("p3", p3_cpu),
                ("a2", a2_cpu),
                ("oracle", oracle_cpu),
            ):
                event_histograms[event][key] += fast_histogram(
                    prediction[sample_index],
                    damage_cpu[sample_index],
                    ignore_index=args.ignore_index,
                )
            merge_counts(
                event_counts[event],
                complementarity_counts(
                    p3_cpu[sample_index],
                    a2_cpu[sample_index],
                    damage_cpu[sample_index],
                    args.ignore_index,
                ),
            )
            event_transition_cubes[event] += expert_transition_cube(
                p3_cpu[sample_index],
                a2_cpu[sample_index],
                damage_cpu[sample_index],
                args.ignore_index,
            )
            update_building_audit(
                building_counts,
                building_per_class,
                p3_cpu[sample_index].numpy(),
                a2_cpu[sample_index].numpy(),
                damage_cpu[sample_index].numpy(),
                args.minimum_building_area,
            )
        processed_images += int(rgb.shape[0])
        if processed_images % 50 == 0:
            print(f"oracle audit: {processed_images}/{len(dataset)}", flush=True)

    metrics = {key: metrics_from_histogram(value) for key, value in histograms.items()}
    events = event_rows(event_histograms, event_counts, event_transition_cubes)
    decision = make_decision(
        metrics["p3"],
        metrics["oracle"],
        events,
        args.minimum_valid_gain_pp,
        args.minimum_damaged_gain_pp,
        args.minimum_positive_event_fraction,
    )
    payload = {
        "protocol": "H7_P3_A2_validation_oracle_v1",
        "split": "val",
        "test_accessed": False,
        "oracle_is_deployable": False,
        "processed_images": processed_images,
        "checkpoints": {
            "p3": {
                "path": str(Path(args.p3_checkpoint).resolve()),
                "sha256": p3_hash,
                "experiment": p3_checkpoint.get("experiment"),
                "step": int(p3_checkpoint.get("step", 0)),
                "strict_load": True,
            },
            "a2": {
                "path": str(Path(args.a2_checkpoint).resolve()),
                "sha256": file_sha256(args.a2_checkpoint),
                "protocol": a2_checkpoint.get("protocol"),
                "variant": a2_checkpoint.get("variant"),
                "step": int(a2_checkpoint.get("step", 0)),
                "strict_load": True,
            },
        },
        "data_contract": {
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "validation_list": str(Path(args.val_data_list_path).resolve()),
            "sar_stats_path": str(Path(args.sar_stats_path).resolve()),
            "p3_view": "unclipped_display_repeat3_then_imagenet_normalization",
            "a2_view": "train_stats_clipped_native_single_channel",
            "same_geometry": True,
        },
        "pixel_metrics": metrics,
        "pixel_confusion": {
            key: value.to(torch.int64).tolist() for key, value in histograms.items()
        },
        "pixel_complementarity": {
            "counts": dict(overall_counts),
            "rates": rates(overall_counts),
            "per_ground_truth_class": {
                key: {"counts": dict(value), "rates": rates(value)}
                for key, value in per_class_counts.items()
            },
        },
        "expert_transition_analysis": {
            "definition": "counts indexed by [ground_truth, P3_prediction, A2_prediction]",
            "cube": overall_transition_cube.tolist(),
            "records": transition_records(overall_transition_cube),
        },
        "damaged_proposal_route_audit": proposal_route_audit(
            overall_transition_cube, a2_proposal_class=2
        ),
        "pseudo_building_complementarity": {
            "definition": (
                "4-connected components extracted separately from each GT damage "
                "class; same-class touching buildings may remain merged"
            ),
            "minimum_area": args.minimum_building_area,
            "counts": dict(building_counts),
            "rates": rates(building_counts),
            "per_ground_truth_class": {
                key: {"counts": dict(value), "rates": rates(value)}
                for key, value in building_per_class.items()
            },
        },
        "calibration": {
            "p3": p3_calibration.result(),
            "a2": a2_calibration.result(),
        },
        "events": events,
        "decision": decision,
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    event_output = Path(args.output_event_tsv) if args.output_event_tsv else output.with_suffix(".events.tsv")
    write_event_tsv(event_output, events)

    print("=" * 78)
    print("H7 validation-only P3/A2 complementarity audit")
    print(f"Processed images : {processed_images}")
    for key in ("p3", "a2", "oracle"):
        value = metrics[key]
        print(
            f"{key.upper():<7} Valid={100*value['valid_miou']:.4f} | "
            f"mIoU4={100*value['miou4']:.4f} | "
            f"Dmg={100*value['damaged_iou']:.4f} | "
            f"Des={100*value['destroyed_iou']:.4f}"
        )
    print(
        "Pixel correction/harm: "
        f"A2-only={overall_counts.get('a2_only_correct', 0):,} | "
        f"P3-only={overall_counts.get('p3_only_correct', 0):,}"
    )
    print(
        "Building correction/harm: "
        f"A2-only={building_counts.get('a2_only_correct', 0):,} | "
        f"P3-only={building_counts.get('p3_only_correct', 0):,}"
    )
    observed = decision["observed"]
    assert isinstance(observed, Mapping)
    print(
        f"Oracle gains: Valid={float(observed['oracle_valid_gain_pp']):+.4f} pp | "
        f"Damaged={float(observed['oracle_damaged_gain_pp']):+.4f} pp | "
        f"positive events={int(observed['positive_event_count'])}/"
        f"{int(observed['event_count'])}"
    )
    route_audit = payload["damaged_proposal_route_audit"]
    assert isinstance(route_audit, Mapping)
    print(
        "A2-damaged proposal audit: "
        f"A2-correct={int(route_audit['a2_correct']):,} | "
        f"P3-correct={int(route_audit['p3_correct']):,} | "
        f"net={int(route_audit['net_a2_correct']):+,} | "
        f"decision={route_audit['recommendation']}"
    )
    print(f"Oracle feasibility: {decision['recommendation']}")
    print(f"Damaged-only route: {route_audit['recommendation']}")
    print(
        "Final routing decision: "
        + (
            "TRAIN_DAMAGED_COUNTERFACTUAL_ROUTER"
            if bool(route_audit["passed"])
            else "REJECT_DAMAGED_ONLY_ROUTE_AND_ANALYZE_PAIRWISE_TRANSITIONS"
        )
    )
    print(f"Saved JSON: {output}")
    print(f"Saved event TSV: {event_output}")
    print("Test accessed: False")
    print("=" * 78)


if __name__ == "__main__":
    main()
