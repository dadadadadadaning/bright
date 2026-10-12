# -*- coding: utf-8 -*-
"""Validation-only E0 audit for RGB--SAR local correspondence.

The audit deliberately precedes any CCU training. It reads frozen H6-A2
stage-3/4 common features and computes a parameter-free local cosine volume.
Only peak, top1-top2 margin, normalized entropy and hard peak offset are
reported. There is no cycle module, learnable projection, loss, checkpoint
write or test-set option.

Ground truth is used only after matching to stratify diagnostics. A feature
cell is retained only when at least ``min_class_purity`` of its image-space
support belongs to one building-damage class, avoiding mixed boundaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
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
    build_h6_sdcr_damageformer,
)


CLASS_NAMES = {1: "intact", 2: "damaged", 3: "destroyed"}
STAGE_NAMES = ("stage3", "stage4")
METRIC_NAMES = (
    "correlation_peak",
    "correlation_margin",
    "correlation_entropy",
    "offset_magnitude",
)
# Every AUROC score is oriented so larger means more likely affected.
AFFECTED_SCORE_NAMES = (
    "one_minus_peak",
    "one_minus_margin",
    "correlation_entropy",
    "offset_magnitude",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_name(sample_name: object) -> str:
    return re.sub(r"_\d+$", "", str(sample_name))


def disaster_type(name: str) -> str:
    value = event_name(name)
    return value.rsplit("-", 1)[-1].lower() if "-" in value else "unknown"


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        finite = values.detach().float()
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return
        self.count += int(finite.numel())
        self.total += float(finite.sum().item())
        self.total_square += float(finite.square().sum().item())

    def summary(self) -> Dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "mean": 0.0, "std": 0.0}
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {"count": self.count, "mean": mean, "std": math.sqrt(variance)}


@dataclass
class BinaryScoreBuffer:
    scores: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)

    def update(self, score: torch.Tensor, label: torch.Tensor) -> None:
        finite = torch.isfinite(score)
        if not finite.any():
            return
        self.scores.append(
            score[finite].detach().float().cpu().numpy().astype(np.float32)
        )
        self.labels.append(label[finite].detach().to(torch.uint8).cpu().numpy())

    def summary(self) -> Dict[str, float | int | None]:
        if not self.scores:
            return {
                "count": 0,
                "negative_count": 0,
                "positive_count": 0,
                "auroc": None,
            }
        score = np.concatenate(self.scores)
        label = np.concatenate(self.labels).astype(np.uint8, copy=False)
        negatives = int((label == 0).sum())
        positives = int((label == 1).sum())
        return {
            "count": int(label.size),
            "negative_count": negatives,
            "positive_count": positives,
            "auroc": binary_auroc(score, label),
        }


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Exact binary AUROC with average ranks for tied scores."""

    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if score.shape != label.shape:
        raise ValueError("AUROC scores and labels must have identical shapes")
    finite = np.isfinite(score)
    score, label = score[finite], label[finite]
    positive_count = int((label == 1).sum())
    negative_count = int((label == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(sorted_score.size, dtype=np.float64)
    start = 0
    while start < sorted_score.size:
        end = start + 1
        while end < sorted_score.size and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    positive_rank_sum = float(ranks[inverse][label == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def parameter_free_local_correspondence(
    rgb: torch.Tensor,
    sar: torch.Tensor,
    *,
    window_size: int = 5,
    temperature: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Compute the four E0 statistics without a trainable parameter."""

    if rgb.shape != sar.shape or rgb.ndim != 4:
        raise ValueError("E0 matching requires equal [B,C,H,W] RGB/SAR tensors")
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("window_size must be an odd integer >= 3")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch, channels, height, width = rgb.shape
    locations = height * width
    neighbours = window_size * window_size
    rgb_unit = F.normalize(rgb.float(), dim=1, eps=1e-6)
    sar_unit = F.normalize(sar.float(), dim=1, eps=1e-6)
    sar_patches = F.unfold(
        sar_unit,
        kernel_size=window_size,
        padding=window_size // 2,
    ).reshape(batch, channels, neighbours, locations)
    scores = torch.einsum(
        "bcl,bckl->bkl",
        rgb_unit.reshape(batch, channels, locations),
        sar_patches,
    ).clamp(-1.0, 1.0)

    top = scores.topk(2, dim=1)
    peak = top.values[:, 0:1]
    margin = (top.values[:, 0:1] - top.values[:, 1:2]).clamp(0.0, 2.0)
    probability = (scores / temperature).softmax(dim=1)
    entropy = -(
        probability.clamp_min(1e-8).log() * probability
    ).sum(dim=1, keepdim=True) / math.log(neighbours)

    radius = window_size // 2
    offsets = torch.tensor(
        [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ],
        dtype=torch.float32,
        device=rgb.device,
    )
    best_offset = offsets[top.indices[:, 0]]
    offset_magnitude = torch.linalg.vector_norm(best_offset, dim=-1).unsqueeze(1)

    def spatial(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, 1, height, width)

    peak_spatial = spatial(peak)
    margin_spatial = spatial(margin)
    entropy_spatial = spatial(entropy).clamp(0.0, 1.0)
    offset_spatial = spatial(offset_magnitude)
    return {
        "correlation_peak": peak_spatial,
        "correlation_margin": margin_spatial,
        "correlation_entropy": entropy_spatial,
        "offset_magnitude": offset_spatial,
        "one_minus_peak": 1.0 - 0.5 * (peak_spatial + 1.0),
        "one_minus_margin": 1.0 - 0.5 * margin_spatial,
    }


def pure_stage_labels(
    damage: torch.Tensor,
    output_size: Sequence[int],
    min_class_purity: float,
) -> torch.Tensor:
    """Assign a stage cell only when one building class dominates its support."""

    if not 0.5 <= min_class_purity <= 1.0:
        raise ValueError("min_class_purity must be in [0.5, 1.0]")
    occupancy = torch.stack(
        [(damage == class_index).float() for class_index in CLASS_NAMES], dim=1
    )
    occupancy = F.adaptive_avg_pool2d(occupancy, tuple(output_size))
    purity, class_offset = occupancy.max(dim=1)
    labels = class_offset + 1
    return torch.where(
        purity >= min_class_purity,
        labels,
        torch.full_like(labels, 255),
    )


def cohen_d(first: Mapping[str, float | int], second: Mapping[str, float | int]) -> float:
    first_count = int(first["count"])
    second_count = int(second["count"])
    if first_count < 2 or second_count < 2:
        return 0.0
    pooled = math.sqrt(
        (
            (first_count - 1) * float(first["std"]) ** 2
            + (second_count - 1) * float(second["std"]) ** 2
        )
        / max(first_count + second_count - 2, 1)
    )
    return 0.0 if pooled <= 1e-12 else (
        float(second["mean"]) - float(first["mean"])
    ) / pooled


def _moment_schema(source, scopes: Sequence[str]) -> Dict[str, object]:
    return {
        scope: {
            stage: {
                class_name: {
                    metric: source[scope][stage][class_name][metric].summary()
                    for metric in METRIC_NAMES
                }
                for class_name in CLASS_NAMES.values()
            }
            for stage in STAGE_NAMES
        }
        for scope in scopes
    }


def _auroc_schema(source, scopes: Sequence[str]) -> Dict[str, object]:
    return {
        scope: {
            stage: {
                signal: source[scope][stage][signal].summary()
                for signal in AFFECTED_SCORE_NAMES
            }
            for stage in STAGE_NAMES
        }
        for scope in scopes
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("E0 parameter-free local correspondence audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--min_class_purity", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError("E0 audit requires the locked H6-A2 checkpoint")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("H6-A2 checkpoint has no model state dictionary")
    model = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H6-A2 load failed: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split="val",
        variant="h3",
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

    moment_factory = lambda: defaultdict(  # noqa: E731
        lambda: defaultdict(lambda: defaultdict(RunningMoments))
    )
    score_factory = lambda: defaultdict(  # noqa: E731
        lambda: defaultdict(BinaryScoreBuffer)
    )
    global_moments = moment_factory()
    event_moments = defaultdict(moment_factory)
    disaster_moments = defaultdict(moment_factory)
    global_scores = score_factory()
    event_scores = defaultdict(score_factory)
    disaster_scores = defaultdict(score_factory)

    processed = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb, sar, _, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)[:, :1]
        damage = damage.to(device, non_blocking=True).long()
        rgb_stages = model.rgb_encoder(rgb)
        sar_stages = model.sar_encoder(sar)

        for stage_index in (3, 4):
            stage = f"stage{stage_index}"
            module = model.joint_stages[stage_index - 1]
            if not hasattr(module, "rgb_common") or not hasattr(module, "sar_common"):
                raise RuntimeError(f"H6-A2 {stage} has no common projections")
            rgb_common = module.rgb_common(rgb_stages[stage_index - 1])
            sar_common = module.sar_common(sar_stages[stage_index - 1])
            diagnostics = parameter_free_local_correspondence(
                rgb_common,
                sar_common,
                window_size=args.window_size,
                temperature=args.temperature,
            )
            labels = pure_stage_labels(
                damage,
                rgb_common.shape[-2:],
                args.min_class_purity,
            )
            radius = args.window_size // 2
            valid_geometry = torch.ones_like(labels, dtype=torch.bool)
            valid_geometry[:, :radius] = False
            valid_geometry[:, -radius:] = False
            valid_geometry[:, :, :radius] = False
            valid_geometry[:, :, -radius:] = False

            for sample_index, raw_name in enumerate(names):
                event = event_name(raw_name)
                disaster = disaster_type(event)
                building_mask = (
                    (labels[sample_index] >= 1)
                    & (labels[sample_index] <= 3)
                    & valid_geometry[sample_index]
                )
                affected = (labels[sample_index] >= 2).to(torch.uint8)
                for metric in METRIC_NAMES:
                    values = diagnostics[metric][sample_index, 0]
                    for class_index, class_name in CLASS_NAMES.items():
                        mask = (
                            (labels[sample_index] == class_index)
                            & valid_geometry[sample_index]
                        )
                        if not mask.any():
                            continue
                        selected = values[mask]
                        global_moments[stage][class_name][metric].update(selected)
                        event_moments[event][stage][class_name][metric].update(selected)
                        disaster_moments[disaster][stage][class_name][metric].update(
                            selected
                        )
                for signal in AFFECTED_SCORE_NAMES:
                    score = diagnostics[signal][sample_index, 0][building_mask]
                    binary = affected[building_mask]
                    global_scores[stage][signal].update(score, binary)
                    event_scores[event][stage][signal].update(score, binary)
                    disaster_scores[disaster][stage][signal].update(score, binary)

        processed += int(rgb.shape[0])
        if processed % 50 == 0:
            print(f"E0 correspondence audit: {processed}/{len(dataset)}", flush=True)

    events = sorted(event_moments)
    disasters = sorted(disaster_moments)
    global_statistics = _moment_schema(
        {"overall": global_moments}, ["overall"]
    )["overall"]
    event_statistics = _moment_schema(event_moments, events)
    disaster_statistics = _moment_schema(disaster_moments, disasters)
    global_auroc = _auroc_schema({"overall": global_scores}, ["overall"])[
        "overall"
    ]
    event_auroc = _auroc_schema(event_scores, events)
    disaster_auroc = _auroc_schema(disaster_scores, disasters)

    separability: Dict[str, Dict[str, object]] = {}
    for stage in STAGE_NAMES:
        separability[stage] = {}
        for metric in METRIC_NAMES:
            intact = global_statistics[stage]["intact"][metric]
            damaged = global_statistics[stage]["damaged"][metric]
            destroyed = global_statistics[stage]["destroyed"][metric]
            separability[stage][metric] = {
                "cohen_d_intact_to_damaged": cohen_d(intact, damaged),
                "cohen_d_intact_to_destroyed": cohen_d(intact, destroyed),
            }

    macro_event_auroc: Dict[str, Dict[str, object]] = {}
    candidates = []
    for stage in STAGE_NAMES:
        macro_event_auroc[stage] = {}
        for signal in AFFECTED_SCORE_NAMES:
            values = [
                float(event_auroc[event][stage][signal]["auroc"])
                for event in events
                if event_auroc[event][stage][signal]["auroc"] is not None
            ]
            consistent = sum(value > 0.5 for value in values)
            macro = float(np.mean(values)) if values else None
            overall = global_auroc[stage][signal]["auroc"]
            macro_event_auroc[stage][signal] = {
                "macro_auroc": macro,
                "consistent_events": consistent,
                "comparable_events": len(values),
            }
            candidates.append(
                (
                    float(overall) if overall is not None else 0.0,
                    macro if macro is not None else 0.0,
                    consistent,
                    len(values),
                    stage,
                    signal,
                )
            )

    best = max(candidates, default=(0.0, 0.0, 0, 0, "", ""))
    proceed = best[0] >= 0.60 and best[2] >= 10 and best[3] >= 10
    decision = (
        "PROCEED_TO_CONTROLLED_E1_E3"
        if proceed
        else "STOP_OR_REDESIGN_LOCAL_CORRELATION"
    )

    payload = {
        "protocol": "CCU_BDA_E0_parameter_free_local_correspondence_v2",
        "split": "val",
        "test_accessed": False,
        "training_performed": False,
        "ground_truth_role": "post_matching_diagnostic_stratification_only",
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
            "protocol": checkpoint.get("protocol"),
            "variant": checkpoint.get("variant"),
            "step": int(checkpoint.get("step", 0)),
            "strict_load": True,
        },
        "matching": {
            "source_features": "locked H6-A2 stage3/4 common projections",
            "trainable_parameters": 0,
            "window_size": args.window_size,
            "temperature": args.temperature,
            "soft_matching_used": False,
            "cycle_used": False,
            "min_class_purity": args.min_class_purity,
            "incomplete_border_neighbourhoods_excluded": True,
        },
        "processed_images": processed,
        "event_count": len(events),
        "disaster_type_count": len(disasters),
        "global_class_statistics": global_statistics,
        "event_class_statistics": event_statistics,
        "disaster_type_class_statistics": disaster_statistics,
        "separability": separability,
        "global_affected_auroc": global_auroc,
        "event_affected_auroc": event_auroc,
        "disaster_type_affected_auroc": disaster_auroc,
        "macro_event_affected_auroc": macro_event_auroc,
        "best_signal": {
            "global_auroc": best[0],
            "macro_event_auroc": best[1],
            "consistent_events": best[2],
            "comparable_events": best[3],
            "stage": best[4],
            "signal": best[5],
        },
        "decision": decision,
        "decision_rule": {
            "minimum_global_auroc": 0.60,
            "minimum_consistent_events": 10,
            "minimum_comparable_events": 10,
        },
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_tsv = output.with_suffix(".events.tsv")
    disaster_tsv = output.with_suffix(".disasters.tsv")
    class_tsv = output.with_suffix(".class_statistics.tsv")
    payload["event_tsv"] = str(event_tsv)
    payload["disaster_tsv"] = str(disaster_tsv)
    payload["class_statistics_tsv"] = str(class_tsv)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    columns = [
        "scope",
        "stage",
        "signal",
        "auroc",
        "intact_count",
        "affected_count",
    ]
    for path, scopes, values in (
        (event_tsv, events, event_auroc),
        (disaster_tsv, disasters, disaster_auroc),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(columns)
            for scope in scopes:
                for stage in STAGE_NAMES:
                    for signal in AFFECTED_SCORE_NAMES:
                        result = values[scope][stage][signal]
                        writer.writerow(
                            [
                                scope,
                                stage,
                                signal,
                                result["auroc"],
                                result["negative_count"],
                                result["positive_count"],
                            ]
                        )

    with class_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["level", "scope", "stage", "class", "metric", "count", "mean", "std"]
        )
        sources = (
            ("overall", ["overall"], {"overall": global_statistics}),
            ("event", events, event_statistics),
            ("disaster_type", disasters, disaster_statistics),
        )
        for level, scopes, values in sources:
            for scope in scopes:
                for stage in STAGE_NAMES:
                    for class_name in CLASS_NAMES.values():
                        for metric in METRIC_NAMES:
                            result = values[scope][stage][class_name][metric]
                            writer.writerow(
                                [
                                    level,
                                    scope,
                                    stage,
                                    class_name,
                                    metric,
                                    result["count"],
                                    result["mean"],
                                    result["std"],
                                ]
                            )

    print("=" * 96)
    print("E0 Parameter-Free Building Local Correspondence Audit")
    print(f"Images/events/types : {processed}/{len(events)}/{len(disasters)}")
    print("Training/test       : NONE / NEVER")
    print(
        "Best signal         : "
        f"{best[4]} {best[5]} | global={best[0]:.4f} | "
        f"event-macro={best[1]:.4f} | events={best[2]}/{best[3]}"
    )
    for stage in STAGE_NAMES:
        means = global_statistics[stage]
        print(
            f"{stage} class means     : "
            f"Peak I/D/X="
            f"{means['intact']['correlation_peak']['mean']:.4f}/"
            f"{means['damaged']['correlation_peak']['mean']:.4f}/"
            f"{means['destroyed']['correlation_peak']['mean']:.4f} | "
            f"Entropy I/D/X="
            f"{means['intact']['correlation_entropy']['mean']:.4f}/"
            f"{means['damaged']['correlation_entropy']['mean']:.4f}/"
            f"{means['destroyed']['correlation_entropy']['mean']:.4f}"
        )
        for signal in AFFECTED_SCORE_NAMES:
            overall = global_auroc[stage][signal]["auroc"]
            macro = macro_event_auroc[stage][signal]
            overall_text = "NA" if overall is None else f"{float(overall):.4f}"
            macro_value = macro["macro_auroc"]
            macro_text = (
                "NA" if macro_value is None else f"{float(macro_value):.4f}"
            )
            print(
                f"{stage:<6} {signal:<24} global={overall_text} "
                f"event-macro={macro_text} "
                f"events={macro['consistent_events']}/{macro['comparable_events']}"
            )
    print("Decision            :", decision)
    print("Saved               :", output)
    print("Event TSV           :", event_tsv)
    print("Disaster TSV        :", disaster_tsv)
    print("Class statistics TSV:", class_tsv)
    print("Test accessed       : False")


if __name__ == "__main__":
    main()
