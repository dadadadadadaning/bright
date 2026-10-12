#!/usr/bin/env python3
"""Frozen, model-free probes for what each H6 native-SAR stage represents.

The probe uses train-only cosine centroids, not a learned neural head.  It
answers whether frozen SAR stages separate Intact/Affected, Damaged/Destroyed,
or merely identify the acquisition event.  It never changes model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

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
    VALID_H6_VARIANTS,
    build_h6_sdcr_damageformer,
)


EVENT_RE = re.compile(r"_\d+$")
VALID_PROTOCOLS = ("H6_SDCR_best_weights_v1", "H6_SDCR_resume_v1")


def event_name(sample: str) -> str:
    return EVENT_RE.sub("", str(sample))


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def binary_auprc(labels: list[int], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positive = int(y.sum())
    if positive == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positive)


def pure_building_cells(
    feature: torch.Tensor,
    damage: torch.Tensor,
    minimum_purity: float,
    maximum_per_class: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    safe = damage.clamp(0, 3)
    valid = (damage != 255).float().unsqueeze(1)
    one_hot = F.one_hot(safe, num_classes=4).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid
    size = feature.shape[-2:]
    mass = F.interpolate(valid, size=size, mode="area")
    fractions = F.interpolate(one_hot, size=size, mode="area") / mass.clamp_min(1e-6)
    purity, labels = fractions.max(1)
    records: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for batch_index in range(feature.shape[0]):
        flat_feature = feature[batch_index].permute(1, 2, 0).reshape(-1, feature.shape[1])
        flat_label = labels[batch_index].reshape(-1)
        flat_purity = purity[batch_index].reshape(-1)
        flat_mass = mass[batch_index, 0].reshape(-1)
        for class_index in (1, 2, 3):
            eligible = torch.nonzero(
                (flat_label == class_index)
                & (flat_purity >= minimum_purity)
                & (flat_mass >= 0.50),
                as_tuple=False,
            ).flatten()
            if eligible.numel() == 0:
                continue
            if eligible.numel() > maximum_per_class:
                chosen_purity = flat_purity[eligible]
                eligible = eligible[
                    torch.topk(chosen_purity, maximum_per_class, sorted=False).indices
                ]
            records.append(
                (
                    flat_feature[eligible].detach().float().cpu(),
                    flat_label[eligible].detach().cpu(),
                    batch_index,
                )
            )
    return records


def normalized(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-6)


@torch.inference_mode()
def build_centroids(model, loader, device, minimum_purity, maximum_per_class, max_images):
    stage_class_sum = [defaultdict(lambda: None) for _ in range(4)]
    stage_class_count = [defaultdict(int) for _ in range(4)]
    stage_event_sum = [defaultdict(lambda: None) for _ in range(4)]
    stage_event_count = [defaultdict(int) for _ in range(4)]
    processed = 0
    for sar_batch in loader:
        _, sar, _, damage, names = sar_batch
        sar = sar[:, :1].to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        stages = model.sar_encoder(sar)
        for stage_index, feature in enumerate(stages):
            records = pure_building_cells(
                feature, damage, minimum_purity, maximum_per_class
            )
            for vectors, labels, batch_index in records:
                event = event_name(names[batch_index])
                for class_index in (1, 2, 3):
                    selected = vectors[labels == class_index]
                    if selected.numel() == 0:
                        continue
                    value = selected.double().sum(0)
                    old = stage_class_sum[stage_index][class_index]
                    stage_class_sum[stage_index][class_index] = value if old is None else old + value
                    stage_class_count[stage_index][class_index] += selected.shape[0]
                    event_old = stage_event_sum[stage_index][event]
                    stage_event_sum[stage_index][event] = value if event_old is None else event_old + value
                    stage_event_count[stage_index][event] += selected.shape[0]
        processed += len(names)
        if processed % 100 == 0:
            print(f"probe train extraction: {processed}", flush=True)
        if max_images and processed >= max_images:
            break

    centroids = []
    for stage_index in range(4):
        classes = {
            key: normalized(value / stage_class_count[stage_index][key])
            for key, value in stage_class_sum[stage_index].items()
            if value is not None
        }
        events = {
            key: normalized(value / stage_event_count[stage_index][key])
            for key, value in stage_event_sum[stage_index].items()
            if value is not None
        }
        centroids.append({"classes": classes, "events": events})
    return centroids, processed


@torch.inference_mode()
def evaluate_centroids(
    model, loader, centroids, device, minimum_purity, maximum_per_class, max_images
):
    records = [
        {
            "affected_labels": [], "affected_scores": [],
            "severity_labels": [], "severity_scores": [],
            "damage_correct": 0, "damage_total": 0,
            "event_correct": defaultdict(int), "event_total": defaultdict(int),
        }
        for _ in range(4)
    ]
    processed = 0
    for batch in loader:
        _, sar, _, damage, names = batch
        sar = sar[:, :1].to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        stages = model.sar_encoder(sar)
        for stage_index, feature in enumerate(stages):
            class_centroids = centroids[stage_index]["classes"]
            event_centroids = centroids[stage_index]["events"]
            if not all(key in class_centroids for key in (1, 2, 3)):
                continue
            class_matrix = torch.stack([class_centroids[key] for key in (1, 2, 3)])
            event_names = sorted(event_centroids)
            event_matrix = torch.stack([event_centroids[key] for key in event_names])
            for vectors, labels, batch_index in pure_building_cells(
                feature, damage, minimum_purity, maximum_per_class
            ):
                vectors = normalized(vectors)
                class_scores = vectors @ class_matrix.T
                prediction = class_scores.argmax(1) + 1
                item = records[stage_index]
                item["damage_correct"] += int((prediction == labels).sum())
                item["damage_total"] += int(labels.numel())

                affected_score = torch.logsumexp(class_scores[:, 1:], dim=1) - class_scores[:, 0]
                item["affected_labels"].extend((labels >= 2).int().tolist())
                item["affected_scores"].extend(affected_score.tolist())
                affected = labels >= 2
                item["severity_labels"].extend((labels[affected] == 3).int().tolist())
                item["severity_scores"].extend(
                    (class_scores[affected, 2] - class_scores[affected, 1]).tolist()
                )

                event = event_name(names[batch_index])
                # Convert predicted centroid rows back to event names without storing strings per cell.
                predicted_indices = (vectors @ event_matrix.T).argmax(1).tolist()
                item["event_correct"][event] += sum(
                    event_names[index] == event for index in predicted_indices
                )
                item["event_total"][event] += len(predicted_indices)
        processed += len(names)
        if processed % 100 == 0:
            print(f"probe validation extraction: {processed}", flush=True)
        if max_images and processed >= max_images:
            break

    output = []
    for stage_index, item in enumerate(records, 1):
        per_event_accuracy = {
            event: item["event_correct"][event] / max(total, 1)
            for event, total in item["event_total"].items()
        }
        output.append(
            {
                "stage": stage_index,
                "sampled_building_cells": item["damage_total"],
                "damage_3class_accuracy": item["damage_correct"] / max(item["damage_total"], 1),
                "affected_auroc": binary_auroc(item["affected_labels"], item["affected_scores"]),
                "affected_auprc": binary_auprc(item["affected_labels"], item["affected_scores"]),
                "severity_auroc": binary_auroc(item["severity_labels"], item["severity_scores"]),
                "severity_auprc": binary_auprc(item["severity_labels"], item["severity_scores"]),
                "event_macro_accuracy": (
                    float(np.mean(list(per_event_accuracy.values())))
                    if per_event_accuracy else None
                ),
                "event_accuracy_by_event": per_event_accuracy,
            }
        )
    return output, processed


def main() -> None:
    parser = argparse.ArgumentParser("Audit frozen H6 native-SAR stage representations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--minimum_purity", type=float, default=0.75)
    parser.add_argument("--maximum_cells_per_class_per_image", type=int, default=128)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") not in VALID_PROTOCOLS:
        raise RuntimeError(f"unsupported H6 checkpoint: {checkpoint.get('protocol')!r}")
    variant = str(checkpoint.get("variant", "")).lower()
    if variant not in VALID_H6_VARIANTS:
        raise RuntimeError(f"unsupported H6 variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("checkpoint has no model state")
    model = build_h6_sdcr_damageformer(variant, pretrained_rgb=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    dataset_variant = "h4" if variant in ("b0", "b1", "b2", "b1r") else "h3"
    common = dict(
        dataset_path=args.dataset_path,
        crop_size=args.crop_size,
        split="val",  # deterministic, full-chip extraction for both lists
        variant=dataset_variant,
        sar_stats_path=args.sar_stats_path,
    )
    train_dataset = HeterogeneousRGBSARDataset(
        common["dataset_path"], args.train_list, common["crop_size"],
        split=common["split"], variant=common["variant"],
        sar_stats_path=common["sar_stats_path"],
    )
    val_dataset = HeterogeneousRGBSARDataset(
        common["dataset_path"], args.val_list, common["crop_size"],
        split=common["split"], variant=common["variant"],
        sar_stats_path=common["sar_stats_path"],
    )
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    centroids, train_images = build_centroids(
        model, DataLoader(train_dataset, **loader_args), device,
        args.minimum_purity, args.maximum_cells_per_class_per_image,
        args.max_train_images,
    )
    stage_results, val_images = evaluate_centroids(
        model, DataLoader(val_dataset, **loader_args), centroids, device,
        args.minimum_purity, args.maximum_cells_per_class_per_image,
        args.max_val_images,
    )
    payload = {
        "protocol": "H6_native_SAR_stage_centroid_probe_v1",
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": file_sha256(args.checkpoint),
            "variant": variant,
            "step": int(checkpoint.get("step", 0)),
            "strict_load": True,
        },
        "train_list": str(Path(args.train_list).resolve()),
        "val_list": str(Path(args.val_list).resolve()),
        "train_images": train_images,
        "val_images": val_images,
        "test_accessed": False,
        "probe": "train-only cosine centroids; no model or probe optimization",
        "stages": stage_results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'Stage':>5} {'Cells':>9} {'3C-Acc':>9} {'Aff-AUC':>9} "
        f"{'Sev-AUC':>9} {'Event-Acc':>10}"
    )
    print("-" * 62)
    for item in stage_results:
        def fmt(value):
            return "NA" if value is None else f"{100*value:.3f}"
        print(
            f"{item['stage']:5d} {item['sampled_building_cells']:9d} "
            f"{fmt(item['damage_3class_accuracy']):>9} "
            f"{fmt(item['affected_auroc']):>9} {fmt(item['severity_auroc']):>9} "
            f"{fmt(item['event_macro_accuracy']):>10}"
        )
    print(f"Saved: {output}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
