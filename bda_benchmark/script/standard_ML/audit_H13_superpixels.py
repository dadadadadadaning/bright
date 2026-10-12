# -*- coding: utf-8 -*-
"""Validation-only audit of H13 grid and RGB-guided superpixel partitions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H13_SPER import (  # noqa: E402
    RGBGuidedSoftSuperpixels,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    DamageFormerRGBSARH6SDCR,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


def event_name(name: str) -> str:
    return str(name).rsplit("_", 1)[0]


def load_a2(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError("H13 superpixel audit requires an H6-A2 checkpoint")
    model = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H6-A2 load failed: {incompatible}")
    return model.to(device).eval()


class Accumulator:
    def __init__(self) -> None:
        self.nodes = 0
        self.building_nodes = 0
        self.occurrence_purity_sum = 0.0
        self.severity_purity_sum = 0.0
        self.mixed_occurrence_nodes = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.base_tp = 0
        self.base_fp = 0
        self.base_fn = 0
        self.affected_pixels = 0
        self.intact_pixels = 0
        self.affected_nodes = 0
        self.node_occurrence_counts = []

    def update(
        self,
        labels: torch.Tensor,
        base_prediction: torch.Tensor,
        region_ids: torch.Tensor,
        node_count: int,
    ) -> None:
        labels = labels.reshape(-1)
        base_prediction = base_prediction.reshape(-1)
        region_ids = region_ids.reshape(-1)
        valid = labels != 255
        building = valid & (labels >= 1) & (labels <= 3)
        affected = labels >= 2
        base_affected = base_prediction >= 2
        self.base_tp += int((building & affected & base_affected).sum())
        self.base_fp += int((building & ~affected & base_affected).sum())
        self.base_fn += int((building & affected & ~base_affected).sum())
        self.affected_pixels += int((building & affected).sum())
        self.intact_pixels += int((building & ~affected).sum())
        self.nodes += int(node_count)
        for node_id in range(node_count):
            member = valid & (region_ids == node_id)
            building_member = member & building
            if int(building_member.sum()) < 2:
                continue
            self.building_nodes += 1
            occurrence = affected[building_member].long()
            occurrence_counts = torch.bincount(occurrence, minlength=2)
            intact_count = int(occurrence_counts[0])
            affected_count = int(occurrence_counts[1])
            self.node_occurrence_counts.append((intact_count, affected_count))
            if affected_count > 0:
                self.affected_nodes += 1
            occurrence_purity = float(occurrence_counts.max()) / float(occurrence.numel())
            self.occurrence_purity_sum += occurrence_purity
            if occurrence_purity < 0.90:
                self.mixed_occurrence_nodes += 1
            severity = labels[building_member].long() - 1
            severity_counts = torch.bincount(severity, minlength=3)
            self.severity_purity_sum += float(severity_counts.max()) / float(severity.numel())
            region_prediction = int(occurrence_counts[1] > occurrence_counts[0])
            truth = occurrence.bool()
            if region_prediction:
                self.tp += int(truth.sum())
                self.fp += int((~truth).sum())
            else:
                self.fn += int(truth.sum())

    def summary(self) -> Dict[str, float | int]:
        denominator = max(self.building_nodes, 1)
        oracle_recall = self.tp / max(self.tp + self.fn, 1)
        oracle_iou = self.tp / max(self.tp + self.fp + self.fn, 1)
        base_recall = self.base_tp / max(self.base_tp + self.base_fn, 1)
        base_iou = self.base_tp / max(
            self.base_tp + self.base_fp + self.base_fn, 1
        )
        threshold_records = []
        # A majority vote is deliberately not the only audit.  Weak damage may
        # occupy a small but coherent part of one region, and H13 uses max/std
        # evidence precisely to preserve it.  The sweep measures whether the
        # fixed partition can represent that trade-off without claiming that
        # validation labels are available at inference.
        for threshold in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
            tp = fp = fn = 0
            for intact_count, affected_count in self.node_occurrence_counts:
                fraction = affected_count / max(intact_count + affected_count, 1)
                if fraction >= threshold:
                    tp += affected_count
                    fp += intact_count
                else:
                    fn += affected_count
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            iou = tp / max(tp + fp + fn, 1)
            threshold_records.append(
                {
                    "threshold": threshold,
                    "affected_recall": recall,
                    "affected_precision": precision,
                    "affected_iou": iou,
                }
            )
        best = max(
            threshold_records,
            key=lambda item: (item["affected_iou"], item["affected_recall"]),
        )
        return {
            "nodes": self.nodes,
            "building_nodes": self.building_nodes,
            "affected_nodes": self.affected_nodes,
            "affected_pixels": self.affected_pixels,
            "intact_pixels": self.intact_pixels,
            "occurrence_purity": self.occurrence_purity_sum / denominator,
            "severity_purity": self.severity_purity_sum / denominator,
            "mixed_occurrence_fraction": self.mixed_occurrence_nodes / denominator,
            "region_majority_oracle_affected_recall": oracle_recall,
            "region_majority_oracle_affected_iou": oracle_iou,
            "best_fraction_oracle": best,
            "fraction_oracle_sweep": threshold_records,
            "h6_a2_affected_recall_at_audit_scale": base_recall,
            "h6_a2_affected_iou_at_audit_scale": base_iou,
        }


def decide_partition_route(records: Mapping[str, Mapping[str, object]]) -> str:
    """Make an evidence-aware decision and refuse degenerate smoke samples."""

    rgb = records["rgb"]
    grid = records["grid"]
    rgb_best = rgb["best_fraction_oracle"]
    grid_best = grid["best_fraction_oracle"]
    if not isinstance(rgb_best, Mapping) or not isinstance(grid_best, Mapping):
        raise TypeError("partition audit is missing fraction-oracle records")
    if min(int(rgb["affected_pixels"]), int(grid["affected_pixels"])) < 100:
        return "INSUFFICIENT_AFFECTED_SUPPORT_RUN_FULL_AUDIT"
    if (
        float(rgb_best["affected_iou"])
        < float(rgb["h6_a2_affected_iou_at_audit_scale"]) + 0.02
    ):
        return "STOP_SUPERPIXEL_ROUTE_LOW_REGION_UPPER_BOUND"
    if (
        float(rgb_best["affected_iou"])
        + 0.005
        < float(grid_best["affected_iou"])
    ):
        return "PROCEED_TO_S1_REGION_CONTROL_ONLY"
    return "PROCEED_TO_S1_S2_CONTROLLED_SCREEN"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Audit H13 superpixels")
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
    parser.add_argument("--cell_size", type=int, default=2)
    parser.add_argument("--color_temperature", type=float, default=0.50)
    parser.add_argument("--compactness", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if "val" not in Path(args.data_list_path).name.lower():
        raise RuntimeError("H13 superpixel audit is validation-only")
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_a2(args.checkpoint, device)
    partitioner = RGBGuidedSoftSuperpixels(
        cell_size=args.cell_size,
        color_temperature=args.color_temperature,
        compactness=args.compactness,
    ).to(device)
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
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
    overall = {mode: Accumulator() for mode in ("grid", "rgb")}
    event_accumulators = defaultdict(
        lambda: {mode: Accumulator() for mode in ("grid", "rgb")}
    )
    processed = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb, sar, _location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar[:, :1], return_dict=True)
        stage_size = outputs["stage2_rgb_common"].shape[-2:]
        rgb_low = F.interpolate(rgb.float(), size=stage_size, mode="bilinear", align_corners=False)
        damage_low = F.interpolate(
            damage.unsqueeze(1).float(), size=stage_size, mode="nearest"
        )[:, 0].long()
        prediction_low = F.interpolate(
            outputs["output_dam"].float(), size=stage_size, mode="bilinear", align_corners=False
        ).argmax(1)
        for mode in ("grid", "rgb"):
            assignment = partitioner(rgb_low, mode)
            winner = assignment.weights.argmax(1)
            ids = assignment.indices.unsqueeze(0).expand(rgb.shape[0], -1, -1)
            hard_ids = torch.gather(ids, 1, winner.unsqueeze(1)).squeeze(1)
            hard_ids = hard_ids.reshape(rgb.shape[0], *stage_size)
            for sample in range(rgb.shape[0]):
                overall[mode].update(
                    damage_low[sample],
                    prediction_low[sample],
                    hard_ids[sample],
                    assignment.node_count,
                )
                event_accumulators[event_name(names[sample])][mode].update(
                    damage_low[sample],
                    prediction_low[sample],
                    hard_ids[sample],
                    assignment.node_count,
                )
        processed += int(rgb.shape[0])
        if processed % 50 == 0:
            print(f"H13 superpixel audit: {processed}/{len(dataset)}", flush=True)

    records = {mode: accumulator.summary() for mode, accumulator in overall.items()}
    events = {
        event: {mode: accumulator.summary() for mode, accumulator in modes.items()}
        for event, modes in sorted(event_accumulators.items())
    }
    rgb = records["rgb"]
    grid = records["grid"]
    recommendation = decide_partition_route(records)
    payload = {
        "protocol": "H13_SPER_validation_partition_audit_v1",
        "test_accessed": False,
        "processed_images": processed,
        "parameters": {
            "cell_size": args.cell_size,
            "color_temperature": args.color_temperature,
            "compactness": args.compactness,
        },
        "records": records,
        "events": events,
        "rgb_minus_grid": {
            key: float(rgb[key]) - float(grid[key])
            for key in (
                "occurrence_purity",
                "severity_purity",
                "region_majority_oracle_affected_recall",
                "region_majority_oracle_affected_iou",
                "affected_pixels",
            )
        },
        "recommendation": recommendation,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=" * 88)
    print("H13 validation-only superpixel audit")
    for mode in ("grid", "rgb"):
        item = records[mode]
        print(
            f"{mode:>4} | occurrence purity={100*float(item['occurrence_purity']):.2f} | "
            f"severity purity={100*float(item['severity_purity']):.2f} | "
            f"affected pixels={int(item['affected_pixels'])} | "
            f"majority recall={100*float(item['region_majority_oracle_affected_recall']):.2f} | "
            f"best fraction IoU={100*float(item['best_fraction_oracle']['affected_iou']):.2f} "
            f"@t={float(item['best_fraction_oracle']['threshold']):.2f} | "
            f"mixed={100*float(item['mixed_occurrence_fraction']):.2f}",
            flush=True,
        )
    print(f"Decision: {recommendation}")
    print(f"Saved: {output}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
