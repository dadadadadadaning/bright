#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.model.p3_reset import P3ResetModel
from bda_benchmark.model.p3_reset.heads import ordinal_expected_severity_loss
from bda_benchmark.model.p3_reset.sar_sources import H6NativeSARSource, MMOVSARDINOSource
from bda_benchmark.script.standard_ML.h10_h3d_runtime import evaluate_h10, metrics_line
from bda_benchmark.script.standard_ML.train_P3_reset_feature_audit import load_h6, state

PROTOCOL = "P3_RESET_model_best_v1"


def load_p3(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(state(payload), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    return model.to(device)


def build_model(args, device):
    p3 = load_p3(args.p3_checkpoint, device)
    source = None; key = "fused"; channels = 256
    if args.sar_source == "b1":
        source = H6NativeSARSource(load_h6(args.h6_checkpoint, device)).to(device)
        key, channels = "stage4", 768
    elif args.sar_source == "b2":
        source = MMOVSARDINOSource(args.sar_dino_checkpoint, args.mmov_source_root).to(device)
        key, channels = "block12", 768
    model = P3ResetModel(p3, args.variant, source, key, channels, interaction=args.interaction).to(device)
    if args.variant in ("d1", "d2"):
        if not args.fusion_checkpoint:
            raise RuntimeError("D1/D2 require --fusion_checkpoint from an admitted C1/C2 run")
        payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
        source_state = state(payload)
        fusion_state = {
            name[len("fusion_residual.") :]: value
            for name, value in source_state.items()
            if name.startswith("fusion_residual.")
        }
        if not fusion_state:
            raise RuntimeError("fusion checkpoint contains no fusion_residual parameters")
        incompatible = model.fusion_residual.load_state_dict(fusion_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict locked-fusion load failed: {incompatible}")
        for parameter in model.fusion_residual.parameters():
            parameter.requires_grad_(False)
    return model


def make_loader(args, split, path, batch, shuffle):
    data = HeterogeneousRGBSARDataset(
        args.dataset_path, path, args.crop_size, split=split, variant="h4",
        sar_stats_path=args.sar_stats_path,
        focus_probability=0.5 if split == "train" else 0.0,
        focus_class_weights=(0.0, 1.0, 4.0, 2.0),
    )
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      generator=torch.Generator().manual_seed(args.seed))


def damage_loss(variant, logits, target, ordinal_weight):
    if variant == "c1s":
        # C1S is deliberately blind to Background/Intact supervision.  It can
        # only learn the conditional Damaged-vs-Destroyed boundary.
        valid = (target == 2) | (target == 3)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        severity = logits[:, 2:4].permute(0, 2, 3, 1)[valid]
        return F.cross_entropy(severity, target[valid] - 2)
    if variant in ("d1", "d2"):
        valid = (target >= 1) & (target <= 3)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        conditional = logits[:, 1:4].permute(0, 2, 3, 1)[valid]
        loss = F.cross_entropy(conditional, target[valid] - 1)
        if variant == "d2":
            loss = loss + ordinal_weight * ordinal_expected_severity_loss(logits, target)
        return loss
    return F.cross_entropy(logits, target.long(), ignore_index=255)


def atomic_json(payload, path):
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def gate(metrics, baseline, details, baseline_details, variant):
    deltas = {
        key: 100 * (metrics[key] - baseline[key]) for key in (
            "valid_miou", "damaged_iou", "destroyed_iou", "affected_recall", "damaged_gate_miss",
            "event_mean_official_style_4class_miou", "event_worst_official_style_4class_miou",
        )
    }
    events = {}
    for event, value in details.get("event_metrics", {}).items():
        if event in baseline_details.get("event_metrics", {}):
            events[event] = 100 * (value["miou4"] - baseline_details["event_metrics"][event]["miou4"])
    nonnegative = sum(value >= 0 for value in events.values())
    if variant == "c1s":
        checks = {
            "valid_miou": deltas["valid_miou"] >= 0.20,
            "damaged_iou_nondecreasing": deltas["damaged_iou"] >= 0.0,
            "destroyed_iou": deltas["destroyed_iou"] >= 0.30,
            "affected_recall_invariant": abs(deltas["affected_recall"]) <= 1e-6,
            "damaged_gate_miss_invariant": abs(deltas["damaged_gate_miss"]) <= 1e-6,
            "event_mean": deltas["event_mean_official_style_4class_miou"] >= 0.0,
            "event_worst": deltas["event_worst_official_style_4class_miou"] >= 0.0,
            "nonnegative_events": nonnegative >= 10,
        }
    else:
        checks = {
            "valid_miou": deltas["valid_miou"] >= 0.30,
            "damaged_iou": deltas["damaged_iou"] >= 0.50,
            "affected_recall": deltas["affected_recall"] >= 0.50,
            "damaged_gate_miss": deltas["damaged_gate_miss"] <= -0.50,
            "event_mean": deltas["event_mean_official_style_4class_miou"] >= 0.0,
            "event_worst": deltas["event_worst_official_style_4class_miou"] >= 0.0,
            "nonnegative_events": nonnegative >= 10,
        }
    return {"passed": all(checks.values()), "checks": checks, "observed_pp": deltas,
            "event_miou4_delta_pp": events, "nonnegative_events": nonnegative}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("c0", "c1", "c1s", "c2", "d1", "d2"), required=True)
    p.add_argument("--sar_source", choices=("b0", "b1", "b2"), default="b0")
    p.add_argument("--feature_decision", default="")
    p.add_argument("--interaction", choices=("c1", "c2"), default="c1")
    p.add_argument("--fusion_checkpoint", default="")
    p.add_argument("--p3_checkpoint", required=True); p.add_argument("--h6_checkpoint", default="")
    p.add_argument("--sar_dino_checkpoint", default=""); p.add_argument("--mmov_source_root", default="")
    p.add_argument("--dataset_path", required=True); p.add_argument("--train_list", required=True); p.add_argument("--val_list", required=True)
    p.add_argument("--sar_stats_path", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--train_batch_size", type=int, default=2); p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8); p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval_every", type=int, default=500); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ordinal_weight", type=float, default=0.05); p.add_argument("--val_max_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=3407)
    return p


def main():
    args = parser().parse_args()
    if args.variant != "c0" and args.sar_source in ("b1", "b2"):
        if not args.feature_decision:
            raise RuntimeError("external fusion requires --feature_decision from completed A0/B0-B2")
        decision = json.loads(Path(args.feature_decision).read_text(encoding="utf-8"))
        if not decision.get("passed") or decision.get("winner") != args.sar_source:
            raise RuntimeError(f"feature-source gate did not admit {args.sar_source}: {decision.get('decision')}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); model = build_model(args, device)
    val_loader = make_loader(args, "val", args.val_list, args.eval_batch_size, False)
    if args.variant in ("d1", "d2"):
        baseline_model = build_model(argparse.Namespace(**{**vars(args), "variant": args.interaction, "fusion_checkpoint": ""}), device)
        fusion_payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
        incompatible = baseline_model.load_state_dict(state(fusion_payload), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict fusion baseline load failed: {incompatible}")
    elif args.variant == "c0":
        baseline_model = model
    else:
        baseline_model = P3ResetModel(load_p3(args.p3_checkpoint, device), "c0").to(device)
    baseline_details = {}; baseline = evaluate_h10(baseline_model, val_loader, device, 255, args.val_max_batches, baseline_details)
    print(metrics_line("Initial P3", baseline), flush=True)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if args.variant == "c0":
        torch.save({"protocol": PROTOCOL, "variant": "c0", "step": 0, "model": model.state_dict(), "metrics": baseline, "test_accessed": False}, output / "best_event_worst.pth")
        atomic_json({"protocol": PROTOCOL, "variant": "c0", "completed_step": 0, "best_metrics": baseline, "exact_p3": True, "test_accessed": False}, output / "training_summary.json")
        return
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("P3 reset candidate has no trainable parameters")
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(args, "train", args.train_list, args.train_batch_size, True)
    completed = micro = 0; best_key = None; best_metrics = None
    while completed < args.steps:
        for rgb, sar, _, damage, _ in train_loader:
            rgb, sar, damage = (value.to(device, non_blocking=True) for value in (rgb, sar, damage))
            outputs = model(rgb, sar, return_dict=True)
            loss = damage_loss(args.variant, outputs["output_dam"], damage, args.ordinal_weight) / args.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimiser step {completed}")
            loss.backward(); micro += 1
            if micro % args.accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimiser.step(); optimiser.zero_grad(set_to_none=True); completed += 1
            if completed % args.eval_every == 0 or completed == args.steps:
                details = {}; metrics = evaluate_h10(model, val_loader, device, 255, args.val_max_batches, details)
                print(metrics_line(f"Validation step={completed}", metrics), flush=True)
                key = (metrics["event_worst_official_style_4class_miou"], metrics["event_mean_official_style_4class_miou"], metrics["damaged_iou"], metrics["valid_miou"])
                if best_key is None or key > best_key:
                    best_key, best_metrics = key, metrics
                    torch.save({"protocol": PROTOCOL, "variant": args.variant, "sar_source": args.sar_source, "step": completed, "model": model.state_dict(), "metrics": metrics, "test_accessed": False}, output / "best_event_worst.pth")
                    atomic_json({"metrics": metrics, "details": details, "gate": gate(metrics, baseline, details, baseline_details, args.variant), "test_accessed": False}, output / "best_validation.json")
            if completed >= args.steps:
                break
    report = json.loads((output / "best_validation.json").read_text(encoding="utf-8"))
    atomic_json({"protocol": PROTOCOL, "variant": args.variant, "sar_source": args.sar_source, "completed_step": completed, "best_metrics": best_metrics, "advancement_gate": report["gate"], "selection": "event_worst -> event_mean -> damaged_iou -> valid_miou", "test_accessed": False}, output / "training_summary.json")


if __name__ == "__main__":
    main()
