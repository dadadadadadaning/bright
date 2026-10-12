#!/usr/bin/env python3
"""A0/B0--B2 validation-only frozen feature audit.

Only a LayerNorm-freeze + Linear I/D/X readout is optimised. Oracle semantic
regions are used on train and validation, so this script deliberately has no
test mode and cannot produce a deployable model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import HeterogeneousRGBSARDataset
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import build_h6_sdcr_damageformer
from bda_benchmark.model.p3_reset.p3_feature_audit import FrozenFeatureReadout, pool_semantic_regions
from bda_benchmark.model.p3_reset.sar_sources import H6NativeSARSource, MMOVSARDINOSource, P3PseudoSARSource

VARIANTS = ("a0_rgb", "a0_pseudo", "a0_rgb_pseudo", "b0", "b1", "b2")
PROTOCOL = "P3_RESET_frozen_feature_audit_v1"


def state(payload):
    for key in ("model", "state_dict"):
        if isinstance(payload, Mapping) and isinstance(payload.get(key), Mapping):
            return payload[key]
    return payload


def load_p3(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(state(payload), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_h6(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("variant", "")).lower() != "a2":
        raise RuntimeError("B1 requires a locked H6-A2 checkpoint")
    model = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = model.load_state_dict(state(payload), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H6-A2 load failed: {incompatible}")
    model.to(device).eval()
    return model


def make_source(args, p3, device):
    if args.variant in ("a0_rgb", "a0_pseudo", "a0_rgb_pseudo", "b0"):
        return P3PseudoSARSource(p3).to(device), 256 if args.variant != "a0_rgb_pseudo" else 512
    if args.variant == "b1":
        return H6NativeSARSource(load_h6(args.h6_checkpoint, device)).to(device), 768
    source = MMOVSARDINOSource(args.sar_dino_checkpoint, args.mmov_source_root).to(device)
    return source, 768


@torch.no_grad()
def extract(variant, p3, source, rgb, sar):
    pseudo = P3PseudoSARSource.to_p3_input(sar)
    if variant.startswith("a0_") or variant == "b0":
        _, _, aux = p3(rgb, pseudo, return_aux=True)
        if variant == "a0_rgb":
            return aux["rgb_feature"]
        if variant == "a0_rgb_pseudo":
            return torch.cat((aux["rgb_feature"], aux["sar_feature"]), dim=1)
        return aux["sar_feature"]
    features = source(sar)
    return features["stage4" if variant == "b1" else "block12"]


def confusion_metrics(hist: torch.Tensor) -> Dict[str, object]:
    hist = hist.double()
    tp = hist.diag()
    precision = tp / hist.sum(0).clamp_min(1)
    recall = tp / hist.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(tp.sum() / hist.sum().clamp_min(1)),
        "recall": [float(value) for value in recall],
        "f1": [float(value) for value in f1],
        "confusion": hist.long().tolist(),
    }


def event_name(name: object) -> str:
    return re.sub(r"_\d+$", "", str(name))


@torch.inference_mode()
def evaluate(variant, p3, source, readout, loader, device, min_area, minimum_purity, max_batches=0):
    readout.eval()
    hist = torch.zeros(3, 3, dtype=torch.long)
    event_hist = defaultdict(lambda: torch.zeros(3, 3, dtype=torch.long))
    regions = images = 0
    for batch_index, (rgb, sar, loc, damage, names) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        rgb, sar, loc, damage = (value.to(device, non_blocking=True) for value in (rgb, sar, loc, damage))
        pooled = pool_semantic_regions(extract(variant, p3, source, rgb, sar), loc, damage, min_area, minimum_purity)
        if pooled.labels.numel():
            prediction = readout(pooled.features).argmax(1)
            encoded = 3 * pooled.labels.cpu() + prediction.cpu()
            hist += torch.bincount(encoded, minlength=9).reshape(3, 3)
            for index, sample_index in enumerate(pooled.batch_indices.cpu().tolist()):
                event_hist[event_name(names[sample_index])][pooled.labels[index].cpu(), prediction[index].cpu()] += 1
            regions += int(pooled.labels.numel())
        images += len(names)
    metrics = confusion_metrics(hist)
    per_event = {key: confusion_metrics(value) for key, value in sorted(event_hist.items())}
    metrics.update({
        "regions": regions,
        "processed_images": images,
        "event_macro_f1": float(np.mean([value["macro_f1"] for value in per_event.values()])) if per_event else 0.0,
        "events": per_event,
    })
    return metrics


def dataset(args, split, data_list, max_iters=None):
    return HeterogeneousRGBSARDataset(
        args.dataset_path, data_list, args.crop_size, max_iters=max_iters,
        split=split, variant="h4", sar_stats_path=args.sar_stats_path,
        focus_probability=0.5 if split == "train" else 0.0,
        focus_class_weights=(0.0, 1.0, 4.0, 2.0),
    )


def loader(data, batch, workers, shuffle, seed):
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def atomic_json(value, path):
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--p3_checkpoint", required=True); p.add_argument("--h6_checkpoint", default="")
    p.add_argument("--sar_dino_checkpoint", default=""); p.add_argument("--mmov_source_root", default="")
    p.add_argument("--dataset_path", required=True); p.add_argument("--train_list", required=True); p.add_argument("--val_list", required=True)
    p.add_argument("--sar_stats_path", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=2); p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval_every", type=int, default=500); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4); p.add_argument("--min_area", type=int, default=32)
    p.add_argument("--minimum_purity", type=float, default=0.75)
    p.add_argument("--val_max_batches", type=int, default=0); p.add_argument("--seed", type=int, default=3407)
    return p


def main():
    args = parser().parse_args()
    if args.variant == "b1" and not args.h6_checkpoint:
        raise ValueError("B1 requires --h6_checkpoint")
    if args.variant == "b2" and (not args.sar_dino_checkpoint or not args.mmov_source_root):
        raise ValueError("B2 requires --sar_dino_checkpoint and --mmov_source_root")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    p3 = load_p3(args.p3_checkpoint, device)
    source, channels = make_source(args, p3, device)
    readout = FrozenFeatureReadout(channels).to(device)
    optimiser = torch.optim.AdamW(readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_data = dataset(args, "train", args.train_list, max_iters=max(args.steps * args.batch_size, len(Path(args.train_list).read_text().splitlines())))
    train_loader = loader(train_data, args.batch_size, args.workers, True, args.seed)
    val_loader = loader(dataset(args, "val", args.val_list), args.eval_batch_size, args.workers, False, args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best = None; completed = 0
    while completed < args.steps:
        for rgb, sar, loc, damage, _ in train_loader:
            rgb, sar, loc, damage = (value.to(device, non_blocking=True) for value in (rgb, sar, loc, damage))
            pooled = pool_semantic_regions(extract(args.variant, p3, source, rgb, sar), loc, damage, args.min_area, args.minimum_purity)
            if not pooled.labels.numel():
                continue
            loss = F.cross_entropy(readout(pooled.features), pooled.labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite readout loss at step {completed + 1}")
            optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step(); completed += 1
            if completed % args.eval_every == 0 or completed == args.steps:
                metrics = evaluate(args.variant, p3, source, readout, val_loader, device, args.min_area, args.minimum_purity, args.val_max_batches)
                print(f"Validation | step={completed} | macroF1={100*metrics['macro_f1']:.4f} | DmgRec={100*metrics['recall'][1]:.4f} | EventF1={100*metrics['event_macro_f1']:.4f}", flush=True)
                if best is None or (metrics["event_macro_f1"], metrics["macro_f1"], metrics["recall"][1]) > (best["event_macro_f1"], best["macro_f1"], best["recall"][1]):
                    best = metrics
                    torch.save({"protocol": PROTOCOL, "variant": args.variant, "step": completed, "readout": readout.state_dict(), "metrics": metrics, "test_accessed": False}, output / "best_readout.pth")
                    atomic_json({"protocol": PROTOCOL, "variant": args.variant, "step": completed, "metrics": metrics, "test_accessed": False}, output / "best_validation.json")
            if completed >= args.steps:
                break
    atomic_json({"protocol": PROTOCOL, "variant": args.variant, "completed_step": completed, "best_metrics": best, "selection": "event_macro_f1 -> macro_f1 -> damaged_recall", "oracle_support": True, "test_accessed": False}, output / "training_summary.json")


if __name__ == "__main__":
    main()
