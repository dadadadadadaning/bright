#!/usr/bin/env python3
"""Partition-only audit for Grid, SLICO and learned V3 soft regions."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    load_a1_parent,
    load_v2_checkpoint,
    parse_batch,
    seed_worker,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import read_list

from .model import DamageFormerRGBAnchoredChangeSuperpixelV3


def _event(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


def _grid_ids(building_ids: torch.Tensor, cell: int) -> torch.Tensor:
    b, h, w = building_ids.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=building_ids.device),
        torch.arange(w, device=building_ids.device),
        indexing="ij",
    )
    grid = (yy // cell) * ((w + cell - 1) // cell) + xx // cell
    rows = []
    for bi in range(b):
        pair = building_ids[bi].long() * (int(grid.max()) + 2) + grid.long() + 1
        pair = torch.where(building_ids[bi] > 0, pair, torch.zeros_like(pair))
        _, inverse = torch.unique(pair, sorted=True, return_inverse=True)
        rows.append(inverse)
    return torch.stack(rows)


def _slico_ids(pre: torch.Tensor, building_ids: torch.Tensor, segments: int) -> torch.Tensor:
    from skimage.segmentation import slic

    size = building_ids.shape[-2:]
    if segments <= 0:
        # Match the default 4x4 grid request budget instead of giving SLICO
        # an unfairly larger number of regions.
        segments = max(4, int(np.ceil(size[0] * size[1] / 16.0)))
    images = F.interpolate(pre.float(), size=size, mode="bilinear", align_corners=False)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    rows = []
    for bi, image in enumerate(images):
        lo = image.min((0, 1), keepdims=True)
        hi = image.max((0, 1), keepdims=True)
        image = (image - lo) / np.maximum(hi - lo, 1.0e-6)
        labels = slic(
            image,
            n_segments=int(segments),
            compactness=10.0,
            slic_zero=True,
            start_label=0,
            channel_axis=-1,
            enforce_connectivity=True,
        )
        building = building_ids[bi].detach().cpu().numpy().astype(np.int64)
        pair = labels.astype(np.int64) * (int(building.max()) + 1) + building
        pair[building == 0] = -1
        _, inverse = np.unique(pair, return_inverse=True)
        rows.append(torch.from_numpy(inverse.reshape(size)).to(building_ids.device))
    return torch.stack(rows).long()


def _learned_ids(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    probabilities = aux["association_probabilities"]
    candidate_ids = aux["candidate_ids"]
    winner = probabilities.argmax(1, keepdim=True)
    hard = torch.gather(candidate_ids, 1, winner).squeeze(1)
    return hard.reshape_as(aux["building_ids"])


def _boundary(value: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(value, dtype=torch.bool)
    edge[:, :, 1:] |= value[:, :, 1:] != value[:, :, :-1]
    edge[:, :, :-1] |= value[:, :, 1:] != value[:, :, :-1]
    edge[:, 1:, :] |= value[:, 1:, :] != value[:, :-1, :]
    edge[:, :-1, :] |= value[:, 1:, :] != value[:, :-1, :]
    return edge


def _sample_metrics(ids: torch.Tensor, target: torch.Tensor, building: torch.Tensor) -> Dict[str, float]:
    valid = building & (target >= 1) & (target <= 3) & (target != 255)
    if not bool(valid.any().item()):
        return {"pixel_accuracy": 0.0, "macro_iou": 0.0, "purity": 0.0,
                "undersegmentation_error": 0.0, "boundary_recall": 0.0,
                "mixed_region_fraction": 0.0, "affected_retention": 0.0}
    oracle = torch.zeros_like(target)
    purity_num = 0.0
    mixed = 0
    regions = 0
    for rid in torch.unique(ids[valid]).tolist():
        mask = valid & (ids == int(rid))
        counts = torch.stack([(target[mask] == cls).sum() for cls in (1, 2, 3)])
        majority = int(counts.argmax()) + 1
        oracle[mask] = majority
        purity_num += float(counts.max())
        mixed += int((counts > 0).sum() > 1)
        regions += 1
    cm = torch.zeros(3, 3, dtype=torch.float64, device=target.device)
    truth = target[valid] - 1
    pred = oracle[valid] - 1
    cm += torch.bincount(truth * 3 + pred, minlength=9).reshape(3, 3)
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    occurrence = torch.where(target == 1, 0, 1)
    gt_edge = _boundary(occurrence[None])[0] & valid
    region_edge = _boundary(ids[None])[0] & building
    tolerant = F.max_pool2d(region_edge[None, None].float(), 3, 1, 1)[0, 0] > 0
    boundary_recall = float((tolerant & gt_edge).sum() / gt_edge.sum().clamp_min(1))
    affected = valid & (target >= 2)
    region_affected_fraction = torch.zeros_like(target, dtype=torch.float32)
    for rid in torch.unique(ids[valid]).tolist():
        mask = valid & (ids == int(rid))
        region_affected_fraction[mask] = affected[mask].float().mean()
    retention = float(
        ((region_affected_fraction >= 0.10) & affected).sum() / affected.sum().clamp_min(1)
    )
    accuracy = float(inter.sum() / cm.sum().clamp_min(1))
    purity = purity_num / max(1, int(valid.sum()))
    return {
        "pixel_accuracy": accuracy,
        "macro_iou": float((inter / union.clamp_min(1)).mean()),
        "purity": purity,
        "undersegmentation_error": 1.0 - purity,
        "boundary_recall": boundary_recall,
        "mixed_region_fraction": mixed / max(1, regions),
        "affected_retention": retention,
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser("V3 partition audit")
    p.add_argument("--mode", choices=("grid", "slico", "learned"), required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--variant", default="p5_anchored_change")
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_list", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--grid_cell", type=int, default=4)
    p.add_argument(
        "--segments", type=int, default=0,
        help="SLICO request count; 0 matches the default 4x4-grid region budget.",
    )
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    if args.mode == "learned" and not args.checkpoint:
        raise ValueError("learned mode requires --checkpoint")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    variant, stage = args.variant, "structure"
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        variant = str(payload.get("variant", variant))
        stage = str(payload.get("stage", "change"))
    model = DamageFormerRGBAnchoredChangeSuperpixelV3(variant).to(device)
    parent = load_a1_parent(model, args.a1_ckpt)
    checkpoint = None
    if args.checkpoint:
        checkpoint = load_v2_checkpoint(model, args.checkpoint)
    model.configure_stage(stage)
    model.eval()

    dataset = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), args.crop_size, type="val"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0,
    )
    values: Dict[str, List[float]] = defaultdict(list)
    events: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    images = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        pre, post, _, damage, names = parse_batch(batch, device)
        _, _, aux = model(pre, post, return_aux=True)
        h, w = aux["building_ids"].shape[-2:]
        target = F.interpolate(damage[:, None].float(), (h, w), mode="nearest")[:, 0].long()
        if args.mode == "grid":
            ids = _grid_ids(aux["building_ids"], args.grid_cell)
        elif args.mode == "slico":
            ids = _slico_ids(pre, aux["building_ids"], args.segments)
        else:
            ids = _learned_ids(aux)
        for bi, name in enumerate(names):
            row = _sample_metrics(ids[bi], target[bi], aux["building_ids"][bi] > 0)
            for key, value in row.items():
                values[key].append(value)
                events[_event(str(name))][key].append(value)
        images += int(pre.shape[0])
        if images % 25 < int(pre.shape[0]):
            print(f"partition audit {images}/{len(dataset)}", flush=True)

    result = {
        "protocol": "DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3_PARTITION_AUDIT",
        "mode": args.mode,
        "variant": variant,
        "stage": stage,
        "processed_images": images,
        "metrics": {key: float(np.mean(row)) for key, row in values.items()},
        "event_metrics": {
            event: {key: float(np.mean(row)) for key, row in metric.items()}
            for event, metric in events.items()
        },
        "parent": parent,
        "checkpoint": checkpoint,
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
