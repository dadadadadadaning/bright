#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate object-level voting post-processing for BRIGHT damage maps."""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from full_evaluate_unified import (
    build_arg_parser,
    build_model,
    import_dataset_class,
    json_ready,
    load_checkpoint,
    parse_batch,
    parse_outputs,
    read_list,
    resize_logits,
    summarize_official_conf,
    update_confusion,
    write_confusion_csv,
)


CLASSES = ["background", "intact", "damaged", "destroyed"]


def iter_component(mask: np.ndarray, visited: np.ndarray, y0: int, x0: int) -> List[Tuple[int, int]]:
    h, w = mask.shape
    queue: deque[Tuple[int, int]] = deque([(y0, x0)])
    visited[y0, x0] = True
    coords: List[Tuple[int, int]] = []
    while queue:
        y, x = queue.popleft()
        coords.append((y, x))
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not visited[yy, xx]:
                visited[yy, xx] = True
                queue.append((yy, xx))
    return coords


def connected_components(mask: np.ndarray) -> Iterable[List[Tuple[int, int]]]:
    mask = mask.astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    ys, xs = np.where(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not visited[y, x]:
            yield iter_component(mask, visited, y, x)


def object_vote_damage_map(
    pred_damage: np.ndarray,
    building_mask: np.ndarray,
    min_area: int = 20,
) -> np.ndarray:
    voted = pred_damage.copy()
    for coords in connected_components(building_mask):
        if len(coords) < min_area:
            continue
        yy = np.asarray([p[0] for p in coords], dtype=np.int64)
        xx = np.asarray([p[1] for p in coords], dtype=np.int64)
        labels = pred_damage[yy, xx]
        counts = np.bincount(labels[(labels >= 1) & (labels <= 3)], minlength=4)
        if counts[1:4].sum() <= 0:
            continue
        voted_label = int(np.argmax(counts[1:4]) + 1)
        voted[yy, xx] = voted_label
    return voted


def prefixed_summary(prefix: str, summary: Dict[str, object]) -> Dict[str, object]:
    aliases = {
        "official_4class_miou": "official_style_4class_miou",
        "target_3class_miou": "target_3class_miou",
        "damaged_iou": "damaged_iou",
        "destroyed_iou": "destroyed_iou",
        "damaged_recall": "damaged_recall",
        "destroyed_recall": "destroyed_recall",
        "damaged_to_intact": "damaged_to_intact",
        "destroyed_to_intact": "destroyed_to_intact",
        "loc_iou": "loc_iou",
        "loc_f1": "loc_f1",
    }
    return {f"{prefix}_{alias}": summary.get(key, "") for alias, key in aliases.items()}


def write_comparison_csv(path: str, before: Dict[str, object], after: Dict[str, object]) -> None:
    fields = [
        "metric",
        "before",
        "after",
        "delta",
    ]
    metrics = [
        "official_style_4class_miou",
        "target_3class_miou",
        "damaged_iou",
        "destroyed_iou",
        "damaged_recall",
        "destroyed_recall",
        "damaged_to_intact",
        "destroyed_to_intact",
        "loc_iou",
        "loc_f1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            b_val = before.get(metric, "")
            a_val = after.get(metric, "")
            try:
                delta = float(a_val) - float(b_val)
            except (TypeError, ValueError):
                delta = ""
            writer.writerow({"metric": metric, "before": b_val, "after": a_val, "delta": delta})


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Evaluate object-level voting post-processing."
    parser.add_argument("--use_object_voting", action="store_true")
    parser.add_argument("--object_min_area", type=int, default=20)
    parser.add_argument(
        "--object_voting_source",
        type=str,
        default="gt_building",
        choices=["gt_building", "pred_building"],
    )
    parser.add_argument("--save_voted_maps", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    map_dir = os.path.join(args.save_dir, "voted_maps")
    if args.save_voted_maps:
        os.makedirs(map_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = read_list(args.data_list_path)
    DatasetClass = import_dataset_class()
    dataset = DatasetClass(args.dataset_path, data_list, args.image_size, None, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(args)
    load_checkpoint(model, args.model_path, device)
    model.to(device)
    model.eval()

    loc_conf = np.zeros((2, 2), dtype=np.int64)
    before_building_conf = np.zeros((2, 2), dtype=np.int64)
    after_building_conf = np.zeros((2, 2), dtype=np.int64)
    before_conf = np.zeros((4, 4), dtype=np.int64)
    after_conf = np.zeros((4, 4), dtype=np.int64)
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating object voting")):
            if args.num_batches is not None and batch_idx >= args.num_batches:
                break

            pre, post, loc_label, dam_label, names = parse_batch(batch, device)
            loc_logits, dam_logits = parse_outputs(model(pre, post))
            loc_logits = resize_logits(loc_logits, loc_label)
            dam_logits = resize_logits(dam_logits, dam_label)
            loc_pred = torch.argmax(loc_logits, dim=1)
            dam_pred = torch.argmax(dam_logits, dim=1)

            loc_np = loc_label.detach().cpu().numpy()
            dam_np = dam_label.detach().cpu().numpy()
            loc_pred_np = loc_pred.detach().cpu().numpy()
            dam_pred_np = dam_pred.detach().cpu().numpy()

            valid_mask = dam_np != 255
            loc_gt_bin = np.isin(dam_np, [1, 2, 3]).astype(np.int64)
            loc_pr_bin = (loc_pred_np > 0).astype(np.int64)
            update_confusion(
                loc_conf,
                loc_gt_bin[valid_mask],
                loc_pr_bin[valid_mask],
                2,
            )
            update_confusion(before_conf, dam_np.reshape(-1), dam_pred_np.reshape(-1), 4)
            update_confusion(
                before_building_conf,
                loc_gt_bin[valid_mask],
                np.isin(dam_pred_np, [1, 2, 3]).astype(np.int64)[valid_mask],
                2,
            )

            voted_batch = []
            for i in range(dam_pred_np.shape[0]):
                if args.object_voting_source == "pred_building":
                    building_mask = loc_pr_bin[i] > 0
                else:
                    building_mask = (loc_gt_bin[i] > 0) & valid_mask[i]
                voted = (
                    object_vote_damage_map(dam_pred_np[i], building_mask, args.object_min_area)
                    if args.use_object_voting else dam_pred_np[i].copy()
                )
                voted_batch.append(voted)

                if args.save_voted_maps:
                    name = names[i] if i < len(names) else f"sample_{sample_offset + i:06d}"
                    safe_name = os.path.basename(str(name)).replace(os.sep, "_") or f"sample_{sample_offset + i:06d}"
                    np.savez_compressed(
                        os.path.join(map_dir, f"{safe_name}.npz"),
                        pred_damage=dam_pred_np[i],
                        voted_damage=voted,
                        gt_damage=dam_np[i],
                        building_mask=building_mask.astype(np.uint8),
                    )

            voted_np = np.stack(voted_batch, axis=0)
            update_confusion(after_conf, dam_np.reshape(-1), voted_np.reshape(-1), 4)
            update_confusion(
                after_building_conf,
                loc_gt_bin[valid_mask],
                np.isin(voted_np, [1, 2, 3]).astype(np.int64)[valid_mask],
                2,
            )
            sample_offset += dam_pred_np.shape[0]

    before = summarize_official_conf(before_conf, loc_conf, before_building_conf)
    after = summarize_official_conf(after_conf, loc_conf, after_building_conf)
    comparison = {
        "use_object_voting": args.use_object_voting,
        "object_min_area": args.object_min_area,
        "object_voting_source": args.object_voting_source,
        **prefixed_summary("before", before),
        **prefixed_summary("after", after),
    }

    with open(os.path.join(args.save_dir, "object_voting_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready(comparison), f, indent=2, ensure_ascii=False)
    write_comparison_csv(os.path.join(args.save_dir, "object_voting_comparison.csv"), before, after)
    write_confusion_csv(os.path.join(args.save_dir, "confusion_before_object_voting.csv"), before_conf, CLASSES)
    write_confusion_csv(os.path.join(args.save_dir, "confusion_after_object_voting.csv"), after_conf, CLASSES)

    print(f"Saved object voting evaluation to: {args.save_dir}")
    print(
        "target_3class_miou: "
        f"{before['target_3class_miou']:.6f} -> {after['target_3class_miou']:.6f}"
    )
    print(
        "destroyed_to_intact: "
        f"{before['destroyed_to_intact']:.6f} -> {after['destroyed_to_intact']:.6f}"
    )


if __name__ == "__main__":
    main()
