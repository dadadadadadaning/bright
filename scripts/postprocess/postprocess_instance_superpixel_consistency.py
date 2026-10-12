#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instance/superpixel consistency post-processing for BRIGHT damage maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_evaluate_unified import (  # noqa: E402
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


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.uint8)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        for key in ("pred", "pred_damage", "mask", "arr_0"):
            if key in data:
                return np.asarray(data[key], dtype=np.uint8)
        raise KeyError(f"No mask-like key found in {path}")
    return np.asarray(Image.open(path), dtype=np.uint8)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def connected_components(mask: np.ndarray) -> Iterable[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(mask.astype(bool))
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        q: deque[Tuple[int, int]] = deque([(y0, x0)])
        visited[y0, x0] = True
        coords: List[Tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            coords.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    q.append((yy, xx))
        yield np.asarray(coords, dtype=np.int64)


def choose_label(labels: np.ndarray, probs: Optional[np.ndarray], dominant_ratio: float) -> Tuple[int, float]:
    labels = labels[(labels >= 1) & (labels <= 3)]
    if labels.size == 0:
        return 0, 0.0
    if probs is not None:
        class_scores = probs[:, 1:4].mean(axis=0)
        label = int(np.argmax(class_scores) + 1)
        return label, float(class_scores[label - 1])
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    label = int(np.argmax(counts[1:4]) + 1)
    return label, float(counts[label] / max(1.0, counts[1:4].sum()))


def apply_region_consistency(
    pred: np.ndarray,
    regions: Iterable[np.ndarray],
    probs: Optional[np.ndarray],
    dominant_ratio: float,
    min_area: int,
    damage_keep_ratio: float,
) -> np.ndarray:
    out = pred.copy()
    for coords in regions:
        if coords.shape[0] < min_area:
            continue
        yy, xx = coords[:, 0], coords[:, 1]
        labels = out[yy, xx]
        building = labels[(labels >= 1) & (labels <= 3)]
        if building.size == 0:
            continue
        damage_ratio = float(np.isin(building, [2, 3]).sum() / building.size)
        region_probs = probs[yy, xx] if probs is not None else None
        label, confidence = choose_label(labels, region_probs, dominant_ratio)
        if label == 0 or confidence < dominant_ratio:
            continue
        if label == 1 and damage_ratio > damage_keep_ratio:
            continue
        out[yy, xx] = label
    return out


def cc_consistency(
    pred: np.ndarray,
    probs: Optional[np.ndarray],
    dominant_ratio: float,
    min_area: int,
    damage_keep_ratio: float,
) -> np.ndarray:
    return apply_region_consistency(
        pred,
        connected_components(pred != 0),
        probs,
        dominant_ratio,
        min_area,
        damage_keep_ratio,
    )


def slic_regions(image: Optional[np.ndarray], pred: np.ndarray, n_segments: int, compactness: float) -> Iterable[np.ndarray]:
    if image is None:
        return connected_components(pred != 0)
    try:
        from skimage.segmentation import slic
    except Exception:
        return connected_components(pred != 0)
    img = image.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    denom = max(float(img.max() - img.min()), 1e-6)
    img = (img - float(img.min())) / denom
    segments = slic(img, n_segments=int(n_segments), compactness=float(compactness), start_label=1, channel_axis=-1)
    regions: List[np.ndarray] = []
    for seg_id in np.unique(segments):
        coords = np.column_stack(np.where(segments == seg_id)).astype(np.int64)
        if coords.size:
            regions.append(coords)
    return regions


def slic_consistency(
    pred: np.ndarray,
    probs: Optional[np.ndarray],
    image: Optional[np.ndarray],
    dominant_ratio: float,
    min_area: int,
    damage_keep_ratio: float,
    n_segments: int,
    compactness: float,
    building_overlap: float,
) -> np.ndarray:
    regions = []
    building = pred != 0
    for coords in slic_regions(image, pred, n_segments, compactness):
        yy, xx = coords[:, 0], coords[:, 1]
        if float(building[yy, xx].mean()) >= building_overlap:
            regions.append(coords)
    return apply_region_consistency(pred, regions, probs, dominant_ratio, min_area, damage_keep_ratio)


def postprocess_one(
    pred: np.ndarray,
    probs: Optional[np.ndarray],
    image: Optional[np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    if args.mode == "cc":
        return cc_consistency(pred, probs, args.dominant_ratio, args.min_area, args.damage_keep_ratio)
    return slic_consistency(
        pred,
        probs,
        image,
        args.dominant_ratio,
        args.min_area,
        args.damage_keep_ratio,
        args.n_segments,
        args.compactness,
        args.building_overlap,
    )


def summarize_row(conf: np.ndarray, loc_conf: np.ndarray, building_conf: np.ndarray) -> Dict[str, Any]:
    summary = summarize_official_conf(conf, loc_conf, building_conf)
    return {
        "mIoU4": summary.get("official_style_4class_miou"),
        "target3": summary.get("target_3class_miou"),
        "background_iou": summary.get("background_iou"),
        "intact_iou": summary.get("intact_iou"),
        "damaged_iou": summary.get("damaged_iou"),
        "destroyed_iou": summary.get("destroyed_iou"),
        "damaged_to_intact": summary.get("damaged_to_intact"),
        "destroyed_to_intact": summary.get("destroyed_to_intact"),
        "damaged_to_destroyed": summary.get("damaged_to_destroyed"),
        "destroyed_to_damaged": summary.get("destroyed_to_damaged"),
        "damaged_to_background": summary.get("damaged_to_background"),
        "destroyed_to_background": summary.get("destroyed_to_background"),
    }


def run_offline(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_out = out_dir / "masks"
    exts = {".png", ".tif", ".tiff", ".npy", ".npz"}
    for pred_path in sorted(p for p in pred_dir.iterdir() if p.suffix.lower() in exts):
        pred = load_mask(pred_path)
        out = postprocess_one(pred, None, None, args)
        save_mask(mask_out / f"{pred_path.stem}.png", out)
    with (out_dir / "postprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def run_model_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = read_list(args.data_list_path)
    DatasetClass = import_dataset_class()
    dataset = DatasetClass(args.dataset_path, data_list, args.image_size, None, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args)
    load_info = load_checkpoint(model, args.model_path, device)
    model.to(device)
    model.eval()

    loc_conf = np.zeros((2, 2), dtype=np.int64)
    building_conf = np.zeros((2, 2), dtype=np.int64)
    before_conf = np.zeros((4, 4), dtype=np.int64)
    after_conf = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Postprocess {args.mode}"):
            pre, post, loc_label, dam_label, names = parse_batch(batch, device)
            loc_logits, dam_logits = parse_outputs(model(pre, post))
            loc_logits = resize_logits(loc_logits, loc_label)
            dam_logits = resize_logits(dam_logits, dam_label)
            prob = F.softmax(dam_logits, dim=1)
            loc_pred = torch.argmax(loc_logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            loc_np = loc_label.detach().cpu().numpy()
            dam_np = dam_label.detach().cpu().numpy()
            loc_pred_np = loc_pred.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            prob_np = prob.permute(0, 2, 3, 1).detach().cpu().numpy()
            pre_np = pre.detach().cpu().numpy()

            valid = dam_np != 255
            loc_gt_bin = np.isin(dam_np, [1, 2, 3]).astype(np.int64)
            update_confusion(loc_conf, loc_gt_bin[valid], (loc_pred_np > 0).astype(np.int64)[valid], 2)
            update_confusion(before_conf, dam_np.reshape(-1), pred_np.reshape(-1), 4)

            refined = []
            for i in range(pred_np.shape[0]):
                image = np.transpose(pre_np[i, :3], (1, 2, 0))
                out = postprocess_one(pred_np[i], prob_np[i], image, args)
                refined.append(out)
                if args.save_masks:
                    safe_name = os.path.basename(str(names[i])) if i < len(names) else f"sample_{len(refined):06d}"
                    save_mask(out_dir / "masks" / f"{safe_name}.png", out)
            refined_np = np.stack(refined, axis=0)
            update_confusion(after_conf, dam_np.reshape(-1), refined_np.reshape(-1), 4)
            update_confusion(building_conf, loc_gt_bin[valid], np.isin(refined_np, [1, 2, 3]).astype(np.int64)[valid], 2)

    before = summarize_row(before_conf, loc_conf, building_conf)
    after = summarize_row(after_conf, loc_conf, building_conf)
    result = {
        "mode": args.mode,
        "dominant_ratio": args.dominant_ratio,
        "min_area": args.min_area,
        "damage_keep_ratio": args.damage_keep_ratio,
        "n_segments": args.n_segments,
        "compactness": args.compactness,
        "load_missing_keys_count": load_info.get("load_missing_keys_count"),
        "load_unexpected_keys_count": load_info.get("load_unexpected_keys_count"),
        **{f"before_{k}": v for k, v in before.items()},
        **after,
    }
    with (out_dir / "summary_official.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2, ensure_ascii=False)
    with (out_dir / "official_metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
    write_confusion_csv(str(out_dir / "confusion_4class_official.csv"), after_conf, CLASSES)
    print(f"Saved postprocess evaluation to: {out_dir}")
    print(f"target3: {after['target3']:.6f}; damaged_iou: {after['damaged_iou']:.6f}; destroyed_iou: {after['destroyed_iou']:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Apply instance/superpixel consistency to predicted masks or model outputs."
    for action in parser._actions:
        if action.dest in {"model_path", "dataset_path", "data_list_path", "save_dir"}:
            action.required = False
    parser.add_argument("--pred_dir", default="", help="Offline prediction mask directory.")
    parser.add_argument("--prob_dir", default="", help="Optional probability directory for offline mode.")
    parser.add_argument("--image_dir", default="", help="Optional image directory for offline SLIC mode.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["cc", "slic"], default="cc")
    parser.add_argument("--dominant_ratio", type=float, default=0.65)
    parser.add_argument("--min_area", type=int, default=50)
    parser.add_argument("--damage_keep_ratio", type=float, default=0.25)
    parser.add_argument("--n_segments", type=int, default=1000)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--building_overlap", type=float, default=0.5)
    parser.add_argument("--save_masks", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pred_dir:
        run_offline(args)
    else:
        missing = [name for name in ("model_path", "dataset_path", "data_list_path") if not getattr(args, name, "")]
        if missing:
            raise SystemExit(f"Model-eval mode requires: {', '.join('--' + m for m in missing)}")
        run_model_eval(args)


if __name__ == "__main__":
    main()
