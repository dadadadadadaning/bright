#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate damaged/destroyed prototype-logit refinement."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
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
    read_list,
    resize_logits,
    summarize_official_conf,
    update_confusion,
    write_confusion_csv,
)


CLASSES = ["background", "intact", "damaged", "destroyed"]


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    for action in parser._actions:
        if action.dest in {"model_path", "dataset_path", "data_list_path", "save_dir"}:
            action.required = False
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prototype_path", required=True)
    parser.add_argument("--test_list", default="")
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--apply_region", choices=["building", "uncertain_damage"], default="building")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.65)
    return parser


def proto_similarity(feat: torch.Tensor, proto: torch.Tensor, topk: int) -> torch.Tensor:
    sim = torch.matmul(F.normalize(feat, dim=1), F.normalize(proto.to(feat.device), dim=1).t())
    if topk <= 1 or sim.shape[1] == 1:
        return sim.max(dim=1).values
    return sim.topk(min(topk, sim.shape[1]), dim=1).values.mean(dim=1)


def summarize(conf: np.ndarray, loc_conf: np.ndarray, building_conf: np.ndarray) -> Dict[str, Any]:
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


def main() -> None:
    args = build_parser().parse_args()
    args.model_path = args.checkpoint or args.model_path
    args.dataset_path = args.dataset_root or args.dataset_path
    args.data_list_path = args.test_list or args.data_list_path
    if not args.model_path or not args.dataset_path or not args.data_list_path:
        raise SystemExit("--checkpoint/--model_path, --dataset_root/--dataset_path, and --test_list/--data_list_path are required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.prototype_path, map_location="cpu")
    prototypes = bank["prototypes"] if isinstance(bank, dict) and "prototypes" in bank else bank
    damaged_proto = prototypes["damaged"].float()
    destroyed_proto = prototypes["destroyed"].float()

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
    conf = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Prototype refine"):
            pre, post, loc_label, dam_label, _names = parse_batch(batch, device)
            outputs = model(pre, post, return_dict=True)
            loc_logits = resize_logits(outputs["output_loc"], loc_label)
            dam_logits = resize_logits(outputs["output_dam"], dam_label).clone()
            feat = F.interpolate(outputs["final_feat_fused_roi"], size=dam_label.shape[-2:], mode="bilinear", align_corners=False)
            b, c, h, w = feat.shape
            flat_feat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            sim_dmg = proto_similarity(flat_feat, damaged_proto, args.topk).reshape(b, h, w)
            sim_des = proto_similarity(flat_feat, destroyed_proto, args.topk).reshape(b, h, w)

            prob = F.softmax(dam_logits, dim=1)
            pred = prob.argmax(dim=1)
            if args.apply_region == "building":
                apply = pred != 0
            else:
                max_prob = prob.max(dim=1).values
                gap_dmg_int = torch.abs(dam_logits[:, 2] - dam_logits[:, 1])
                gap_des_dmg = torch.abs(dam_logits[:, 3] - dam_logits[:, 2])
                apply = (pred != 0) & ((max_prob < args.uncertainty_threshold) | (gap_dmg_int < 0.5) | (gap_des_dmg < 0.5))
            dam_logits[:, 2] = torch.where(apply, dam_logits[:, 2] + float(args.alpha) * sim_dmg, dam_logits[:, 2])
            dam_logits[:, 3] = torch.where(apply, dam_logits[:, 3] + float(args.alpha) * sim_des, dam_logits[:, 3])

            refined = dam_logits.argmax(dim=1)
            loc_pred = loc_logits.argmax(dim=1)
            dam_np = dam_label.detach().cpu().numpy()
            refined_np = refined.detach().cpu().numpy()
            loc_np = loc_label.detach().cpu().numpy()
            loc_pred_np = loc_pred.detach().cpu().numpy()
            valid = dam_np != 255
            loc_gt_bin = np.isin(dam_np, [1, 2, 3]).astype(np.int64)
            update_confusion(loc_conf, loc_gt_bin[valid], (loc_pred_np > 0).astype(np.int64)[valid], 2)
            update_confusion(conf, dam_np.reshape(-1), refined_np.reshape(-1), 4)
            update_confusion(building_conf, loc_gt_bin[valid], np.isin(refined_np, [1, 2, 3]).astype(np.int64)[valid], 2)

    row = summarize(conf, loc_conf, building_conf)
    row.update(
        {
            "alpha": args.alpha,
            "topk": args.topk,
            "apply_region": args.apply_region,
            "uncertainty_threshold": args.uncertainty_threshold,
            "load_missing_keys_count": load_info.get("load_missing_keys_count"),
            "load_unexpected_keys_count": load_info.get("load_unexpected_keys_count"),
        }
    )
    with (out_dir / "summary_official.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(row), f, indent=2, ensure_ascii=False)
    with (out_dir / "official_metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in row.items():
            f.write(f"{key}: {value}\n")
    write_confusion_csv(str(out_dir / "confusion_4class_official.csv"), conf, CLASSES)
    print(f"Saved prototype refine evaluation to: {out_dir}")
    print(f"target3={row['target3']:.6f}, damaged_iou={row['damaged_iou']:.6f}, destroyed_iou={row['destroyed_iou']:.6f}")


if __name__ == "__main__":
    main()
