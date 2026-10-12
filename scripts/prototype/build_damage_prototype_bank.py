#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build damaged/destroyed feature prototypes from the BRIGHT training set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_evaluate_unified import build_arg_parser, build_model, import_dataset_class, load_checkpoint, parse_batch, read_list  # noqa: E402


CLASS_IDS = {"damaged": 2, "destroyed": 3}


def cap_append(store: List[torch.Tensor], feats: torch.Tensor, max_items: int) -> None:
    if feats.numel() == 0:
        return
    current = sum(t.shape[0] for t in store)
    remaining = max_items - current
    if remaining <= 0:
        return
    if feats.shape[0] > remaining:
        perm = torch.randperm(feats.shape[0], device=feats.device)[:remaining]
        feats = feats[perm]
    store.append(feats.detach().cpu())


def simple_kmeans(x: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    if k <= 1 or x.shape[0] <= k:
        return x.mean(dim=0, keepdim=True) if x.shape[0] > 0 else x
    init_idx = torch.linspace(0, x.shape[0] - 1, steps=k).long()
    centers = x[init_idx].clone()
    for _ in range(iters):
        sim = torch.matmul(F.normalize(x, dim=1), F.normalize(centers, dim=1).t())
        assign = sim.argmax(dim=1)
        new_centers = []
        for i in range(k):
            part = x[assign == i]
            new_centers.append(part.mean(dim=0) if part.numel() else centers[i])
        centers = torch.stack(new_centers, dim=0)
    return centers


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    for action in parser._actions:
        if action.dest in {"model_path", "dataset_path", "data_list_path", "save_dir"}:
            action.required = False
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--train_list", default="")
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_clusters", type=int, default=1)
    parser.add_argument("--max_pixels_per_class", type=int, default=200000)
    parser.add_argument("--classes", nargs="+", default=["damaged", "destroyed"], choices=list(CLASS_IDS))
    parser.add_argument("--feature_layer", choices=["penultimate", "damage_head_feature"], default="penultimate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.model_path = args.checkpoint or args.model_path
    args.dataset_path = args.dataset_root or args.dataset_path
    args.data_list_path = args.train_list or args.data_list_path
    if not args.model_path or not args.dataset_path or not args.data_list_path:
        raise SystemExit("--checkpoint/--model_path, --dataset_root/--dataset_path, and --train_list/--data_list_path are required.")

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = read_list(args.data_list_path)
    DatasetClass = import_dataset_class()
    dataset = DatasetClass(args.dataset_path, data_list, args.image_size, None, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args)
    load_info = load_checkpoint(model, args.model_path, device)
    model.to(device)
    model.eval()

    samples: Dict[str, List[torch.Tensor]] = {name: [] for name in args.classes}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building prototypes"):
            pre, post, _loc, dam, _names = parse_batch(batch, device)
            outputs = model(pre, post, return_dict=True)
            feat = outputs["final_feat_fused_roi"]
            feat = F.interpolate(feat, size=dam.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            labels = dam.reshape(-1)
            for name in args.classes:
                cls_id = CLASS_IDS[name]
                cls_feat = feat[labels == cls_id]
                if cls_feat.numel():
                    cls_feat = F.normalize(cls_feat, dim=1)
                    cap_append(samples[name], cls_feat, int(args.max_pixels_per_class))
            if all(sum(t.shape[0] for t in samples[name]) >= args.max_pixels_per_class for name in args.classes):
                break

    prototypes: Dict[str, torch.Tensor] = {}
    metadata = {
        "checkpoint": args.model_path,
        "train_list": args.data_list_path,
        "dataset_root": args.dataset_path,
        "num_clusters": args.num_clusters,
        "max_pixels_per_class": args.max_pixels_per_class,
        "feature_layer": args.feature_layer,
        "load_missing_keys_count": load_info.get("load_missing_keys_count"),
        "load_unexpected_keys_count": load_info.get("load_unexpected_keys_count"),
        "classes": {},
    }
    for name, chunks in samples.items():
        if not chunks:
            raise RuntimeError(f"No features collected for class {name}")
        x = torch.cat(chunks, dim=0)
        centers = simple_kmeans(x, int(args.num_clusters))
        prototypes[name] = F.normalize(centers, dim=1).cpu()
        metadata["classes"][name] = {"num_pixels": int(x.shape[0]), "num_prototypes": int(centers.shape[0])}

    torch.save({"prototypes": prototypes, "metadata": metadata}, save_path / "prototypes.pt")
    with (save_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved prototype bank to: {save_path}")


if __name__ == "__main__":
    main()
