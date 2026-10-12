#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from priority_runtime import build_dataset, parse_batch
from train_Priority_Modular import STAGES, compute_metrics, update_hist
from model.DamageFormer_Priority_Modular import DamageFormerPriorityModular
from model.priority_region_ops import build_instance_map_from_binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES)); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_path", required=True); ap.add_argument("--data_list_path", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--image_size", type=int, default=640); ap.add_argument("--batch_size", type=int, default=1); ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--output_json", required=True); ap.add_argument("--min_instance_area", type=int, default=8)
    ap.add_argument("--topk_ratio", type=float, default=0.1); ap.add_argument("--boundary_radius", type=int, default=2); ap.add_argument("--ring_radius", type=int, default=4)
    ap.add_argument("--instance_source", choices=["gt_cc", "provided", "none"], default="gt_cc")
    args = ap.parse_args(); cfg = STAGES[args.stage]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(args.dataset_path, args.data_list_path, args.image_size, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = DamageFormerPriorityModular(gamma_mode=cfg["gamma_mode"], instance_enabled=cfg["instance"], pool_mode=cfg["pool"], area_mode=cfg["area"], hierarchical_head=cfg["hierarchical"], topk_ratio=args.topk_ratio, boundary_radius=args.boundary_radius, ring_radius=args.ring_radius, use_pretrained_swin=False).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu"); sd = ck.get("model_state_dict", ck.get("state_dict", ck)); sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False); model.eval(); hist = torch.zeros(4, 4, dtype=torch.float64)
    inst_total = inst_correct = des_gt = des_pred = 0
    with torch.no_grad():
        for batch in loader:
            pre, post, loc, dam, inst_hint, _ = parse_batch(batch, device)
            inst = None
            if cfg["instance"] and args.instance_source != "none":
                if args.instance_source == "provided" and inst_hint is not None and int(inst_hint.max()) > 1: inst = inst_hint
                else: inst = build_instance_map_from_binary(loc == 1, args.min_instance_area)
            out = model(pre, post, inst, dam if inst is not None else None)
            update_hist(hist, out["output_dam"].argmax(1), dam, 4)
            if inst is not None and out["instance_labels"].numel():
                valid = (out["instance_labels"] >= 1) & (out["instance_labels"] <= 3); y = out["instance_labels"][valid]; p = out["building_logits3"][valid].argmax(-1) + 1
                inst_total += int(y.numel()); inst_correct += int((p == y).sum()); des_gt += int((y == 3).sum()); des_pred += int((p == 3).sum())
    m = compute_metrics(hist); m.update({"instance_acc": inst_correct / max(inst_total, 1), "instance_total": inst_total, "destroyed_gt_instances": des_gt, "destroyed_pred_instances": des_pred, "gamma": float(model.hcdm.gamma())})
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True); Path(args.output_json).write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
