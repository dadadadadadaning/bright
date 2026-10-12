#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from priority_runtime import build_dataset, parse_batch, set_seed
from model.DamageFormer_Priority_Modular import DamageFormerPriorityModular
from model.priority_region_ops import binary_dilate, binary_erode, build_instance_map_from_binary, collect_instance_records, resize_instance_map


def main():
    ap = argparse.ArgumentParser("B1 offline instance-pooling diagnosis; no training")
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--dataset_path", required=True); ap.add_argument("--data_list_path", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--image_size", type=int, default=640); ap.add_argument("--batch_size", type=int, default=1); ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--output_dir", required=True); ap.add_argument("--min_instance_area", type=int, default=8); ap.add_argument("--boundary_radius", type=int, default=2); ap.add_argument("--ring_radius", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=0); ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(args.dataset_path, args.data_list_path, args.image_size, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = DamageFormerPriorityModular(gamma_mode="positive", instance_enabled=False, use_pretrained_swin=False).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)); sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False); model.eval()
    rows = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches: break
            pre, post, loc, dam, inst_hint, names = parse_batch(batch, device)
            out = model(pre, post)
            prob = torch.softmax(out["output_dam"].float(), 1)
            feat_size = out["final_feat_damage"].shape[-2:]
            inst = inst_hint if (inst_hint is not None and int(inst_hint.max()) > 1) else build_instance_map_from_binary(loc == 1, args.min_instance_area)
            inst_s = resize_instance_map(inst, feat_size)
            dam_s = F.interpolate(dam[:, None].float(), feat_size, mode="nearest")[:, 0].long()
            p_s = F.interpolate(prob, feat_size, mode="bilinear", align_corners=False)
            for rec in collect_instance_records(inst_s, dam_s):
                inside = inst_s[rec.batch_index] == rec.instance_id
                boundary = inside & ~binary_erode(inside, args.boundary_radius)
                ring = binary_dilate(inside, args.ring_radius) & ~inside
                def stats(mask):
                    vals = p_s[rec.batch_index, 3, mask]
                    if vals.numel() == 0: return (0.0, 0.0, 0.0, 0.0)
                    k10 = max(1, int(vals.numel() * .10)); k20 = max(1, int(vals.numel() * .20))
                    return float(vals.mean()), float(vals.max()), float(vals.topk(k10).values.mean()), float(vals.topk(k20).values.mean())
                im, ix, i10, i20 = stats(inside); bm, bx, b10, b20 = stats(boundary); rm, rx, r10, r20 = stats(ring)
                cls_inside = p_s[rec.batch_index, :, inside].mean(1).argmax().item() if inside.any() else -1
                entropy = 0.0
                if inside.any():
                    q = p_s[rec.batch_index, :, inside].mean(1); q = q / q.sum().clamp_min(1e-8)
                    entropy = float(-(q * q.clamp_min(1e-8).log()).sum())
                rows.append({"batch": bi, "sample": str(names[rec.batch_index]) if isinstance(names, (list, tuple)) else str(names), "instance_id": rec.instance_id, "gt_label": rec.label, "area": rec.area, "pred_inside": cls_inside, "entropy": entropy,
                             "inside_mean_des": im, "inside_max_des": ix, "inside_top10_des": i10, "inside_top20_des": i20,
                             "boundary_mean_des": bm, "boundary_max_des": bx, "boundary_top10_des": b10, "boundary_top20_des": b20,
                             "ring_mean_des": rm, "ring_max_des": rx, "ring_top10_des": r10, "ring_top20_des": r20})
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(outdir / "per_instance.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {}
    for cls in (1, 2, 3):
        rr = [r for r in rows if r["gt_label"] == cls]
        summary[str(cls)] = {"count": len(rr)}
        for key in ("inside_mean_des", "inside_top10_des", "boundary_mean_des", "ring_mean_des", "entropy"):
            summary[str(cls)][key] = sum(r[key] for r in rr) / max(len(rr), 1)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
