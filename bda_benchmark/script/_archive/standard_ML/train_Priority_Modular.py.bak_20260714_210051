#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from priority_runtime import BDA_ROOT, build_dataset, parse_batch, set_seed
from priority_losses import (
    class_conditional_consistency_loss,
    flat_instance_loss,
    hierarchical_instance_loss,
    intact_alignment_loss,
    ramp_weight,
    segmentation_loss,
)
from model.DamageFormer_Priority_Modular import DamageFormerPriorityModular
from model.priority_region_ops import build_instance_map_from_binary


STAGES = {
    "B0A": dict(gamma_mode="fixed_zero", instance=False, pool="mean", area="inside", hierarchical=False, ia=False, consistency=False),
    "B0B": dict(gamma_mode="positive", instance=False, pool="mean", area="inside", hierarchical=False, ia=False, consistency=False),
    "B0C": dict(gamma_mode="positive", instance=False, pool="mean", area="inside", hierarchical=False, ia=True, consistency=False),
    "B2":  dict(gamma_mode="positive", instance=True, pool="mean", area="inside", hierarchical=False, ia=False, consistency=False),
    "B3":  dict(gamma_mode="positive", instance=True, pool="mean_topk", area="inside", hierarchical=False, ia=False, consistency=False),
    "B4B": dict(gamma_mode="positive", instance=True, pool="mean_topk", area="inside_boundary", hierarchical=False, ia=False, consistency=False),
    "B4R": dict(gamma_mode="positive", instance=True, pool="mean_topk", area="inside_boundary_ring", hierarchical=False, ia=False, consistency=False),
    "B5":  dict(gamma_mode="positive", instance=True, pool="mean_topk", area="inside_boundary_ring", hierarchical=True, ia=False, consistency=False),
    "B6":  dict(gamma_mode="positive", instance=True, pool="mean_topk", area="inside_boundary_ring", hierarchical=True, ia=False, consistency=True),
}


def update_hist(hist, pred, gt, classes, ignore=255):
    m = (gt != ignore) & (gt >= 0) & (gt < classes)
    idx = classes * gt[m] + pred[m]
    hist += torch.bincount(idx, minlength=classes * classes).reshape(classes, classes).double().cpu()


def compute_metrics(hist):
    inter = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = inter / union.clamp_min(1)
    row = hist / hist.sum(1, keepdim=True).clamp_min(1)
    return {
        "iou": [float(x * 100) for x in iou],
        "miou4": float(iou.mean() * 100),
        "miou3": float(iou[1:4].mean() * 100),
        "dmg_to_int": float(row[2, 1] * 100),
        "des_to_int": float(row[3, 1] * 100),
        "confusion": hist.tolist(),
    }


@torch.no_grad()
def validate(model, loader, device, args, cfg):
    model.eval()
    hist = torch.zeros(4, 4, dtype=torch.float64)
    destroyed_gt_inst = destroyed_pred_inst = instance_total = instance_correct = 0
    for batch_index, batch in enumerate(loader):
        if args.val_max_batches > 0 and batch_index >= args.val_max_batches:
            break
        pre, post, loc, dam, inst_hint, _ = parse_batch(batch, device)
        inst = inst_hint if (inst_hint is not None and int(inst_hint.max()) > 1) else build_instance_map_from_binary(loc == 1, args.min_instance_area)
        out = model(pre, post, inst if cfg["instance"] else None, dam if cfg["instance"] else None)
        pred = out["output_dam"].argmax(1)
        update_hist(hist, pred, dam, 4, args.ignore_index)
        if cfg["instance"] and out["instance_labels"].numel():
            valid = (out["instance_labels"] >= 1) & (out["instance_labels"] <= 3)
            y = out["instance_labels"][valid]
            p = out["building_logits3"][valid].argmax(-1) + 1
            instance_total += int(y.numel()); instance_correct += int((p == y).sum())
            destroyed_gt_inst += int((y == 3).sum()); destroyed_pred_inst += int((p == 3).sum())
    m = compute_metrics(hist)
    m.update({
        "instance_acc": instance_correct / max(instance_total, 1),
        "instance_total": instance_total,
        "destroyed_gt_instances": destroyed_gt_inst,
        "destroyed_pred_instances": destroyed_pred_inst,
        "gamma": float(model.hcdm.gamma().detach().cpu()),
    })
    return m


def load_checkpoint(model, path: str):
    if not path:
        return
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict", ck.get("model_state_dict", ck.get("model", ck)))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: {path}\nmissing={len(missing)}, unexpected={len(unexpected)}")
    if missing: print("missing sample:", missing[:20])
    if unexpected: print("unexpected sample:", unexpected[:20])


def main():
    ap = argparse.ArgumentParser("Prioritized modular BRIGHT experiments B0A-B6")
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--train_dataset_path", required=True); ap.add_argument("--train_data_list_path", required=True)
    ap.add_argument("--val_dataset_path", required=True); ap.add_argument("--val_data_list_path", required=True)
    ap.add_argument("--output_root", required=True); ap.add_argument("--exp_name", default="")
    ap.add_argument("--pretrained_model_path", default=""); ap.add_argument("--resume", default="")
    ap.add_argument("--image_size", type=int, default=640); ap.add_argument("--train_batch_size", type=int, default=2); ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=8); ap.add_argument("--optimizer_steps", type=int, default=50000); ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=1e-4); ap.add_argument("--weight_decay", type=float, default=5e-3); ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--amp_dtype", choices=["none", "fp16", "bf16"], default="bf16"); ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--diff_channels", type=int, default=64); ap.add_argument("--roi_alpha", type=float, default=0.1)
    ap.add_argument("--gamma_init", type=float, default=0.0025); ap.add_argument("--gamma_max", type=float, default=0.1); ap.add_argument("--align_dim", type=int, default=128)
    ap.add_argument("--lambda_ia", type=float, default=0.005); ap.add_argument("--ia_warmup", type=int, default=8000); ap.add_argument("--ia_ramp", type=int, default=2000)
    ap.add_argument("--lambda_inst", type=float, default=0.02); ap.add_argument("--inst_warmup", type=int, default=3000); ap.add_argument("--inst_ramp", type=int, default=2000)
    ap.add_argument("--lambda_cons", type=float, default=0.005); ap.add_argument("--cons_warmup", type=int, default=8000); ap.add_argument("--cons_ramp", type=int, default=4000)
    ap.add_argument("--topk_ratio", type=float, default=0.1); ap.add_argument("--boundary_radius", type=int, default=2); ap.add_argument("--ring_radius", type=int, default=4); ap.add_argument("--min_instance_area", type=int, default=8)
    ap.add_argument("--min_building_pixels_log", type=int, default=64, help="Diagnostic threshold only; crops are not discarded.")
    ap.add_argument("--log_interval", type=int, default=20); ap.add_argument("--val_interval", type=int, default=500); ap.add_argument("--save_interval", type=int, default=5000); ap.add_argument("--val_max_batches", type=int, default=0)
    ap.add_argument("--no_pretrained_swin", action="store_true")
    args = ap.parse_args()
    cfg = STAGES[args.stage]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = build_dataset(args.train_dataset_path, args.train_data_list_path, args.image_size, "train")
    val_ds = build_dataset(args.val_dataset_path, args.val_data_list_path, args.image_size, "val")
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = DamageFormerPriorityModular(
        gamma_mode=cfg["gamma_mode"], gamma_init=args.gamma_init, gamma_max=args.gamma_max,
        diff_channels=args.diff_channels, align_dim=args.align_dim, roi_alpha=args.roi_alpha,
        instance_enabled=cfg["instance"], pool_mode=cfg["pool"], area_mode=cfg["area"],
        hierarchical_head=cfg["hierarchical"], topk_ratio=args.topk_ratio,
        boundary_radius=args.boundary_radius, ring_radius=args.ring_radius,
        use_pretrained_swin=not args.no_pretrained_swin,
    ).to(device)
    load_checkpoint(model, args.pretrained_model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    exp_name = args.exp_name or f"{args.stage}_EBS{args.train_batch_size * args.accumulation_steps}_{args.optimizer_steps}steps"
    outdir = Path(args.output_root) / exp_name; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps({**vars(args), "resolved_stage": cfg}, indent=2, ensure_ascii=False), encoding="utf-8")
    start_step, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"], strict=True); optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_step = int(ck["step"]); best = float(ck.get("best_miou3", -1))
    optimizer.zero_grad(set_to_none=True)
    opt_step, micro_step = start_step, 0
    background_like = total_crops = 0
    while opt_step < args.optimizer_steps:
        for batch in train_loader:
            if opt_step >= args.optimizer_steps: break
            model.train()
            pre, post, loc, dam, inst_hint, _ = parse_batch(batch, device)
            building_pixels = (loc == 1).flatten(1).sum(1)
            background_like += int((building_pixels < args.min_building_pixels_log).sum())
            total_crops += int(loc.shape[0])
            inst = None
            if cfg["instance"]:
                inst = inst_hint if (inst_hint is not None and int(inst_hint.max()) > 1) else build_instance_map_from_binary(loc == 1, args.min_instance_area)
            amp = device.type == "cuda" and args.amp_dtype != "none"
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                out = model(pre, post, inst, dam if cfg["instance"] else None)
                l_loc, _, _ = segmentation_loss(out["output_loc"], loc, args.ignore_index)
                l_dam, _, _ = segmentation_loss(out["output_dam"], dam, args.ignore_index)
                l_ia, _ = intact_alignment_loss(out, dam, args.ignore_index)
                if cfg["instance"]:
                    l_inst, istat = hierarchical_instance_loss(out) if cfg["hierarchical"] else flat_instance_loss(out)
                else:
                    l_inst, istat = out["output_dam"].sum() * 0.0, {"instance_count": 0, "destroyed_gt": 0, "destroyed_pred": 0}
                l_cons = class_conditional_consistency_loss(out, dam) if cfg["consistency"] else out["output_dam"].sum() * 0.0
                w_ia = ramp_weight(opt_step, args.lambda_ia if cfg["ia"] else 0.0, args.ia_warmup, args.ia_ramp)
                w_inst = ramp_weight(opt_step, args.lambda_inst if cfg["instance"] else 0.0, args.inst_warmup, args.inst_ramp)
                w_cons = ramp_weight(opt_step, args.lambda_cons if cfg["consistency"] else 0.0, args.cons_warmup, args.cons_ramp)
                total = l_loc + l_dam + w_ia * l_ia + w_inst * l_inst + w_cons * l_cons
                scaled = total / args.accumulation_steps
            scaler.scale(scaled).backward(); micro_step += 1
            if micro_step % args.accumulation_steps != 0: continue
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); opt_step += 1
            if opt_step % args.log_interval == 0:
                bg_ratio = background_like / max(total_crops, 1)
                sw = out["scale_weights"].detach().mean(dim=(0, 2, 3)).cpu().tolist()
                print(
                    f"step {opt_step:06d}/{args.optimizer_steps} total={float(total):.4f} loc={float(l_loc):.4f} dam={float(l_dam):.4f} "
                    f"inst={float(l_inst):.4f} cons={float(l_cons):.4f} ia={float(l_ia):.4f} "
                    f"w_inst={w_inst:.4g} w_cons={w_cons:.4g} w_ia={w_ia:.4g} gamma={float(model.hcdm.gamma()):.6f} "
                    f"scale={'/'.join(f'{x:.3f}' for x in sw)} bg_like={bg_ratio:.3f} "
                    f"inst_n={istat.get('instance_count',0)} des_gt/pred={istat.get('destroyed_gt',0)}/{istat.get('destroyed_pred',0)}",
                    flush=True,
                )
            if opt_step % args.val_interval == 0 or opt_step == args.optimizer_steps:
                m = validate(model, val_loader, device, args, cfg)
                print("VALID", json.dumps(m, ensure_ascii=False), flush=True)
                payload = {"step": opt_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_miou3": max(best, m["miou3"]), "metrics": m, "args": vars(args), "stage": cfg}
                torch.save(payload, outdir / "last.pth")
                if m["miou3"] > best:
                    best = m["miou3"]
                    torch.save(payload, outdir / "best_miou3.pth")
                    print(f"NEW BEST miou3={best:.4f}")
            if opt_step % args.save_interval == 0:
                torch.save({"step": opt_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_miou3": best, "args": vars(args), "stage": cfg}, outdir / f"step_{opt_step:06d}.pth")


if __name__ == "__main__":
    main()
