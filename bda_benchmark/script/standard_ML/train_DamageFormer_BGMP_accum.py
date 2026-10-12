# -*- coding: utf-8 -*-
"""S2 training: S1 + Damage Prototype Memory (K configurable).

Recommended 1x5090 S2 launch:
  micro batch=4, accumulation=4 -> effective batch=16.

Important prototype detail under accumulation:
The current prototype memory updates once per micro-batch. The CLI argument
``--proto_step_momentum`` expresses the desired EMA retention per optimizer step.
The script converts it to micro-batch momentum as m_micro = m_step ** (1/ACC),
so four micro-updates retain approximately the same old-memory weight as one
m_step=0.99 update.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"

for import_root in (PROJECT_ROOT, BDA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BGMP import DamageFormer_BGMP
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2: DamageFormer_BGMP with gradient accumulation")
    p.add_argument("--dataset", type=str, default="BRIGHT")
    p.add_argument("--train_dataset_path", type=str, required=True)
    p.add_argument("--train_data_list_path", type=str, required=True)
    p.add_argument("--val_dataset_path", type=str, required=True)
    p.add_argument("--val_data_list_path", type=str, required=True)
    p.add_argument("--test_dataset_path", type=str, required=True)
    p.add_argument("--test_data_list_path", type=str, required=True)

    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--optimizer_steps", type=int, default=50000)
    p.add_argument(
        "--max_iters",
        type=int,
        default=800000,
        help="Number of training crops/samples passed to the dataset, NOT optimizer steps.",
    )
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--roi_alpha", type=float, default=0.1)
    p.add_argument("--proto_k", type=int, default=1)
    p.add_argument("--proto_dim", type=int, default=128)
    p.add_argument(
        "--proto_step_momentum",
        type=float,
        default=0.99,
        help="Desired prototype EMA retention per optimizer step; converted to micro-batch momentum.",
    )
    p.add_argument("--proto_tau", type=float, default=0.1)
    p.add_argument(
        "--proto_warmup_iters",
        type=int,
        default=5000,
        help="Warm-up measured in optimizer steps, because current_iter=optimizer_step.",
    )

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_type", type=str, default="S2_BGMP_K1")
    p.add_argument("--model_param_path", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def read_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(
    output_loc: torch.Tensor,
    output_clf: torch.Tensor,
    labels_loc: torch.Tensor,
    labels_clf: torch.Tensor,
) -> torch.Tensor:
    ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
    lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
    ce_loss_clf = F.cross_entropy(output_clf, labels_clf)
    lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
    return ce_loss_loc + ce_loss_clf + 0.75 * lovasz_loss_clf + 0.5 * lovasz_loss_loc


def building_prior_stats(aux: Dict[str, torch.Tensor], labels_loc: torch.Tensor) -> Dict[str, float]:
    roi = aux["roi_weight"].detach()
    p = aux["p_building"].detach()
    gt = F.interpolate(labels_loc.unsqueeze(1).float(), size=roi.shape[-2:], mode="nearest")[:, 0]
    valid = gt != 255
    bmask = (gt > 0) & valid
    bgmask = (gt == 0) & valid

    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
        x2 = x[:, 0]
        if not bool(mask.any().item()):
            return float("nan")
        return float(x2[mask].mean().item())

    rb = masked_mean(roi, bmask)
    rbg = masked_mean(roi, bgmask)
    return {
        "p_mean": float(p.mean().item()),
        "roi_mean": float(roi.mean().item()),
        "roi_build": rb,
        "roi_bg": rbg,
        "roi_gap": rb - rbg if np.isfinite(rb) and np.isfinite(rbg) else float("nan"),
    }


def prototype_class_accuracy(aux: Dict[str, object], labels_clf: torch.Tensor) -> float:
    sim = aux.get("proto_class_similarity")
    if sim is None:
        return float("nan")
    feat = aux["final_feat_fused_proto"]
    target = F.interpolate(
        labels_clf.unsqueeze(1).float(), size=feat.shape[-2:], mode="nearest"
    )[:, 0].long().reshape(-1)
    valid = (target >= 1) & (target <= 3)
    if not bool(valid.any().item()):
        return float("nan")
    pred = sim.argmax(dim=1) + 1
    return float((pred[valid] == target[valid]).float().mean().item())


def prototype_diagnostics(model: DamageFormer_BGMP, previous: torch.Tensor | None) -> Tuple[Dict[str, object], torch.Tensor]:
    memory = model.damage_proto_memory
    proto = memory.prototypes.detach()
    norms = proto.norm(dim=-1)
    current = proto.clone()
    drift = float("nan") if previous is None else float((current - previous).norm(dim=-1).mean().item())
    pnorm = F.normalize(proto, dim=-1)
    class_mean = F.normalize(pnorm.mean(dim=1), dim=-1)
    interclass = class_mean @ class_mean.t()
    return {
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "drift": drift,
        "interclass_cos": interclass.cpu().tolist(),
    }, current


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataset_path: str,
    data_names: list[str],
    eval_batch_size: int,
    device: torch.device,
    num_workers: int,
    split_name: str,
):
    print(f"---------starting {split_name}-----------", flush=True)
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    dataset = MultimodalDamageAssessmentDatset(dataset_path, data_names, 1024, None, "test")
    loader = DataLoader(dataset, batch_size=eval_batch_size, num_workers=num_workers, drop_last=False)
    model.eval()
    for pre, post, labels_loc, labels_clf, _ in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc_t = labels_loc.to(device, non_blocking=True).long()
        labels_clf_t = labels_clf.to(device, non_blocking=True).long()
        output_loc, output_clf = model(pre, post)
        pred_loc = output_loc.argmax(dim=1).cpu().numpy()
        pred_clf = output_clf.argmax(dim=1).cpu().numpy()
        gt_loc = labels_loc_t.cpu().numpy()
        gt_clf = labels_clf_t.cpu().numpy()
        evaluator_loc.add_batch(gt_loc, pred_loc)
        evaluator_clf.add_batch(gt_clf[gt_loc > 0], pred_clf[gt_loc > 0])
        evaluator_total.add_batch(gt_clf, pred_clf)

    loc_f1 = evaluator_loc.Pixel_F1_score()
    damage_f1 = evaluator_clf.Damage_F1_score()
    harmonic_f1 = len(damage_f1) / np.sum(1.0 / damage_f1)
    oa = evaluator_total.Pixel_Accuracy()
    ious = evaluator_total.Intersection_over_Union()
    miou = evaluator_total.Mean_Intersection_over_Union()
    print(f"{split_name}: OA={100*oa:.4f}, mIoU={100*miou:.4f}, IoU={100*ious}", flush=True)
    return loc_f1, harmonic_f1, oa, miou, ious


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    device = torch.device("cuda:0")

    effective_bs = args.train_batch_size * args.accumulation_steps
    expected_crops = effective_bs * args.optimizer_steps
    if effective_bs != 16:
        raise ValueError(
            f"Paper-equivalent run requires effective batch size 16, got {effective_bs} = "
            f"{args.train_batch_size} micro_batch * {args.accumulation_steps} accumulation."
        )
    if args.max_iters != expected_crops:
        raise ValueError(
            f"Schedule mismatch: max_iters is dataset crop count. For effective_bs={effective_bs} and "
            f"optimizer_steps={args.optimizer_steps}, it must be {expected_crops}, got {args.max_iters}."
        )
    if not 0.0 < args.proto_step_momentum < 1.0:
        raise ValueError("proto_step_momentum must be in (0,1)")
    proto_micro_momentum = args.proto_step_momentum ** (1.0 / args.accumulation_steps)

    train_names = read_list(args.train_data_list_path)
    val_names = read_list(args.val_data_list_path)
    test_names = read_list(args.test_data_list_path)

    model = DamageFormer_BGMP(
        roi_alpha=args.roi_alpha,
        proto_k=args.proto_k,
        proto_dim=args.proto_dim,
        proto_momentum=proto_micro_momentum,
        proto_tau=args.proto_tau,
        proto_warmup_iters=args.proto_warmup_iters,
    ).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = MultimodalDamageAssessmentDatset(
        args.train_dataset_path,
        train_names,
        crop_size=args.crop_size,
        max_iters=args.max_iters,
        type="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.model_param_path) / args.dataset / f"{args.model_type}_{now}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Training schedule", flush=True)
    print(f"  GPU                    : 1", flush=True)
    print(f"  micro batch            : {args.train_batch_size}", flush=True)
    print(f"  accumulation_steps     : {args.accumulation_steps}", flush=True)
    print(f"  effective batch size   : {effective_bs}", flush=True)
    print(f"  optimizer steps        : {args.optimizer_steps}", flush=True)
    print(f"  training crops         : {args.max_iters}", flush=True)
    print(f"  lr / weight_decay      : {args.learning_rate} / {args.weight_decay}", flush=True)
    print(f"  proto K / dim / tau    : {args.proto_k} / {args.proto_dim} / {args.proto_tau}", flush=True)
    print(f"  proto warmup opt steps : {args.proto_warmup_iters}", flush=True)
    print(f"  proto EMA m_step       : {args.proto_step_momentum:.8f}", flush=True)
    print(f"  proto EMA m_micro      : {proto_micro_momentum:.8f}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    accum_counter = 0
    best_miou = -1.0
    best_round: Dict[str, object] = {}
    running_raw_loss = 0.0
    previous_proto: torch.Tensor | None = None

    for data in tqdm(loader, total=len(loader)):
        pre, post, labels_loc, labels_clf, _ = data
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc = labels_loc.to(device, non_blocking=True).long()
        labels_clf = labels_clf.to(device, non_blocking=True).long()

        output_loc, output_clf, aux = model(
            pre,
            post,
            damage_target=labels_clf,
            current_iter=optimizer_step,
            return_aux=True,
        )
        raw_loss = compute_loss(output_loc, output_clf, labels_loc, labels_clf)
        (raw_loss / args.accumulation_steps).backward()
        running_raw_loss += float(raw_loss.detach().item())
        accum_counter += 1
        if accum_counter < args.accumulation_steps:
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        accum_counter = 0

        if optimizer_step % args.log_every == 0:
            bg = building_prior_stats(aux, labels_loc)
            proto_acc = prototype_class_accuracy(aux, labels_clf)
            diag, previous_proto = prototype_diagnostics(model, previous_proto)
            avg_loss = running_raw_loss / (args.log_every * args.accumulation_steps)
            print(
                f"opt_step {optimizer_step}/{args.optimizer_steps} | loss {avg_loss:.4f} | "
                f"roi_gap {bg['roi_gap']:.4f} | proto_active {aux['proto_active']} | "
                f"proto_init {aux['proto_initialized'].tolist()} | gamma {float(aux['proto_gamma']):.4f} | "
                f"proto_cls_acc {proto_acc:.4f} | proto_norm [{diag['norm_min']:.4f},{diag['norm_max']:.4f}] | "
                f"proto_drift {diag['drift']:.6f} | usage {aux['proto_usage_ratio'].detach().cpu().tolist()}",
                flush=True,
            )
            if optimizer_step % args.eval_every == 0:
                print(f"prototype inter-class cosine={diag['interclass_cos']}", flush=True)
            running_raw_loss = 0.0

        if optimizer_step % args.eval_every == 0:
            val = evaluate_split(
                model, args.val_dataset_path, val_names, args.eval_batch_size,
                device, max(1, args.num_workers // 2), "validation"
            )
            test = evaluate_split(
                model, args.test_dataset_path, test_names, args.eval_batch_size,
                device, max(1, args.num_workers // 2), "testing"
            )
            if val[3] > best_miou:
                best_miou = float(val[3])
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                best_round = {
                    "best optimizer step": optimizer_step,
                    "loc f1 (val)": val[0] * 100,
                    "clf f1 (val)": val[1] * 100,
                    "OA (val)": val[2] * 100,
                    "mIoU (val)": val[3] * 100,
                    "sub class IoU (val)": val[4] * 100,
                    "loc f1 (test)": test[0] * 100,
                    "clf f1 (test)": test[1] * 100,
                    "OA (test)": test[2] * 100,
                    "mIoU (test)": test[3] * 100,
                    "sub class IoU (test)": test[4] * 100,
                }
            model.train()

        if optimizer_step >= args.optimizer_steps:
            break

    if optimizer_step != args.optimizer_steps:
        raise RuntimeError(
            f"Training ended at {optimizer_step} optimizer steps, expected {args.optimizer_steps}. "
            "Check max_iters, batch size, and accumulation settings."
        )
    print("The accuracy of the best round is", best_round, flush=True)


if __name__ == "__main__":
    main()
