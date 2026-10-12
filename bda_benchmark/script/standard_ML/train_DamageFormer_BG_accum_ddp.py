# -*- coding: utf-8 -*-
"""S1 training: DamageFormer + building-prior soft ROI.

Paper-equivalent optimization schedule under memory-limited hardware:
- AdamW, lr=1e-4, weight_decay=5e-3
- CE + Lovasz-Softmax objective unchanged
- effective batch size = local_batch * world_size * accumulation_steps = 16
- optimizer steps = 50,000
- global training crops = 800,000

Recommended 2x4090 launch:
  local batch/GPU=4, world_size=2, accumulation=2 -> effective batch=16.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"

for import_root in (PROJECT_ROOT, BDA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BG import DamageFormer_BG
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1: DamageFormer_BG with DDP + gradient accumulation")
    p.add_argument("--dataset", type=str, default="BRIGHT")
    p.add_argument("--train_dataset_path", type=str, required=True)
    p.add_argument("--train_data_list_path", type=str, required=True)
    p.add_argument("--val_dataset_path", type=str, required=True)
    p.add_argument("--val_data_list_path", type=str, required=True)
    p.add_argument("--test_dataset_path", type=str, required=True)
    p.add_argument("--test_data_list_path", type=str, required=True)

    p.add_argument("--train_batch_size_per_gpu", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--optimizer_steps", type=int, default=50000)
    p.add_argument(
        "--max_iters",
        type=int,
        default=800000,
        help="Global number of training crops/samples passed to the dataset, NOT optimizer steps.",
    )
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--eval_every", type=int, default=500, help="Evaluate every N optimizer steps")
    p.add_argument("--log_every", type=int, default=10, help="Log every N optimizer steps")

    p.add_argument("--roi_alpha", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--model_type", type=str, default="S1_BG_alpha01")
    p.add_argument("--model_param_path", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def read_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    torch.cuda.manual_seed_all(actual)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def compute_loss(
    output_loc: torch.Tensor,
    output_clf: torch.Tensor,
    labels_loc: torch.Tensor,
    labels_clf: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
    lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
    ce_loss_clf = F.cross_entropy(output_clf, labels_clf)
    lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
    total = ce_loss_loc + ce_loss_clf + 0.75 * lovasz_loss_clf + 0.5 * lovasz_loss_loc
    return total, {
        "ce_loc": ce_loss_loc.detach(),
        "lovasz_loc": lovasz_loss_loc.detach(),
        "ce_clf": ce_loss_clf.detach(),
        "lovasz_clf": lovasz_loss_clf.detach(),
    }


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

    roi_build = masked_mean(roi, bmask)
    roi_bg = masked_mean(roi, bgmask)
    return {
        "p_mean": float(p.mean().item()),
        "roi_mean": float(roi.mean().item()),
        "roi_min": float(roi.amin().item()),
        "roi_max": float(roi.amax().item()),
        "roi_build": roi_build,
        "roi_bg": roi_bg,
        "roi_gap": roi_build - roi_bg if np.isfinite(roi_build) and np.isfinite(roi_bg) else float("nan"),
    }


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
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    try:
        set_seed(args.seed, rank)
        effective_bs = args.train_batch_size_per_gpu * world_size * args.accumulation_steps
        expected_crops = effective_bs * args.optimizer_steps
        if effective_bs != 16:
            raise ValueError(
                f"Paper-equivalent run requires effective batch size 16, got {effective_bs} = "
                f"{args.train_batch_size_per_gpu} local_bs * {world_size} GPUs * "
                f"{args.accumulation_steps} accumulation."
            )
        if args.max_iters != expected_crops:
            raise ValueError(
                f"Schedule mismatch: max_iters is dataset crop count. For effective_bs={effective_bs} and "
                f"optimizer_steps={args.optimizer_steps}, it must be {expected_crops}, got {args.max_iters}."
            )

        train_names = read_list(args.train_data_list_path)
        val_names = read_list(args.val_data_list_path)
        test_names = read_list(args.test_data_list_path)

        model = DamageFormer_BG(roi_alpha=args.roi_alpha).to(device)
        if args.resume:
            state = torch.load(args.resume, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        train_dataset = MultimodalDamageAssessmentDatset(
            args.train_dataset_path,
            train_names,
            crop_size=args.crop_size,
            max_iters=args.max_iters,
            type="train",
        )
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        ) if world_size > 1 else None
        loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size_per_gpu,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if sampler is not None:
            sampler.set_epoch(0)

        if is_main:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(args.model_param_path) / args.dataset / f"{args.model_type}_{now}"
            save_dir.mkdir(parents=True, exist_ok=True)
            print("Training schedule", flush=True)
            print(f"  GPUs/world_size       : {world_size}", flush=True)
            print(f"  local batch/GPU       : {args.train_batch_size_per_gpu}", flush=True)
            print(f"  accumulation_steps    : {args.accumulation_steps}", flush=True)
            print(f"  effective batch size  : {effective_bs}", flush=True)
            print(f"  optimizer steps       : {args.optimizer_steps}", flush=True)
            print(f"  global training crops : {args.max_iters}", flush=True)
            print(f"  lr / weight_decay     : {args.learning_rate} / {args.weight_decay}", flush=True)
        else:
            save_dir = None

        optimizer.zero_grad(set_to_none=True)
        optimizer_step = 0
        accum_counter = 0
        best_miou = -1.0
        best_round: Dict[str, object] = {}
        running_raw_loss = 0.0

        iterator = tqdm(loader, disable=not is_main, total=len(loader))
        for data in iterator:
            pre, post, labels_loc, labels_clf, _ = data
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            labels_loc = labels_loc.to(device, non_blocking=True).long()
            labels_clf = labels_clf.to(device, non_blocking=True).long()

            sync_now = (accum_counter + 1) == args.accumulation_steps
            sync_context = nullcontext() if (world_size == 1 or sync_now) else model.no_sync()
            with sync_context:
                output_loc, output_clf, aux = model(pre, post, return_aux=True)
                raw_loss, _ = compute_loss(output_loc, output_clf, labels_loc, labels_clf)
                (raw_loss / args.accumulation_steps).backward()

            running_raw_loss += float(raw_loss.detach().item())
            accum_counter += 1
            if accum_counter < args.accumulation_steps:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accum_counter = 0

            if is_main and optimizer_step % args.log_every == 0:
                stats = building_prior_stats(aux, labels_loc)
                avg_loss = running_raw_loss / (args.log_every * args.accumulation_steps)
                print(
                    f"opt_step {optimizer_step}/{args.optimizer_steps} | loss {avg_loss:.4f} | "
                    f"p_build {stats['p_mean']:.4f} | roi {stats['roi_mean']:.4f} "
                    f"[{stats['roi_min']:.4f},{stats['roi_max']:.4f}] | "
                    f"roi_build {stats['roi_build']:.4f} | roi_bg {stats['roi_bg']:.4f} | "
                    f"roi_gap {stats['roi_gap']:.4f}",
                    flush=True,
                )
                running_raw_loss = 0.0

            if optimizer_step % args.eval_every == 0:
                if world_size > 1:
                    dist.barrier()
                if is_main:
                    base_model = unwrap(model)
                    val = evaluate_split(
                        base_model, args.val_dataset_path, val_names, args.eval_batch_size,
                        device, max(1, args.num_workers // 2), "validation"
                    )
                    test = evaluate_split(
                        base_model, args.test_dataset_path, test_names, args.eval_batch_size,
                        device, max(1, args.num_workers // 2), "testing"
                    )
                    if val[3] > best_miou:
                        best_miou = float(val[3])
                        torch.save(base_model.state_dict(), save_dir / "best_model.pth")
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
                    base_model.train()
                if world_size > 1:
                    dist.barrier()

            if optimizer_step >= args.optimizer_steps:
                break

        if optimizer_step != args.optimizer_steps:
            raise RuntimeError(
                f"Training ended at {optimizer_step} optimizer steps, expected {args.optimizer_steps}. "
                "Check dataset length, batch size, and accumulation settings."
            )
        if is_main:
            print("The accuracy of the best round is", best_round, flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
