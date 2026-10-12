# -*- coding: utf-8 -*-
"""S3V5 role-aware K=[1,3,3] prototype training with 2-GPU DDP.

Recommended 2x RTX 4090 schedule:
  local batch/GPU=4, world_size=2, accumulation=2 -> EBS=16
  optimizer_steps=50000, global crops=800000

Prototype memory is synchronized inside the model by all-reducing each role's
normalized feature sum and count before EMA. This is required because prototype
buffers are not synchronized by gradient all-reduce.
"""
from __future__ import annotations

import argparse
import json
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

BDA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BDA_ROOT.parent

# Support both import styles used in the BRIGHT repository:
#   from dataset.xxx import ...
#   from bda_benchmark.dataset import ...
for import_root in (PROJECT_ROOT, BDA_ROOT):
    import_root = str(import_root)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BGMP_K3Gated_DDP import DamageFormer_BGMP_K3Gated_DDP
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3V5 K3Gated 2-GPU DDP")
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
    p.add_argument("--max_iters", type=int, default=800000)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--roi_alpha", type=float, default=0.1)
    p.add_argument("--proto_dim", type=int, default=256)
    p.add_argument("--proto_step_momentum", type=float, default=0.99)
    p.add_argument("--proto_tau", type=float, default=0.1)
    p.add_argument("--proto_warmup_iters", type=int, default=5000)
    p.add_argument("--core_confidence", type=float, default=0.70)
    p.add_argument("--lambda_damaged", type=float, default=0.05)
    p.add_argument("--lambda_destroyed", type=float, default=0.05)
    p.add_argument("--proto_evidence_clip", type=float, default=4.0)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--model_type",
        type=str,
        default="S3V5_K133_HardGatedLogit_2x4090_EBS16_50k",
    )
    p.add_argument("--model_param_path", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_test_during_training", action="store_true")
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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


def disable_unused_swin_heads(model: torch.nn.Module) -> None:
    """Remove torchvision Swin norm/head parameters unused by the extractor."""
    for encoder_name in ("encoder_1", "encoder_2"):
        encoder = getattr(model, encoder_name, None)
        swin = getattr(encoder, "swin_transformer", None)
        if swin is not None:
            if hasattr(swin, "norm"):
                swin.norm = nn.Identity()
            if hasattr(swin, "head"):
                swin.head = nn.Identity()


def broadcast_buffers(model: torch.nn.Module, src: int = 0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for buffer in model.buffers():
        dist.broadcast(buffer, src=src)


def load_checkpoint_model(model: torch.nn.Module, path: str) -> Dict[str, object]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        state = obj.get("model", obj.get("state_dict", obj))
    else:
        state = obj
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return obj if isinstance(obj, dict) else {}


def compute_loss(
    output_loc: torch.Tensor,
    output_clf: torch.Tensor,
    labels_loc: torch.Tensor,
    labels_clf: torch.Tensor,
) -> torch.Tensor:
    ce_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
    lovasz_loc = L.lovasz_softmax(
        F.softmax(output_loc, dim=1), labels_loc, ignore=255
    )
    ce_clf = F.cross_entropy(output_clf, labels_clf)
    lovasz_clf = L.lovasz_softmax(
        F.softmax(output_clf, dim=1), labels_clf, ignore=255
    )
    return ce_loc + ce_clf + 0.75 * lovasz_clf + 0.5 * lovasz_loc


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


@torch.no_grad()
def building_prior_stats(
    aux: Dict[str, object], labels_loc: torch.Tensor
) -> Dict[str, float]:
    roi = aux["roi_weight"].detach()
    p = aux["p_building"].detach()
    gt = F.interpolate(
        labels_loc.unsqueeze(1).float(), size=roi.shape[-2:], mode="nearest"
    )[:, 0]
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
        "roi_gap": rb - rbg
        if np.isfinite(rb) and np.isfinite(rbg)
        else float("nan"),
    }


@torch.no_grad()
def prototype_class_accuracy(
    aux: Dict[str, object], labels_clf: torch.Tensor
) -> float:
    sim = aux.get("proto_class_similarity")
    if sim is None:
        return float("nan")
    feat = aux["final_feat_fused_proto"]
    target = F.interpolate(
        labels_clf.unsqueeze(1).float(),
        size=feat.shape[-2:],
        mode="nearest",
    )[:, 0].long().reshape(-1)
    valid = (target >= 1) & (target <= 3)
    if not bool(valid.any().item()):
        return float("nan")
    pred = sim.argmax(dim=1) + 1
    return float((pred[valid] == target[valid]).float().mean().item())


@torch.no_grad()
def prototype_diagnostics(
    model: torch.nn.Module, previous: torch.Tensor | None
) -> Tuple[Dict[str, object], torch.Tensor]:
    memory = model.damage_proto_memory
    proto = memory.prototypes.detach()
    valid = memory.prototype_valid.detach()
    valid_proto = proto[valid]
    norms = valid_proto.norm(dim=-1)
    current = proto.clone()
    drift = (
        float("nan")
        if previous is None
        else float((current - previous).norm(dim=-1)[valid].mean().item())
    )

    class_means = []
    for class_idx in range(proto.shape[0]):
        p = F.normalize(proto[class_idx, valid[class_idx]], dim=-1)
        class_means.append(F.normalize(p.mean(dim=0), dim=0))
    class_mean = torch.stack(class_means, dim=0)
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
    print(f"--------- starting {split_name} ---------", flush=True)
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    dataset = MultimodalDamageAssessmentDatset(
        dataset_path, data_names, 1024, None, "test"
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model.eval()
    for pre, post, labels_loc, labels_clf, _ in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc_t = labels_loc.to(device, non_blocking=True).long()
        labels_clf_t = labels_clf.to(device, non_blocking=True).long()
        output_loc, output_clf = model(
            pre,
            post,
            damage_target=None,
            current_iter=None,
            return_aux=False,
        )
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
    print(
        f"{split_name}: locF1={100*loc_f1:.4f}, "
        f"damageF1={100*harmonic_f1:.4f}, OA={100*oa:.4f}, "
        f"mIoU={100*miou:.4f}, IoU={100*ious}",
        flush=True,
    )
    return loc_f1, harmonic_f1, oa, miou, ious


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    best_miou: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "best_miou": best_miou,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0

    try:
        if world_size != 2:
            raise ValueError(
                f"This script expects exactly 2 GPUs, got WORLD_SIZE={world_size}"
            )
        set_seed(args.seed, rank)

        effective_bs = (
            args.train_batch_size_per_gpu
            * world_size
            * args.accumulation_steps
        )
        expected_crops = effective_bs * args.optimizer_steps
        if effective_bs != 16:
            raise ValueError(
                "Paper-equivalent EBS must be 16, got "
                f"{args.train_batch_size_per_gpu} x {world_size} x "
                f"{args.accumulation_steps} = {effective_bs}"
            )
        if args.max_iters != expected_crops:
            raise ValueError(
                f"max_iters must be {expected_crops}, got {args.max_iters}"
            )
        if not 0.0 < args.proto_step_momentum < 1.0:
            raise ValueError("proto_step_momentum must be in (0,1)")

        proto_micro_momentum = args.proto_step_momentum ** (
            1.0 / args.accumulation_steps
        )

        train_names = read_list(args.train_data_list_path)
        val_names = read_list(args.val_data_list_path)
        test_names = read_list(args.test_data_list_path)

        model = DamageFormer_BGMP_K3Gated_DDP(
            roi_alpha=args.roi_alpha,
            detach_building_prior=True,
            use_pretrained_swin=True,
            proto_dim=args.proto_dim,
            proto_momentum=proto_micro_momentum,
            proto_tau=args.proto_tau,
            proto_warmup_iters=args.proto_warmup_iters,
            core_confidence=args.core_confidence,
            lambda_damaged=args.lambda_damaged,
            lambda_destroyed=args.lambda_destroyed,
            evidence_clip=args.proto_evidence_clip,
        ).to(device)
        disable_unused_swin_heads(model)

        resume_obj: Dict[str, object] = {}
        if args.resume:
            resume_obj = load_checkpoint_model(model, args.resume)

        broadcast_buffers(model, src=0)
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        optimizer = optim.AdamW(
            ddp_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        optimizer_step = 0
        best_miou = -1.0
        if resume_obj:
            if "optimizer" in resume_obj:
                optimizer.load_state_dict(resume_obj["optimizer"])
            optimizer_step = int(resume_obj.get("optimizer_step", 0))
            best_miou = float(resume_obj.get("best_miou", -1.0))

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
        )
        loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size_per_gpu,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        sampler.set_epoch(0)

        if is_main:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = (
                Path(args.model_param_path)
                / args.dataset
                / f"{args.model_type}_{now}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            print("Training schedule", flush=True)
            print(f"  GPUs/world_size         : {world_size}", flush=True)
            print(
                f"  local batch/GPU         : {args.train_batch_size_per_gpu}",
                flush=True,
            )
            print(
                f"  accumulation_steps      : {args.accumulation_steps}",
                flush=True,
            )
            print(f"  effective batch size    : {effective_bs}", flush=True)
            print(f"  optimizer steps         : {args.optimizer_steps}", flush=True)
            print(f"  global training crops   : {args.max_iters}", flush=True)
            print(
                f"  lr / weight_decay       : "
                f"{args.learning_rate} / {args.weight_decay}",
                flush=True,
            )
            print("  proto K roles           : [1,3,3]", flush=True)
            print(
                f"  proto dim / tau         : {args.proto_dim} / {args.proto_tau}",
                flush=True,
            )
            print(
                f"  proto EMA step / micro  : "
                f"{args.proto_step_momentum:.8f} / "
                f"{proto_micro_momentum:.8f}",
                flush=True,
            )
            print(
                "  prototype synchronization: NCCL all-reduce per role update",
                flush=True,
            )
            print(
                f"  correction lambdas      : "
                f"{args.lambda_damaged} / {args.lambda_destroyed}",
                flush=True,
            )
        else:
            save_dir = None

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        running_raw_loss = 0.0
        previous_proto: torch.Tensor | None = None
        best_round: Dict[str, object] = {}

        iterator = tqdm(loader, total=len(loader), disable=not is_main)
        for data in iterator:
            pre, post, labels_loc, labels_clf, _ = data
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            labels_loc = labels_loc.to(device, non_blocking=True).long()
            labels_clf = labels_clf.to(device, non_blocking=True).long()

            sync_now = (accum_counter + 1) == args.accumulation_steps
            sync_context = nullcontext() if sync_now else ddp_model.no_sync()
            with sync_context:
                output_loc, output_clf, aux = ddp_model(
                    pre,
                    post,
                    damage_target=labels_clf,
                    current_iter=optimizer_step,
                    return_aux=True,
                )
                raw_loss = compute_loss(
                    output_loc, output_clf, labels_loc, labels_clf
                )
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
                local_avg = running_raw_loss / (
                    args.log_every * args.accumulation_steps
                )
                avg_loss = reduce_mean_scalar(local_avg, device)
                running_raw_loss = 0.0
                if is_main:
                    base_model = unwrap(ddp_model)
                    bg = building_prior_stats(aux, labels_loc)
                    proto_acc = prototype_class_accuracy(aux, labels_clf)
                    diag, previous_proto = prototype_diagnostics(
                        base_model, previous_proto
                    )
                    print(
                        f"opt_step {optimizer_step}/{args.optimizer_steps} | "
                        f"loss {avg_loss:.4f} | roi_gap {bg['roi_gap']:.4f} | "
                        f"proto_active {aux['proto_active']} | "
                        f"proto_init "
                        f"{aux['proto_initialized'].detach().cpu().tolist()} | "
                        f"lambda {float(aux['proto_gamma']):.4f} | "
                        f"proto_cls_acc {proto_acc:.4f} | "
                        f"proto_norm [{diag['norm_min']:.4f},"
                        f"{diag['norm_max']:.4f}] | "
                        f"proto_drift {diag['drift']:.6f} | "
                        f"usage "
                        f"{aux['proto_usage_ratio'].detach().cpu().tolist()}",
                        flush=True,
                    )
                    if optimizer_step % args.eval_every == 0:
                        print(
                            f"prototype inter-class cosine="
                            f"{diag['interclass_cos']}",
                            flush=True,
                        )

            if optimizer_step % args.eval_every == 0:
                dist.barrier()
                if is_main:
                    base_model = unwrap(ddp_model)
                    val = evaluate_split(
                        base_model,
                        args.val_dataset_path,
                        val_names,
                        args.eval_batch_size,
                        device,
                        max(1, args.num_workers // 2),
                        "validation",
                    )
                    test = None
                    if args.eval_test_during_training:
                        test = evaluate_split(
                            base_model,
                            args.test_dataset_path,
                            test_names,
                            args.eval_batch_size,
                            device,
                            max(1, args.num_workers // 2),
                            "testing",
                        )

                    save_training_state(
                        save_dir / "latest.pth",
                        base_model,
                        optimizer,
                        optimizer_step,
                        best_miou,
                        args,
                    )
                    if val[3] > best_miou:
                        best_miou = float(val[3])
                        torch.save(
                            base_model.state_dict(),
                            save_dir / "best_model.pth",
                        )
                        save_training_state(
                            save_dir / "best_training_state.pth",
                            base_model,
                            optimizer,
                            optimizer_step,
                            best_miou,
                            args,
                        )
                        best_round = {
                            "best optimizer step": optimizer_step,
                            "loc f1 (val)": val[0] * 100,
                            "clf f1 (val)": val[1] * 100,
                            "OA (val)": val[2] * 100,
                            "mIoU (val)": val[3] * 100,
                            "sub class IoU (val)": (val[4] * 100).tolist(),
                        }
                        if test is not None:
                            best_round.update(
                                {
                                    "loc f1 (test)": test[0] * 100,
                                    "clf f1 (test)": test[1] * 100,
                                    "OA (test)": test[2] * 100,
                                    "mIoU (test)": test[3] * 100,
                                    "sub class IoU (test)": (
                                        test[4] * 100
                                    ).tolist(),
                                }
                            )
                        (save_dir / "best_metrics.json").write_text(
                            json.dumps(
                                best_round, indent=2, ensure_ascii=False
                            ),
                            encoding="utf-8",
                        )
                    base_model.train()
                dist.barrier()

            if optimizer_step >= args.optimizer_steps:
                break

        if optimizer_step != args.optimizer_steps:
            raise RuntimeError(
                f"Training ended at {optimizer_step}, expected "
                f"{args.optimizer_steps}. Check max_iters/batch/ACC."
            )

        dist.barrier()
        if is_main:
            base_model = unwrap(ddp_model)
            best_path = save_dir / "best_model.pth"
            if best_path.exists():
                base_model.load_state_dict(
                    torch.load(best_path, map_location="cpu"), strict=True
                )
            final_test = evaluate_split(
                base_model,
                args.test_dataset_path,
                test_names,
                args.eval_batch_size,
                device,
                max(1, args.num_workers // 2),
                "final testing from best validation checkpoint",
            )
            final_summary = {
                **best_round,
                "final test loc f1": final_test[0] * 100,
                "final test clf f1": final_test[1] * 100,
                "final test OA": final_test[2] * 100,
                "final test mIoU": final_test[3] * 100,
                "final test IoU": (final_test[4] * 100).tolist(),
            }
            (save_dir / "final_summary.json").write_text(
                json.dumps(final_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("Final summary:", final_summary, flush=True)
        dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
