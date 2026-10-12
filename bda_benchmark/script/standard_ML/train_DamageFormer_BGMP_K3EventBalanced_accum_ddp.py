# -*- coding: utf-8 -*-
"""
S3V6 training for 2 x RTX 4090.

Primary protocol
----------------
- local batch/GPU = 4
- world size = 2
- accumulation = 2
- effective batch = 16
- optimizer steps = 50,000
- global crops = 800,000

Algorithmic differences from S3V5
---------------------------------
1. Main-head protection: training loss is computed from main logits only;
   prototype correction is inference-only by default.
2. Event-balanced prototype update: each event contributes equally to a role
   prototype, after image-level normalization.
3. Optional moderate event-balanced damage loss, controlled by
   --event_loss_alpha. Use 0.0 for clean S3V6-A and 0.25 for S3V6-B.
4. Validation reports event-macro target3 mIoU and saves a dedicated checkpoint
   selected by that validation-only metric.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
for import_root in (PROJECT_ROOT, BDA_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BGMP_K3EventBalanced_DDP import (
    DamageFormer_BGMP_K3EventBalanced_DDP,
)
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S3V6 event-balanced K3 prototype training with 2-GPU DDP"
    )
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
    p.add_argument("--proto_evidence_margin", type=float, default=0.03)
    p.add_argument("--max_pixels_per_image_role", type=int, default=2048)

    p.add_argument(
        "--event_loss_alpha",
        type=float,
        default=0.0,
        help=(
            "Blend coefficient for event-balanced damage CE/Lovasz. "
            "0.0 keeps the original loss; 0.25 is the recommended second ablation."
        ),
    )
    p.add_argument(
        "--event_weight_power",
        type=float,
        default=0.5,
        help="Inverse event-frequency exponent used when event_loss_alpha > 0.",
    )
    p.add_argument("--event_weight_min", type=float, default=0.5)
    p.add_argument("--event_weight_max", type=float, default=2.0)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--model_type",
        type=str,
        default="S3V6_EventBalancedK133_PosthocRescue_2x4090_EBS16_50k",
    )
    p.add_argument("--model_param_path", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        result = [line.strip() for line in f if line.strip()]
    if not result:
        raise RuntimeError(f"Empty data list: {path}")
    return result


def infer_event_name(sample_name: str) -> str:
    stem = Path(str(sample_name).replace("\\", "/")).stem.lower()
    stem = re.sub(r"_(pre|post)_disaster$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


def build_event_vocabulary(train_names: List[str]) -> Tuple[List[str], Dict[str, int]]:
    events = sorted({infer_event_name(name) for name in train_names})
    if not events:
        raise RuntimeError("No event names were parsed from train list.")
    return events, {event: idx for idx, event in enumerate(events)}


def event_ids_from_batch(
    sample_names: List[str],
    event_to_id: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    ids = []
    for name in sample_names:
        event = infer_event_name(name)
        if event not in event_to_id:
            raise KeyError(f"Event {event!r} is not in training event vocabulary.")
        ids.append(event_to_id[event])
    return torch.tensor(ids, dtype=torch.long, device=device)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
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


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def disable_unused_swin_heads(model: nn.Module) -> None:
    for encoder_name in ("encoder_1", "encoder_2"):
        encoder = getattr(model, encoder_name, None)
        swin = getattr(encoder, "swin_transformer", None)
        if swin is None:
            continue
        if hasattr(swin, "norm"):
            swin.norm = nn.Identity()
        if hasattr(swin, "head"):
            swin.head = nn.Identity()


def broadcast_buffers(model: nn.Module, src: int = 0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for buffer in model.buffers():
        dist.broadcast(buffer, src=src)


def load_checkpoint_model(model: nn.Module, path: str) -> Dict[str, object]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        state = obj.get("model", obj.get("state_dict", obj))
    else:
        state = obj
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(clean, strict=True)
    return obj if isinstance(obj, dict) else {}


def build_event_weights(
    train_names: List[str],
    event_to_id: Dict[str, int],
    power: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    counts = Counter(infer_event_name(name) for name in train_names)
    raw = np.zeros(len(event_to_id), dtype=np.float64)
    for event, event_id in event_to_id.items():
        raw[event_id] = 1.0 / (max(counts[event], 1) ** power)
    raw /= raw.mean()
    raw = np.clip(raw, minimum, maximum)
    raw /= raw.mean()
    return torch.tensor(raw, dtype=torch.float32)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)


def compute_loss(
    output_loc: torch.Tensor,
    output_clf: torch.Tensor,
    labels_loc: torch.Tensor,
    labels_clf: torch.Tensor,
    sample_event_weights: torch.Tensor,
    event_loss_alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Original objective plus optional event-balanced damage component."""
    ce_loc = F.cross_entropy(
        output_loc,
        labels_loc,
        ignore_index=255,
    )
    lovasz_loc = L.lovasz_softmax(
        F.softmax(output_loc, dim=1),
        labels_loc,
        ignore=255,
    )

    ce_damage_standard = F.cross_entropy(
        output_clf,
        labels_clf,
        ignore_index=255,
    )
    lovasz_damage_standard = L.lovasz_softmax(
        F.softmax(output_clf, dim=1),
        labels_clf,
        ignore=255,
    )

    if event_loss_alpha > 0.0:
        ce_map = F.cross_entropy(
            output_clf,
            labels_clf,
            ignore_index=255,
            reduction="none",
        )
        per_image_ce = []
        per_image_lovasz = []
        probabilities = F.softmax(output_clf, dim=1)
        for batch_idx in range(output_clf.shape[0]):
            valid = labels_clf[batch_idx] != 255
            if bool(valid.any().item()):
                per_image_ce.append(ce_map[batch_idx][valid].mean())
            else:
                per_image_ce.append(output_clf[batch_idx].sum() * 0.0)
            per_image_lovasz.append(
                L.lovasz_softmax(
                    probabilities[batch_idx : batch_idx + 1],
                    labels_clf[batch_idx : batch_idx + 1],
                    ignore=255,
                )
            )
        per_image_ce_tensor = torch.stack(per_image_ce)
        per_image_lovasz_tensor = torch.stack(per_image_lovasz)
        ce_damage_balanced = weighted_mean(
            per_image_ce_tensor,
            sample_event_weights,
        )
        lovasz_damage_balanced = weighted_mean(
            per_image_lovasz_tensor,
            sample_event_weights,
        )
        alpha = float(event_loss_alpha)
        ce_damage = (
            (1.0 - alpha) * ce_damage_standard
            + alpha * ce_damage_balanced
        )
        lovasz_damage = (
            (1.0 - alpha) * lovasz_damage_standard
            + alpha * lovasz_damage_balanced
        )
    else:
        ce_damage = ce_damage_standard
        lovasz_damage = lovasz_damage_standard

    total = (
        ce_loc
        + ce_damage
        + 0.75 * lovasz_damage
        + 0.5 * lovasz_loc
    )
    stats = {
        "ce_loc": float(ce_loc.detach().item()),
        "lovasz_loc": float(lovasz_loc.detach().item()),
        "ce_damage": float(ce_damage.detach().item()),
        "lovasz_damage": float(lovasz_damage.detach().item()),
    }
    return total, stats


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def fast_hist(gt: np.ndarray, pred: np.ndarray, classes: int = 4) -> np.ndarray:
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid = (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    index = classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(index, minlength=classes * classes).reshape(classes, classes)


def iou_from_hist(hist: np.ndarray) -> np.ndarray:
    intersection = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - intersection
    result = np.full(hist.shape[0], np.nan, dtype=np.float64)
    valid = union > 0
    result[valid] = intersection[valid] / union[valid]
    return result


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset_path: str,
    data_names: List[str],
    eval_batch_size: int,
    device: torch.device,
    num_workers: int,
    split_name: str,
) -> Dict[str, object]:
    print(f"--------- starting {split_name} ---------", flush=True)
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    event_hist: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((4, 4), dtype=np.int64)
    )

    dataset = MultimodalDamageAssessmentDatset(
        dataset_path,
        data_names,
        1024,
        None,
        "test",
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    for pre, post, labels_loc, labels_clf, sample_names in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc_t = labels_loc.to(device, non_blocking=True).long()
        labels_clf_t = labels_clf.to(device, non_blocking=True).long()

        output_loc, output_clf = model(
            pre,
            post,
            damage_target=None,
            event_ids=None,
            current_iter=None,
            return_aux=False,
        )

        pred_loc = output_loc.argmax(dim=1).cpu().numpy()
        pred_clf = output_clf.argmax(dim=1).cpu().numpy()
        gt_loc = labels_loc_t.cpu().numpy()
        gt_clf = labels_clf_t.cpu().numpy()

        evaluator_loc.add_batch(gt_loc, pred_loc)
        evaluator_clf.add_batch(
            gt_clf[gt_loc > 0],
            pred_clf[gt_loc > 0],
        )
        evaluator_total.add_batch(gt_clf, pred_clf)

        for batch_idx, sample_name in enumerate(sample_names):
            event = infer_event_name(sample_name)
            event_hist[event] += fast_hist(
                gt_clf[batch_idx],
                pred_clf[batch_idx],
                classes=4,
            )

    loc_f1 = evaluator_loc.Pixel_F1_score()
    damage_f1 = evaluator_clf.Damage_F1_score()
    harmonic_f1 = len(damage_f1) / np.sum(1.0 / damage_f1)
    oa = evaluator_total.Pixel_Accuracy()
    ious = evaluator_total.Intersection_over_Union()
    miou = evaluator_total.Mean_Intersection_over_Union()

    event_target3 = {}
    for event, hist in event_hist.items():
        iou = iou_from_hist(hist)
        present_damage = iou[1:4]
        present_damage = present_damage[~np.isnan(present_damage)]
        event_target3[event] = (
            float(present_damage.mean())
            if present_damage.size > 0
            else float("nan")
        )
    valid_event_scores = [
        value for value in event_target3.values() if np.isfinite(value)
    ]
    event_macro_target3 = (
        float(np.mean(valid_event_scores))
        if valid_event_scores
        else float("nan")
    )
    finite_event_items = {
        event: value
        for event, value in event_target3.items()
        if np.isfinite(value)
    }
    worst_event = (
        min(finite_event_items, key=finite_event_items.get)
        if finite_event_items
        else "N/A"
    )
    worst_event_target3 = (
        finite_event_items[worst_event]
        if worst_event != "N/A"
        else float("nan")
    )

    result = {
        "loc_f1": float(loc_f1),
        "damage_hf1": float(harmonic_f1),
        "oa": float(oa),
        "miou": float(miou),
        "ious": np.asarray(ious).tolist(),
        "event_macro_target3": float(event_macro_target3),
        "worst_event": worst_event,
        "worst_event_target3": float(worst_event_target3),
        "event_target3": event_target3,
    }

    print(
        f"{split_name}: locF1={100*loc_f1:.4f}, "
        f"damageHF1={100*harmonic_f1:.4f}, OA={100*oa:.4f}, "
        f"mIoU={100*miou:.4f}, IoU={100*np.asarray(ious)}, "
        f"eventMacroTarget3={100*event_macro_target3:.4f}, "
        f"worst={worst_event}:{100*worst_event_target3:.4f}",
        flush=True,
    )
    return result


@torch.no_grad()
def prototype_diagnostics(model: nn.Module) -> Dict[str, object]:
    memory = model.damage_proto_memory
    proto = F.normalize(memory.prototypes.detach(), dim=-1)
    valid = memory.prototype_valid.detach()
    norms = memory.prototypes.detach()[valid].norm(dim=-1)

    within = {}
    names = ["Intact", "Damaged", "Destroyed"]
    for class_idx, name in enumerate(names):
        p = proto[class_idx, valid[class_idx]]
        within[name] = (p @ p.t()).cpu().tolist()

    centres = []
    for class_idx in range(3):
        p = proto[class_idx, valid[class_idx]]
        centres.append(F.normalize(p.mean(dim=0), dim=0))
    centre = torch.stack(centres)

    return {
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "within_class_cosine": within,
        "interclass_cosine": (centre @ centre.t()).cpu().tolist(),
        "event_usage_ratio": (
            memory.prototype_event_updates
            * valid.float()
            / (
                memory.prototype_event_updates
                * valid.float()
            ).sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        ).cpu().tolist(),
        "image_usage_ratio": (
            memory.prototype_image_updates
            * valid.float()
            / (
                memory.prototype_image_updates
                * valid.float()
            ).sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        ).cpu().tolist(),
        "pixel_usage_ratio": (
            memory.prototype_pixel_observations
            * valid.float()
            / (
                memory.prototype_pixel_observations
                * valid.float()
            ).sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        ).cpu().tolist(),
    }


def save_training_state(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    optimizer_step: int,
    best_values: Dict[str, float],
    args: argparse.Namespace,
    event_vocabulary: List[str],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "best_values": best_values,
            "args": vars(args),
            "event_vocabulary": event_vocabulary,
        },
        path,
    )


def maybe_save_best(
    metric_name: str,
    metric_value: float,
    best_values: Dict[str, float],
    save_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    optimizer_step: int,
    args: argparse.Namespace,
    event_vocabulary: List[str],
    val_result: Dict[str, object],
) -> bool:
    if not np.isfinite(metric_value):
        return False
    if metric_value <= best_values[metric_name]:
        return False

    best_values[metric_name] = float(metric_value)
    torch.save(
        model.state_dict(),
        save_dir / f"best_{metric_name}.pth",
    )
    save_training_state(
        save_dir / f"best_{metric_name}_training_state.pth",
        model,
        optimizer,
        optimizer_step,
        best_values,
        args,
        event_vocabulary,
    )
    metadata = {
        "selection_metric": metric_name,
        "selection_value": float(metric_value),
        "optimizer_step": int(optimizer_step),
        "validation": val_result,
    }
    (save_dir / f"best_{metric_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"NEW BEST {metric_name}: {metric_value:.6f} "
        f"at step {optimizer_step}",
        flush=True,
    )
    return True


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0

    try:
        if world_size != 2:
            raise ValueError(
                f"This script expects exactly 2 GPUs, got WORLD_SIZE={world_size}."
            )
        if not 0.0 <= args.event_loss_alpha <= 1.0:
            raise ValueError("event_loss_alpha must be in [0,1].")

        set_seed(args.seed, rank)

        effective_bs = (
            args.train_batch_size_per_gpu
            * world_size
            * args.accumulation_steps
        )
        expected_crops = effective_bs * args.optimizer_steps
        if effective_bs != 16:
            raise ValueError(
                "Expected EBS=16, got "
                f"{args.train_batch_size_per_gpu} x {world_size} x "
                f"{args.accumulation_steps} = {effective_bs}."
            )
        if args.max_iters != expected_crops:
            raise ValueError(
                f"max_iters must be {expected_crops}, got {args.max_iters}."
            )

        train_names = read_list(args.train_data_list_path)
        val_names = read_list(args.val_data_list_path)
        test_names = read_list(args.test_data_list_path)
        event_vocabulary, event_to_id = build_event_vocabulary(train_names)
        # Persist this resolved value inside checkpoint args for strict evaluation.
        args.num_events = len(event_vocabulary)
        event_weights_cpu = build_event_weights(
            train_names,
            event_to_id,
            power=args.event_weight_power,
            minimum=args.event_weight_min,
            maximum=args.event_weight_max,
        )
        event_weights = event_weights_cpu.to(device)

        # Memory updates once per micro-batch. Convert the requested optimizer
        # step retention to an equivalent micro-batch retention.
        proto_micro_momentum = (
            args.proto_step_momentum
            ** (1.0 / args.accumulation_steps)
        )

        model = DamageFormer_BGMP_K3EventBalanced_DDP(
            roi_alpha=args.roi_alpha,
            detach_building_prior=True,
            use_pretrained_swin=True,
            proto_dim=args.proto_dim,
            num_events=len(event_vocabulary),
            proto_momentum=proto_micro_momentum,
            proto_tau=args.proto_tau,
            proto_warmup_iters=args.proto_warmup_iters,
            core_confidence=args.core_confidence,
            lambda_damaged=args.lambda_damaged,
            lambda_destroyed=args.lambda_destroyed,
            evidence_clip=args.proto_evidence_clip,
            evidence_margin=args.proto_evidence_margin,
            apply_proto_during_training=False,
            rescue_only_main_intact=True,
            max_pixels_per_image_role=args.max_pixels_per_image_role,
        ).to(device)
        disable_unused_swin_heads(model)

        optimizer_step = 0
        best_values = {
            "val_miou": -1.0,
            "val_damage_hf1": -1.0,
            "val_event_macro_target3": -1.0,
        }
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

        if resume_obj:
            if "optimizer" in resume_obj:
                optimizer.load_state_dict(resume_obj["optimizer"])
            optimizer_step = int(resume_obj.get("optimizer_step", 0))
            stored_best = resume_obj.get("best_values", {})
            if isinstance(stored_best, dict):
                best_values.update(stored_best)

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
            (save_dir / "event_vocabulary.json").write_text(
                json.dumps(
                    {
                        "event_vocabulary": event_vocabulary,
                        "train_event_counts": dict(
                            Counter(
                                infer_event_name(name)
                                for name in train_names
                            )
                        ),
                        "event_weights": {
                            event: float(event_weights_cpu[event_to_id[event]])
                            for event in event_vocabulary
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print("Training schedule", flush=True)
            print(f"  GPUs/world_size          : {world_size}", flush=True)
            print(
                f"  local batch/GPU          : {args.train_batch_size_per_gpu}",
                flush=True,
            )
            print(
                f"  accumulation_steps       : {args.accumulation_steps}",
                flush=True,
            )
            print(f"  effective batch size     : {effective_bs}", flush=True)
            print(f"  optimizer steps          : {args.optimizer_steps}", flush=True)
            print(f"  global training crops    : {args.max_iters}", flush=True)
            print(f"  event vocabulary ({len(event_vocabulary)}): {event_vocabulary}", flush=True)
            print(
                "  prototype update         : image-normalized + equal-event EMA",
                flush=True,
            )
            print(
                "  prototype train effect   : OFF (main-head protected)",
                flush=True,
            )
            print(
                "  inference rescue         : main-predicted Intact only",
                flush=True,
            )
            print(
                f"  evidence margin          : {args.proto_evidence_margin}",
                flush=True,
            )
            print(
                f"  event loss alpha         : {args.event_loss_alpha}",
                flush=True,
            )
            print(
                f"  proto EMA step / micro   : "
                f"{args.proto_step_momentum:.8f} / "
                f"{proto_micro_momentum:.8f}",
                flush=True,
            )
        else:
            save_dir = None

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        running_loss = 0.0
        last_loss_stats: Dict[str, float] = {}

        iterator = tqdm(loader, total=len(loader), disable=not is_main)
        for pre, post, labels_loc, labels_clf, sample_names in iterator:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            labels_loc = labels_loc.to(device, non_blocking=True).long()
            labels_clf = labels_clf.to(device, non_blocking=True).long()
            event_ids = event_ids_from_batch(
                list(sample_names),
                event_to_id,
                device,
            )
            sample_weights = event_weights[event_ids]

            sync_now = (accum_counter + 1) == args.accumulation_steps
            sync_context = nullcontext() if sync_now else ddp_model.no_sync()
            with sync_context:
                output_loc, output_clf, aux = ddp_model(
                    pre,
                    post,
                    damage_target=labels_clf,
                    event_ids=event_ids,
                    current_iter=optimizer_step,
                    return_aux=True,
                )
                # Because apply_proto_during_training=False, output_clf is the
                # protected main classifier output.
                raw_loss, last_loss_stats = compute_loss(
                    output_loc,
                    output_clf,
                    labels_loc,
                    labels_clf,
                    sample_event_weights=sample_weights,
                    event_loss_alpha=args.event_loss_alpha,
                )
                (raw_loss / args.accumulation_steps).backward()

            running_loss += float(raw_loss.detach().item())
            accum_counter += 1
            if accum_counter < args.accumulation_steps:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accum_counter = 0

            if optimizer_step % args.log_every == 0:
                local_average = running_loss / (
                    args.log_every * args.accumulation_steps
                )
                average_loss = reduce_mean_scalar(local_average, device)
                running_loss = 0.0
                if is_main:
                    base_model = unwrap(ddp_model)
                    diagnostics = prototype_diagnostics(base_model)
                    print(
                        f"opt_step {optimizer_step}/{args.optimizer_steps} | "
                        f"loss {average_loss:.4f} | "
                        f"ceD {last_loss_stats.get('ce_damage', float('nan')):.4f} | "
                        f"lovD {last_loss_stats.get('lovasz_damage', float('nan')):.4f} | "
                        f"proto_ready {aux['proto_active']} | "
                        f"correction_in_train {aux['proto_correction_applied']} | "
                        f"event_usage {aux['proto_event_usage_ratio'].detach().cpu().tolist()} | "
                        f"image_usage {aux['proto_image_usage_ratio'].detach().cpu().tolist()} | "
                        f"norm [{diagnostics['norm_min']:.4f},"
                        f"{diagnostics['norm_max']:.4f}]",
                        flush=True,
                    )

            if optimizer_step % args.eval_every == 0:
                dist.barrier()
                if is_main:
                    base_model = unwrap(ddp_model)
                    val_result = evaluate_split(
                        base_model,
                        args.val_dataset_path,
                        val_names,
                        args.eval_batch_size,
                        device,
                        max(1, args.num_workers // 2),
                        "validation",
                    )

                    save_training_state(
                        save_dir / "latest.pth",
                        base_model,
                        optimizer,
                        optimizer_step,
                        best_values,
                        args,
                        event_vocabulary,
                    )

                    maybe_save_best(
                        "val_miou",
                        float(val_result["miou"]),
                        best_values,
                        save_dir,
                        base_model,
                        optimizer,
                        optimizer_step,
                        args,
                        event_vocabulary,
                        val_result,
                    )
                    maybe_save_best(
                        "val_damage_hf1",
                        float(val_result["damage_hf1"]),
                        best_values,
                        save_dir,
                        base_model,
                        optimizer,
                        optimizer_step,
                        args,
                        event_vocabulary,
                        val_result,
                    )
                    maybe_save_best(
                        "val_event_macro_target3",
                        float(val_result["event_macro_target3"]),
                        best_values,
                        save_dir,
                        base_model,
                        optimizer,
                        optimizer_step,
                        args,
                        event_vocabulary,
                        val_result,
                    )
                    (save_dir / "prototype_diagnostics_latest.json").write_text(
                        json.dumps(
                            prototype_diagnostics(base_model),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    base_model.train()
                dist.barrier()

            if optimizer_step >= args.optimizer_steps:
                break

        if optimizer_step != args.optimizer_steps:
            raise RuntimeError(
                f"Training ended at {optimizer_step}; expected "
                f"{args.optimizer_steps}."
            )

        dist.barrier()
        if is_main:
            base_model = unwrap(ddp_model)
            # Primary final report: checkpoint selected only by validation
            # event-macro damage performance, not by test results.
            primary = save_dir / "best_val_event_macro_target3.pth"
            if not primary.exists():
                primary = save_dir / "best_val_miou.pth"
            base_model.load_state_dict(
                torch.load(primary, map_location="cpu"),
                strict=True,
            )
            final_test = evaluate_split(
                base_model,
                args.test_dataset_path,
                test_names,
                args.eval_batch_size,
                device,
                max(1, args.num_workers // 2),
                "final testing from validation event-macro checkpoint",
            )
            summary = {
                "primary_checkpoint": str(primary),
                "best_validation_values": best_values,
                "final_test": final_test,
                "prototype_diagnostics": prototype_diagnostics(base_model),
            }
            (save_dir / "final_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("Final summary:", summary, flush=True)
        dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
