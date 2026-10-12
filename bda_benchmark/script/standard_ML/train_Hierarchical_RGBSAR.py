#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified trainer for S5/S6/S7/S8 hierarchical RGB-SAR DamageFormer on BRIGHT.

Key protocol choices
--------------------
* --max_steps means OPTIMIZER STEPS, not micro-batch iterations.
* --train_batch_size is per-process; global EBS = batch * world_size * accumulation.
* Main supervision is CE + Lovasz for localization and damage segmentation.
* S5/S6/S7 add no auxiliary alignment loss.
* S8 adds warmup/ramped intact-only Stage3/Stage4 cosine alignment.
* Best checkpoint is selected by valid_mIoU = mean IoU(intact, damaged, destroyed).
* torchrun/DDP is supported automatically when LOCAL_RANK is present.
* CUDA AMP is configurable: fp16 uses GradScaler; bf16 uses autocast without scaling.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# -----------------------------------------------------------------------------
# Repository imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (SCRIPT_DIR / "../../..").resolve()
BDA_ROOT = ROOT / "bda_benchmark"
for p in (str(ROOT), str(BDA_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def import_model(experiment: str):
    experiment = experiment.upper()
    if experiment not in ("S5", "S6", "S7", "S8"):
        raise ValueError(f"Unsupported experiment {experiment!r}; choose S5/S6/S7/S8.")
    try:
        from model.DamageFormer_Hierarchical_RGBSAR import DamageFormer_Hierarchical_RGBSAR
    except ImportError:
        from DamageFormer_Hierarchical_RGBSAR import DamageFormer_Hierarchical_RGBSAR
    return DamageFormer_Hierarchical_RGBSAR


def import_dataset():
    """Import BRIGHT dataset loader without hiding the real missing dependency."""
    try:
        from dataset.make_data_loader import MultimodalDamageAssessmentDatset
        return MultimodalDamageAssessmentDatset
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or "unknown"

        install_hints = {
            "imageio": (
                "imageio",
                "python -m pip install imageio",
            ),
            "cv2": (
                "OpenCV Python binding (cv2)",
                "python -m pip install opencv-python-headless",
            ),
        }

        if missing in install_hints:
            package_name, install_cmd = install_hints[missing]
            raise ImportError(
                f"BRIGHT dataset loader requires {package_name}, but module {missing!r} "
                "cannot be imported in the CURRENT Python environment.\n"
                f"Python executable: {sys.executable}\n"
                f"Install with:\n  {install_cmd}\n"
                "Then verify the whole BRIGHT loader before training with:\n"
                "  python script/standard_ML/check_BRIGHT_env.py"
            ) from exc

        if missing in {"dataset", "dataset.make_data_loader"}:
            raise ImportError(
                "Cannot find BRIGHT dataset.make_data_loader. "
                f"Resolved ROOT={ROOT}, BDA_ROOT={BDA_ROOT}, Python={sys.executable}. "
                "Run from BRIGHT-master/bda_benchmark or check the script location."
            ) from exc

        raise ImportError(
            "A dependency required by dataset.make_data_loader is missing: "
            f"{missing!r}. Python executable: {sys.executable}. Original error: {exc}"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Failed to import MultimodalDamageAssessmentDatset from "
            "dataset.make_data_loader. "
            f"Resolved ROOT={ROOT}, BDA_ROOT={BDA_ROOT}, Python={sys.executable}. "
            f"Original error: {exc}"
        ) from exc


# -----------------------------------------------------------------------------
# Distributed/runtime utilities
# -----------------------------------------------------------------------------
def setup_runtime(requested_device: str) -> Tuple[torch.device, bool, int, int, int]:
    if "LOCAL_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun/DDP requested but CUDA is unavailable.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return torch.device(f"cuda:{local_rank}"), True, rank, world_size, local_rank

    requested = requested_device.lower().strip()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cpu":
        device = torch.device("cpu")
    elif requested == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested.startswith("cuda:"):
        device = torch.device(requested)
    else:
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")
    return device, False, 0, 1, 0


def cleanup_runtime(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_random_seed(seed: int, rank: int, deterministic: bool) -> torch.Generator:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    generator = torch.Generator()
    generator.manual_seed(actual_seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_dataset(dataset_path: str, data_list_path: str, image_size: int, split: str):
    DatasetClass = import_dataset()
    return DatasetClass(dataset_path, read_list(data_list_path), image_size, None, split)


def parse_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        def pick(keys: Iterable[str]):
            for key in keys:
                if key in batch:
                    return batch[key]
            raise KeyError(f"Cannot find keys {list(keys)} in batch keys {list(batch.keys())}")

        pre = pick(["pre_img", "pre_data", "pre", "image_pre"])
        post = pick(["post_img", "post_data", "post", "image_post"])
        loc = pick(["label_loc", "labels_loc", "loc", "building_mask"])
        dam = pick(["label_dam", "labels_clf", "damage", "label_cls"])
        name = batch.get("file_name", batch.get("name", batch.get("id", [""] * pre.shape[0])))
    else:
        if len(batch) == 5:
            pre, post, loc, dam, name = batch
        elif len(batch) == 6:
            pre, post, _building_mask, loc, dam, name = batch
        else:
            raise ValueError(f"Unsupported BRIGHT batch length: {len(batch)}")

    pre = pre.to(device, non_blocking=True).float()
    post = post.to(device, non_blocking=True).float()
    loc = loc.to(device, non_blocking=True).long()
    dam = dam.to(device, non_blocking=True).long()
    if loc.dim() == 4 and loc.size(1) == 1:
        loc = loc.squeeze(1)
    if dam.dim() == 4 and dam.size(1) == 1:
        dam = dam.squeeze(1)
    return pre, post, loc, dam, name


# -----------------------------------------------------------------------------
# Lovasz-Softmax
# -----------------------------------------------------------------------------
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    if p == 0:
        return gt_sorted
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        # PyTorch 2.11 forbids overlapping in-place slice updates.
        # Compute the Lovasz discrete gradient out-of-place instead.
        jaccard = torch.cat(
            [
                jaccard[:1],
                jaccard[1:] - jaccard[:-1],
            ],
            dim=0,
        )
    return jaccard


def flatten_probas(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if labels.dim() == 4 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    if labels.shape[-2:] != probas.shape[-2:]:
        labels = F.interpolate(
            labels.unsqueeze(1).float(), size=probas.shape[-2:], mode="nearest"
        ).squeeze(1).long()
    c = probas.shape[1]
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.contiguous().view(-1)
    if ignore_index is None:
        return probas, labels
    valid = labels != ignore_index
    return probas[valid], labels[valid]


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | List[int] = "present",
) -> torch.Tensor:
    if probas.numel() == 0:
        return probas.sum() * 0.0
    class_ids = list(range(probas.size(1))) if classes in ("all", "present") else list(classes)
    losses: List[torch.Tensor] = []
    for cls in class_ids:
        fg = (labels == cls).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | List[int] = "present",
    ignore: int | None = 255,
) -> torch.Tensor:
    p, y = flatten_probas(probas, labels, ignore)
    return lovasz_softmax_flat(p, y, classes)


# -----------------------------------------------------------------------------
# S8 intact-anchored multi-level alignment
# -----------------------------------------------------------------------------
def schedule_scale(current_step: int, warmup: int, ramp: int) -> float:
    if current_step < warmup:
        return 0.0
    if ramp > 0 and current_step < warmup + ramp:
        return float(current_step - warmup) / float(ramp)
    return 1.0


def _cosine_distance_map(rgb: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
    rgb = F.normalize(rgb, dim=1)
    sar = F.normalize(sar, dim=1)
    return 1.0 - (rgb * sar).sum(dim=1)


def intact_anchored_multilevel_alignment_loss(
    outputs: Dict[str, torch.Tensor],
    damage_gt: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses: List[torch.Tensor] = []
    stats: Dict[str, torch.Tensor] = {}
    class_names = {1: "intact", 2: "damaged", 3: "destroyed"}

    for stage in (3, 4):
        # Cosine alignment is accumulated in FP32 for numerical stability.
        rgb = outputs[f"ia_rgb_s{stage}"].float()
        sar = outputs[f"ia_sar_s{stage}"].float()
        target = F.interpolate(
            damage_gt.unsqueeze(1).float(), size=rgb.shape[-2:], mode="nearest"
        ).squeeze(1).long()
        distance = _cosine_distance_map(rgb, sar)
        intact_mask = target == 1
        if intact_mask.any():
            losses.append(distance[intact_mask].mean())
        for cls, name in class_names.items():
            mask = target == cls
            if mask.any():
                stats[f"ia_s{stage}_{name}_dist"] = distance[mask].detach().mean()
                stats[f"ia_s{stage}_{name}_count"] = mask.detach().sum().float()
            else:
                stats[f"ia_s{stage}_{name}_dist"] = distance.detach().sum() * 0.0
                stats[f"ia_s{stage}_{name}_count"] = distance.detach().sum() * 0.0

    if not losses:
        zero = outputs["output_dam"].sum() * 0.0
        return zero, stats
    return torch.stack(losses).mean(), stats


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def fast_hist(pred: torch.Tensor, label: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    if label.dim() == 4 and label.size(1) == 1:
        label = label.squeeze(1)
    mask = label != ignore_index
    pred = pred[mask].long()
    label = label[mask].long()
    valid = (label >= 0) & (label < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    label = label[valid]
    if label.numel() == 0:
        return torch.zeros(num_classes, num_classes, device=pred.device, dtype=torch.float64)
    hist = torch.bincount(
        num_classes * label.reshape(-1) + pred.reshape(-1),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist.double()


def iou_from_hist(hist: torch.Tensor) -> torch.Tensor:
    hist = hist.double()
    inter = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - inter
    return inter / union.clamp_min(1e-12)


def f1_from_hist(hist: torch.Tensor) -> torch.Tensor:
    hist = hist.double()
    tp = torch.diag(hist)
    precision = tp / hist.sum(0).clamp_min(1e-12)
    recall = tp / hist.sum(1).clamp_min(1e-12)
    return 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)


def metrics_from_hist(loc_hist: torch.Tensor, dam_hist: torch.Tensor) -> Dict[str, float]:
    loc_iou = iou_from_hist(loc_hist)
    loc_f1 = f1_from_hist(loc_hist)
    dam_iou = iou_from_hist(dam_hist)
    dam_f1 = f1_from_hist(dam_hist)
    oa = torch.diag(dam_hist).sum() / dam_hist.sum().clamp_min(1e-12)

    gt_damaged = dam_hist[2].sum().clamp_min(1.0)
    gt_destroyed = dam_hist[3].sum().clamp_min(1.0)
    damaged_to_intact = dam_hist[2, 1] / gt_damaged
    destroyed_to_intact = dam_hist[3, 1] / gt_destroyed
    valid_miou = dam_iou[1:4].mean()
    damage_score = (
        valid_miou
        + 0.3 * dam_iou[2]
        + 0.3 * dam_iou[3]
        - 0.2 * damaged_to_intact
        - 0.2 * destroyed_to_intact
    )

    metrics = {
        "loc_iou": float(loc_iou[1].item()),
        "loc_f1": float(loc_f1[1].item()),
        "oa": float(oa.item()),
        "miou4": float(dam_iou.mean().item()),
        "valid_miou": float(valid_miou.item()),
        "damage_macro_f1": float(dam_f1[1:4].mean().item()),
        "background_iou": float(dam_iou[0].item()),
        "intact_iou": float(dam_iou[1].item()),
        "damaged_iou": float(dam_iou[2].item()),
        "destroyed_iou": float(dam_iou[3].item()),
        "intact_f1": float(dam_f1[1].item()),
        "damaged_f1": float(dam_f1[2].item()),
        "destroyed_f1": float(dam_f1[3].item()),
        "damaged_to_intact": float(damaged_to_intact.item()),
        "destroyed_to_intact": float(destroyed_to_intact.item()),
        "damage_score": float(damage_score.item()),
    }
    for gt_cls, name in ((2, "gt_damaged"), (3, "gt_destroyed")):
        row = dam_hist[gt_cls]
        row_pct = 100.0 * row / row.sum().clamp_min(1.0)
        for pred_cls, pred_name in enumerate(("background", "intact", "damaged", "destroyed")):
            metrics[f"{name}_pred_{pred_name}_pct"] = float(row_pct[pred_cls].item())
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    experiment: str,
    ignore_index: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    experiment = experiment.upper()
    model.eval()
    loc_hist = torch.zeros(2, 2, device=device, dtype=torch.float64)
    dam_hist = torch.zeros(4, 4, device=device, dtype=torch.float64)
    diag_sum: Dict[str, float] = {}
    diag_count: Dict[str, int] = {}

    ia_dist_sum = {name: 0.0 for name in ("intact", "damaged", "destroyed")}
    ia_dist_count = {name: 0.0 for name in ("intact", "damaged", "destroyed")}
    scale_sum = {name: [0.0] * 4 for name in ("intact", "damaged", "destroyed")}
    scale_count = {name: 0.0 for name in ("intact", "damaged", "destroyed")}

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        pre, post, loc, dam, _ = parse_batch(batch, device)
        outputs = model(pre, post, return_dict=True)
        pred_loc = outputs["output_loc"].argmax(1)
        pred_dam = outputs["output_dam"].argmax(1)
        loc_hist += fast_hist(pred_loc, loc, 2, ignore_index)
        dam_hist += fast_hist(pred_dam, dam, 4, ignore_index)

        for key in (
            "hcdm_stage1_abs_mean", "hcdm_stage2_abs_mean", "hcdm_stage3_abs_mean",
            "hcdm_stage4_abs_mean", "hcdm_fused_abs_mean", "scale_w1_mean",
            "scale_w2_mean", "scale_w3_mean", "scale_w4_mean",
            "building_prior_stage1_mean", "building_prior_stage2_mean",
            "building_prior_stage3_mean", "building_prior_stage4_mean",
        ):
            if key in outputs:
                value = float(outputs[key].detach().float().mean().item())
                diag_sum[key] = diag_sum.get(key, 0.0) + value
                diag_count[key] = diag_count.get(key, 0) + 1

        if experiment in ("S7", "S8"):
            weights = outputs["scale_weights"]
            target = F.interpolate(
                dam.unsqueeze(1).float(), size=weights.shape[-2:], mode="nearest"
            ).squeeze(1).long()
            for cls, name in ((1, "intact"), (2, "damaged"), (3, "destroyed")):
                mask = target == cls
                count = float(mask.sum().item())
                if count > 0:
                    scale_count[name] += count
                    for s in range(4):
                        scale_sum[name][s] += float(weights[:, s][mask].sum().item())

        if experiment == "S8":
            for stage in (3, 4):
                rgb = outputs[f"ia_rgb_s{stage}"]
                sar = outputs[f"ia_sar_s{stage}"]
                target = F.interpolate(
                    dam.unsqueeze(1).float(), size=rgb.shape[-2:], mode="nearest"
                ).squeeze(1).long()
                dist_map = _cosine_distance_map(rgb, sar)
                for cls, name in ((1, "intact"), (2, "damaged"), (3, "destroyed")):
                    mask = target == cls
                    if mask.any():
                        ia_dist_sum[name] += float(dist_map[mask].sum().item())
                        ia_dist_count[name] += float(mask.sum().item())

    metrics = metrics_from_hist(loc_hist, dam_hist)
    raw = unwrap_model(model)
    metrics["gamma_hcdm"] = float(raw.hcdm.gamma_hcdm.detach().item())
    for key, total in diag_sum.items():
        metrics[key] = total / max(1, diag_count[key])

    if experiment in ("S7", "S8"):
        for name in ("intact", "damaged", "destroyed"):
            for s in range(4):
                metrics[f"scale_w{s+1}_{name}"] = (
                    scale_sum[name][s] / max(1.0, scale_count[name])
                )

    if experiment == "S8":
        for name in ("intact", "damaged", "destroyed"):
            metrics[f"ia_{name}_distance"] = (
                ia_dist_sum[name] / max(1.0, ia_dist_count[name])
            )

    model.train()
    return metrics


# -----------------------------------------------------------------------------
# Checkpoint I/O
# -----------------------------------------------------------------------------
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def load_pretrained_weights(model: nn.Module, path: str, rank: int) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = clean_state_dict(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main(rank):
        print(f"Loaded pretrained/model weights: {path}", flush=True)
        print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}", flush=True)
        if missing:
            print("  first missing:", missing[:12], flush=True)
        if unexpected:
            print("  first unexpected:", unexpected[:12], flush=True)


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    experiment: str,
    args,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "experiment": experiment,
        "metrics": metrics,
        "args": vars(args),
    }
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    atomic_save(payload, path)


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    path: str,
    rank: int,
) -> Tuple[int, Dict[str, float]]:
    if not path:
        return 0, {}
    ckpt = torch.load(path, map_location="cpu")
    state = clean_state_dict(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and scaler.is_enabled() and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = int(ckpt.get("step", ckpt.get("iter", 0)))
    saved_metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    if is_main(rank):
        print(
            f"Resumed {path} at optimizer step {step} | "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return step, dict(saved_metrics)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ce_fn: nn.Module,
    use_lovasz: bool,
    ce_weight: float,
    lovasz_weight: float,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep segmentation losses in FP32 even when the forward pass uses AMP.
    logits_fp32 = logits.float()
    ce = ce_fn(logits_fp32, labels)
    lv = logits_fp32.sum() * 0.0
    if use_lovasz:
        lv = lovasz_softmax(F.softmax(logits_fp32, dim=1), labels, classes="present", ignore=ignore_index)
    total = float(ce_weight) * ce + float(lovasz_weight) * lv
    return total, ce, lv


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"loc_F1={100*metrics['loc_f1']:.4f}, "
        f"damage_macro_F1={100*metrics['damage_macro_f1']:.4f}, "
        f"OA={100*metrics['oa']:.4f}, "
        f"mIoU4={100*metrics['miou4']:.4f}, "
        f"valid_mIoU={100*metrics['valid_miou']:.4f} | "
        f"IoU(bg/int/dmg/des)={100*metrics['background_iou']:.2f}/"
        f"{100*metrics['intact_iou']:.2f}/{100*metrics['damaged_iou']:.2f}/"
        f"{100*metrics['destroyed_iou']:.2f} | "
        f"Dmg->Int={100*metrics['damaged_to_intact']:.2f}% "
        f"Des->Int={100*metrics['destroyed_to_intact']:.2f}% | "
        f"gamma={metrics['gamma_hcdm']:.6f}"
    )



def build_grad_scaler(enabled: bool):
    """Create a CUDA GradScaler across old and new PyTorch AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_autocast(dtype: torch.dtype, enabled: bool):
    """Create a CUDA autocast context across old and new PyTorch APIs."""
    if hasattr(torch, "autocast"):
        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=enabled,
        )
    return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def main() -> None:
    parser = argparse.ArgumentParser("Train S5/S6/S7/S8 hierarchical RGB-SAR DamageFormer")
    parser.add_argument("--experiment", required=True, choices=["S5", "S6", "S7", "S8", "s5", "s6", "s7", "s8"])
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--output_root", default=str(ROOT / "saved_weights" / "S5_S8"))
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--pretrained_model_path", default="")
    parser.add_argument("--resume", default="")

    parser.add_argument("--image_size", "--crop_size", dest="image_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", "--optimizer_steps", dest="max_steps", type=int, default=50000, help="Number of optimizer steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_pretrained_swin", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="CUDA AMP dtype. fp16 uses GradScaler; bf16 uses autocast without scaling.",
    )

    parser.add_argument("--use_lovasz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument("--diff_channels", type=int, default=64)
    parser.add_argument("--gamma_init", type=float, default=0.0)
    parser.add_argument("--roi_alpha", type=float, default=0.1, help="S6/S7/S8 soft background floor")
    parser.add_argument("--align_dim", type=int, default=128, help="S8 only")
    parser.add_argument("--lambda_ia", type=float, default=0.005, help="S8 only")
    parser.add_argument("--ia_warmup", type=int, default=8000, help="Optimizer steps; S8 only")
    parser.add_argument("--ia_ramp", type=int, default=2000, help="Optimizer steps; S8 only")
    parser.add_argument("--no_detach_building_prior", action="store_true", help="Allow damage loss gradients into building prior; default is detached.")

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--val_max_batches", type=int, default=0, help="0 = full validation set")
    parser.add_argument("--save_interval", type=int, default=5000)
    args = parser.parse_args()

    experiment = args.experiment.upper()
    device, distributed, rank, world_size, local_rank = setup_runtime(args.device)
    generator = set_random_seed(args.seed, rank, args.deterministic)
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    use_grad_scaler = amp_enabled and args.amp_dtype == "fp16"
    scaler = build_grad_scaler(enabled=use_grad_scaler)

    try:
        if is_main(rank):
            print("=" * 90, flush=True)
            print(f"Experiment: {experiment}", flush=True)
            print(f"Device: {device} | DDP={distributed} | world_size={world_size}", flush=True)
            print(
                f"Protocol: image={args.image_size}, micro_batch={args.train_batch_size}, "
                f"accum={args.accumulation_steps}, max_optimizer_steps={args.max_steps}, "
                f"lr={args.learning_rate}, wd={args.weight_decay}, "
                f"amp={args.amp_dtype}",
                flush=True,
            )
            print(
                f"Effective batch per optimizer step = "
                f"{args.train_batch_size * args.accumulation_steps * world_size}",
                flush=True,
            )
            descriptions = {
                "S5": "HCDM: four-stage RGB-SAR relation, fixed concatenation.",
                "S6": "BG-HCDM: S5 + detached building prior at every stage.",
                "S7": "AS-BG-HCDM: S6 + adaptive softmax scale fusion.",
                "S8": (
                    f"IA-MLA: S7 + intact-only Stage3/4 alignment; "
                    f"lambda={args.lambda_ia}, warmup={args.ia_warmup}, ramp={args.ia_ramp}."
                ),
            }
            print(descriptions[experiment], flush=True)
            print("=" * 90, flush=True)

        train_set = build_dataset(
            args.train_dataset_path, args.train_data_list_path, args.image_size, "train"
        )
        val_set = build_dataset(
            args.val_dataset_path, args.val_data_list_path, args.image_size, "val"
        )
        train_sampler = (
            DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=generator if train_sampler is None else None,
            persistent_workers=args.num_workers > 0,
        )
        # Full validation loader. In DDP, rank 0 evaluates the raw module alone.
        val_loader = DataLoader(
            val_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            worker_init_fn=seed_worker,
            persistent_workers=args.num_workers > 0,
        )
        if len(train_loader) == 0:
            raise RuntimeError("Training loader is empty. Check batch size and dataset list.")

        ModelClass = import_model(experiment)
        common_kwargs = dict(
            variant=experiment,
            diff_channels=args.diff_channels,
            gamma_init=args.gamma_init,
            roi_alpha=args.roi_alpha,
            align_dim=args.align_dim,
            detach_building_prior=not args.no_detach_building_prior,
            use_pretrained_swin=not args.no_pretrained_swin,
            return_dict=True,
        )
        raw_model = ModelClass(**common_kwargs).to(device)
        load_pretrained_weights(raw_model, args.pretrained_model_path, rank)

        optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        start_step, resumed_best_metrics = resume_training(
            raw_model, optimizer, scaler, args.resume, rank
        )
        model: nn.Module = raw_model
        if distributed:
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        loc_ce_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
        dam_ce_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = args.exp_name or f"{experiment}_Hierarchical_RGBSAR_seed{args.seed}_{timestamp}"
        exp_dir = Path(args.output_root) / exp_name
        if is_main(rank):
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
            print(f"Experiment directory: {exp_dir}", flush=True)
            print(f"Train samples={len(train_set)} | Val samples={len(val_set)}", flush=True)

        optimizer.zero_grad(set_to_none=True)
        step = start_step
        micro_in_accum = 0
        epoch = 0
        best_metrics: Dict[str, float] = dict(resumed_best_metrics)
        best_valid_miou = float(best_metrics.get("valid_miou", -1.0))

        while step < args.max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                model.train()
                pre, post, loc, dam, _ = parse_batch(batch, device)

                # With DDP + gradient accumulation, synchronize gradients only on the
                # final micro-step of each optimizer step. PyTorch DDP requires the
                # forward pass to be inside no_sync() as well as backward.
                will_sync = (micro_in_accum + 1) >= args.accumulation_steps
                sync_context = (
                    nullcontext()
                    if (not distributed or will_sync)
                    else model.no_sync()
                )

                with sync_context:
                    with cuda_autocast(dtype=amp_dtype, enabled=amp_enabled):
                        outputs = model(pre, post, return_dict=True)

                    loss_loc, loss_loc_ce, loss_loc_lv = segmentation_loss(
                        outputs["output_loc"], loc, loc_ce_fn, args.use_lovasz,
                        args.ce_weight, args.lovasz_weight, args.ignore_index,
                    )
                    loss_dam, loss_dam_ce, loss_dam_lv = segmentation_loss(
                        outputs["output_dam"], dam, dam_ce_fn, args.use_lovasz,
                        args.ce_weight, args.lovasz_weight, args.ignore_index,
                    )

                    loss_ia = outputs["output_dam"].float().sum() * 0.0
                    ia_stats: Dict[str, torch.Tensor] = {}
                    ia_scale = 0.0
                    lambda_ia_current = 0.0
                    if experiment == "S8":
                        loss_ia, ia_stats = intact_anchored_multilevel_alignment_loss(outputs, dam)
                        # The next optimizer step is step+1; use that for schedule semantics.
                        ia_scale = schedule_scale(step + 1, args.ia_warmup, args.ia_ramp)
                        lambda_ia_current = args.lambda_ia * ia_scale

                    total_loss = loss_loc + loss_dam + lambda_ia_current * loss_ia
                    scaled_loss = total_loss / float(args.accumulation_steps)
                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                micro_in_accum += 1

                if micro_in_accum < args.accumulation_steps:
                    continue

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_in_accum = 0
                step += 1

                if is_main(rank) and step % args.log_interval == 0:
                    gamma = float(unwrap_model(model).hcdm.gamma_hcdm.detach().item())
                    stage_means = [
                        float(outputs[f"hcdm_stage{i}_abs_mean"].detach().float().mean().item())
                        for i in range(1, 5)
                    ]
                    msg = (
                        f"step {step:06d}/{args.max_steps} | total {total_loss.item():.4f} | "
                        f"loc {loss_loc.item():.4f} (ce {loss_loc_ce.item():.4f}, lv {loss_loc_lv.item():.4f}) | "
                        f"dam {loss_dam.item():.4f} (ce {loss_dam_ce.item():.4f}, lv {loss_dam_lv.item():.4f}) | "
                        f"gamma {gamma:.6f} | HCDM stage abs "
                        + "/".join(f"{x:.4f}" for x in stage_means)
                    )
                    if experiment in ("S7", "S8"):
                        msg += (
                            " | scale_w "
                            + "/".join(
                                f"{float(outputs[f'scale_w{i}_mean'].detach().float().mean().item()):.3f}"
                                for i in range(1, 5)
                            )
                        )
                    if experiment == "S8":
                        msg += (
                            f" | IA {loss_ia.item():.4f} lambda_now {lambda_ia_current:.6f} "
                            f"scale {ia_scale:.3f}"
                        )
                        for name in ("intact", "damaged", "destroyed"):
                            vals = []
                            for s in (3, 4):
                                key = f"ia_s{s}_{name}_dist"
                                if key in ia_stats:
                                    vals.append(float(ia_stats[key].item()))
                            if vals:
                                msg += f" | {name}_dist_batch={sum(vals)/len(vals):.4f}"
                    print(msg, flush=True)

                do_val = (step % args.val_interval == 0) or (step == args.max_steps)
                if do_val:
                    if distributed:
                        dist.barrier()
                    if is_main(rank):
                        eval_model = unwrap_model(model)
                        metrics = evaluate_model(
                            eval_model,
                            val_loader,
                            device,
                            experiment,
                            args.ignore_index,
                            args.val_max_batches,
                        )
                        print("--------- validation ---------", flush=True)
                        print(format_metrics(metrics), flush=True)
                        print(
                            "GT damaged prediction row (%): "
                            f"bg={metrics['gt_damaged_pred_background_pct']:.2f}, "
                            f"int={metrics['gt_damaged_pred_intact_pct']:.2f}, "
                            f"dmg={metrics['gt_damaged_pred_damaged_pct']:.2f}, "
                            f"des={metrics['gt_damaged_pred_destroyed_pct']:.2f}",
                            flush=True,
                        )
                        print(
                            "GT destroyed prediction row (%): "
                            f"bg={metrics['gt_destroyed_pred_background_pct']:.2f}, "
                            f"int={metrics['gt_destroyed_pred_intact_pct']:.2f}, "
                            f"dmg={metrics['gt_destroyed_pred_damaged_pct']:.2f}, "
                            f"des={metrics['gt_destroyed_pred_destroyed_pct']:.2f}",
                            flush=True,
                        )
                        if experiment in ("S7", "S8"):
                            for name in ("intact", "damaged", "destroyed"):
                                print(
                                    f"Scale weights | {name}: "
                                    + ", ".join(
                                        f"S{i}={metrics[f'scale_w{i}_{name}']:.4f}" for i in range(1, 5)
                                    ),
                                    flush=True,
                                )
                        if experiment == "S8":
                            print(
                                "IA distances: "
                                f"intact={metrics['ia_intact_distance']:.5f}, "
                                f"damaged={metrics['ia_damaged_distance']:.5f}, "
                                f"destroyed={metrics['ia_destroyed_distance']:.5f}",
                                flush=True,
                            )

                        with open(exp_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps({"step": step, **metrics}, ensure_ascii=False) + "\n")

                        if metrics["valid_miou"] > best_valid_miou:
                            best_valid_miou = metrics["valid_miou"]
                            best_metrics = dict(metrics)
                            save_checkpoint(
                                exp_dir / "best_valid_miou.pth",
                                model,
                                optimizer,
                                scaler,
                                step,
                                experiment,
                                args,
                                metrics,
                            )
                            print(
                                f"NEW BEST valid_mIoU={100*best_valid_miou:.4f} at step={step}",
                                flush=True,
                            )
                        else:
                            print(
                                f"No new valid_mIoU best; current={100*metrics['valid_miou']:.4f}, "
                                f"best={100*best_valid_miou:.4f}",
                                flush=True,
                            )
                    if distributed:
                        dist.barrier()

                if is_main(rank) and step % args.save_interval == 0:
                    save_checkpoint(
                        exp_dir / "latest_resume.pth",
                        model,
                        optimizer,
                        scaler,
                        step,
                        experiment,
                        args,
                        best_metrics,
                    )

                if step >= args.max_steps:
                    break
            epoch += 1

        if is_main(rank):
            save_checkpoint(
                exp_dir / "final.pth",
                model,
                optimizer,
                scaler,
                step,
                experiment,
                args,
                best_metrics,
            )
            print("Training completed.", flush=True)
            print(f"Best valid_mIoU={100*best_valid_miou:.4f}", flush=True)
            print(f"Artifacts: {exp_dir}", flush=True)

    finally:
        cleanup_runtime(distributed)


if __name__ == "__main__":
    main()
