# -*- coding: utf-8 -*-
"""2-GPU DDP trainer for IARM-DamageFormer on BRIGHT.

The script preserves the official DamageFormer objective and adds controlled
auxiliary terms. Presets are deliberately incremental so each hypothesis can
be falsified independently.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
BDA_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = BDA_ROOT.parent
for import_root in (PROJECT_ROOT, BDA_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_IARM import DamageFormer_IARM
import util_func.lovasz_loss as L


CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]

PRESETS: Dict[str, Dict[str, object]] = {
    "E0_BASE": {
        "enable_residual_fusion": False,
        "normal_weight": 0.0,
        "cmu_weight": 0.0,
        "flow_smooth_weight": 0.0,
        "flow_magnitude_weight": 0.0,
        "damage_exist_weight": 0.0,
        "anchor_weight": 0.0,
        "cross_event_contrast_weight": 0.0,
    },
    "E1_CMU_NORMAL": {
        "enable_residual_fusion": False,
        "normal_weight": 0.05,
        "cmu_weight": 0.02,
        "flow_smooth_weight": 0.005,
        "flow_magnitude_weight": 0.001,
        "damage_exist_weight": 0.0,
        "anchor_weight": 0.0,
        "cross_event_contrast_weight": 0.0,
    },
    "E2_IAR_EXIST": {
        "enable_residual_fusion": True,
        "normal_weight": 0.05,
        "cmu_weight": 0.02,
        "flow_smooth_weight": 0.005,
        "flow_magnitude_weight": 0.001,
        "damage_exist_weight": 0.20,
        "anchor_weight": 0.0,
        "cross_event_contrast_weight": 0.0,
    },
    "E3_IAR_METRIC": {
        "enable_residual_fusion": True,
        "normal_weight": 0.05,
        "cmu_weight": 0.02,
        "flow_smooth_weight": 0.005,
        "flow_magnitude_weight": 0.001,
        "damage_exist_weight": 0.20,
        "anchor_weight": 0.02,
        "cross_event_contrast_weight": 0.02,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IARM-DamageFormer 2-GPU DDP trainer")
    p.add_argument("--dataset", default="BRIGHT")
    p.add_argument("--train_dataset_path", required=True)
    p.add_argument("--train_data_list_path", required=True)
    p.add_argument("--val_dataset_path", required=True)
    p.add_argument("--val_data_list_path", required=True)
    p.add_argument("--test_dataset_path", required=True)
    p.add_argument("--test_data_list_path", required=True)

    p.add_argument("--experiment", choices=sorted(PRESETS), default="E2_IAR_EXIST")
    p.add_argument("--residual_stages", default="4", help="Comma-separated subset of 3,4")
    p.add_argument("--residual_dim", type=int, default=128)
    p.add_argument("--flow_hidden_dim", type=int, default=64)
    p.add_argument("--max_offset_pixels", type=float, default=2.0)
    p.add_argument("--detach_backbone_for_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable_residual_fusion", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--residual_gamma_max", type=float, default=0.25)
    p.add_argument("--roi_alpha", type=float, default=0.1)

    # Auxiliary loss weights. None means use the selected preset.
    p.add_argument("--normal_weight", type=float, default=None)
    p.add_argument("--cmu_weight", type=float, default=None)
    p.add_argument("--flow_smooth_weight", type=float, default=None)
    p.add_argument("--flow_magnitude_weight", type=float, default=None)
    p.add_argument("--damage_exist_weight", type=float, default=None)
    p.add_argument("--anchor_weight", type=float, default=None)
    p.add_argument("--cross_event_contrast_weight", type=float, default=None)
    p.add_argument("--cmu_temperature", type=float, default=0.1)
    p.add_argument("--cmu_max_samples", type=int, default=256)
    p.add_argument("--contrast_temperature", type=float, default=0.1)
    p.add_argument("--contrast_samples_per_image_class", type=int, default=12)

    p.add_argument("--gaussian_step_momentum", type=float, default=0.99)
    p.add_argument("--gaussian_variance_floor", type=float, default=2.0e-3)
    p.add_argument("--max_pixels_per_image_class", type=int, default=256)

    p.add_argument("--train_batch_size_per_gpu", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--optimizer_steps", type=int, default=50000)
    p.add_argument("--max_iters", type=int, default=800000)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--model_type", default="IARM_DamageFormer")
    p.add_argument("--model_param_path", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--init_checkpoint", default=None, help="Optional BG/baseline weights-only warm start")
    p.add_argument("--eval_test_during_training", action="store_true")
    return p.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    preset = PRESETS[args.experiment]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for key in (
        "normal_weight",
        "cmu_weight",
        "flow_smooth_weight",
        "flow_magnitude_weight",
        "damage_exist_weight",
        "anchor_weight",
        "cross_event_contrast_weight",
    ):
        value = float(getattr(args, key))
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
        setattr(args, key, value)


def parse_stages(text: str) -> Tuple[int, ...]:
    stages = tuple(sorted({int(token.strip()) for token in text.split(",") if token.strip()}))
    if not stages or any(stage not in (3, 4) for stage in stages):
        raise ValueError("--residual_stages must be 3, 4, or 3,4")
    return stages


def read_list(path: str) -> List[str]:
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"Empty list: {path}")
    return names


def infer_event_name(sample_name: str) -> str:
    stem = Path(str(sample_name).replace("\\", "/")).stem.lower()
    stem = re.sub(r"_(pre|post)_disaster$", "", stem)
    return re.sub(r"_\d+$", "", stem)


def build_event_vocabulary(names: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    events = sorted({infer_event_name(name) for name in names})
    return events, {event: idx for idx, event in enumerate(events)}


def event_ids_from_batch(names: Sequence[str], mapping: Dict[str, int], device: torch.device) -> torch.Tensor:
    result = []
    for name in names:
        event = infer_event_name(str(name))
        if event not in mapping:
            raise KeyError(f"Unknown training event {event!r}")
        result.append(mapping[event])
    return torch.tensor(result, dtype=torch.long, device=device)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --nproc_per_node=2")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


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


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, temp, _use_new_zipfile_serialization=False)
        os.replace(temp, path)
    except Exception:
        if temp.exists():
            temp.unlink(missing_ok=True)
        raise


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_resume(model: nn.Module, path: str) -> Dict[str, object]:
    obj = torch.load(path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(clean_state_dict(state), strict=True)
    return obj if isinstance(obj, dict) else {}


def load_initial_weights(model: nn.Module, path: str) -> None:
    obj = torch.load(path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    incompatible = model.load_state_dict(clean_state_dict(state), strict=False)
    allowed_missing_prefixes = (
        "optical_projectors.",
        "sar_projectors.",
        "aligners.",
        "normal_mappers.",
        "residual_encoders.",
        "residual_fuser.",
        "damage_exist_head.",
        "residual_to_damage.",
        "residual_gamma_raw",
        "class_distribution_memory.",
    )
    bad_missing = [k for k in incompatible.missing_keys if not k.startswith(allowed_missing_prefixes)]
    # Swin terminal norm/head may exist in an old checkpoint but are unused.
    bad_unexpected = [
        k for k in incompatible.unexpected_keys
        if not (k.endswith(".head.weight") or k.endswith(".head.bias") or k.endswith(".norm.weight") or k.endswith(".norm.bias"))
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Unsafe init checkpoint mismatch: missing={bad_missing}, unexpected={bad_unexpected}"
        )


def resize_target(target: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(target.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def masked_normal_relation_loss(aux: Dict[str, torch.Tensor], labels: torch.Tensor, stages: Sequence[int]) -> torch.Tensor:
    losses = []
    for stage in stages:
        prediction = aux[f"normal_prediction_s{stage}"]
        target = aux[f"aligned_sar_metric_s{stage}"].detach()
        labels_small = resize_target(labels, prediction.shape[-2:])
        mask = labels_small == 1
        if not bool(mask.any().item()):
            losses.append(prediction.sum() * 0.0)
            continue
        pred_vec = prediction.permute(0, 2, 3, 1)[mask]
        target_vec = target.permute(0, 2, 3, 1)[mask]
        smooth_l1 = F.smooth_l1_loss(pred_vec, target_vec)
        cosine = 1.0 - F.cosine_similarity(pred_vec, target_vec, dim=1).mean()
        losses.append(smooth_l1 + cosine)
    return torch.stack(losses).mean()


def paired_cmu_info_nce(
    optical: torch.Tensor,
    aligned_sar: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    max_samples: int,
) -> torch.Tensor:
    labels_small = resize_target(labels, optical.shape[-2:])
    mask = labels_small == 1
    optical_vec = optical.permute(0, 2, 3, 1)[mask]
    sar_vec = aligned_sar.permute(0, 2, 3, 1)[mask]
    if optical_vec.shape[0] < 2:
        return optical.sum() * 0.0
    if optical_vec.shape[0] > max_samples:
        index = torch.randperm(optical_vec.shape[0], device=optical.device)[:max_samples]
        optical_vec = optical_vec[index]
        sar_vec = sar_vec[index]
    optical_vec = F.normalize(optical_vec, dim=1)
    sar_vec = F.normalize(sar_vec, dim=1)
    logits = optical_vec @ sar_vec.t() / temperature
    target_index = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target_index) + F.cross_entropy(logits.t(), target_index))


def cmu_loss(aux: Dict[str, torch.Tensor], labels: torch.Tensor, stages: Sequence[int], temperature: float, max_samples: int) -> torch.Tensor:
    losses = []
    per_stage_samples = max(2, max_samples // len(stages))
    for stage in stages:
        losses.append(
            paired_cmu_info_nce(
                aux[f"optical_metric_s{stage}"],
                aux[f"aligned_sar_metric_s{stage}"],
                labels,
                temperature,
                per_stage_samples,
            )
        )
    return torch.stack(losses).mean()


def flow_regularization(aux: Dict[str, torch.Tensor], stages: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    smooth_losses = []
    magnitude_losses = []
    for stage in stages:
        flow = aux[f"flow_s{stage}"]
        dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean() if flow.shape[-1] > 1 else flow.sum() * 0.0
        dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean() if flow.shape[-2] > 1 else flow.sum() * 0.0
        smooth_losses.append(dx + dy)
        magnitude_losses.append(flow.abs().mean())
    return torch.stack(smooth_losses).mean(), torch.stack(magnitude_losses).mean()


def damage_exist_loss(output_exist: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(labels, 255)
    target[labels == 1] = 0
    target[(labels == 2) | (labels == 3)] = 1
    if not bool((target != 255).any().item()):
        return output_exist.sum() * 0.0
    return F.cross_entropy(output_exist, target, ignore_index=255)


def anchor_distribution_loss(aux: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    logits = aux["anchor_logits_feat"]
    if not bool(aux["anchor_ready"].item()):
        return logits.sum() * 0.0
    target = resize_target(labels, logits.shape[-2:])
    target = torch.where((target >= 1) & (target <= 3), target - 1, torch.full_like(target, 255))
    return F.cross_entropy(logits, target, ignore_index=255)


def _select_metric_samples(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    main_logits_feat: torch.Tensor,
    samples_per_image_class: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-shape class-balanced samples with half hard-anchor preference."""
    batch, dim, height, width = embedding.shape
    target = resize_target(labels, (height, width))
    prediction = main_logits_feat.detach().argmax(dim=1)
    slots = batch * 3 * samples_per_image_class
    features = embedding.new_zeros(slots, dim)
    out_labels = torch.full((slots,), -1, dtype=torch.long, device=embedding.device)
    out_events = torch.full((slots,), -1, dtype=torch.long, device=embedding.device)
    valid = torch.zeros(slots, dtype=torch.bool, device=embedding.device)
    unique_ids = torch.arange(slots, device=embedding.device, dtype=torch.long) + rank * 10_000_000

    cursor = 0
    for batch_idx in range(batch):
        flat_embedding = embedding[batch_idx].permute(1, 2, 0).reshape(-1, dim)
        flat_target = target[batch_idx].reshape(-1)
        flat_prediction = prediction[batch_idx].reshape(-1)
        for class_idx, label_value in enumerate((1, 2, 3)):
            candidate = torch.where(flat_target == label_value)[0]
            selected = candidate.new_empty((0,), dtype=torch.long)
            if candidate.numel() > 0:
                hard = candidate[flat_prediction[candidate] != label_value]
                hard_quota = samples_per_image_class // 2
                if hard.numel() > 0:
                    hard = hard[torch.randperm(hard.numel(), device=hard.device)[:hard_quota]]
                    selected = hard
                remaining_quota = samples_per_image_class - selected.numel()
                if remaining_quota > 0:
                    if selected.numel() > 0:
                        keep_mask = ~torch.isin(candidate, selected)
                        pool = candidate[keep_mask]
                    else:
                        pool = candidate
                    if pool.numel() > 0:
                        random_part = pool[torch.randperm(pool.numel(), device=pool.device)[:remaining_quota]]
                        selected = torch.cat([selected, random_part])
            count = min(int(selected.numel()), samples_per_image_class)
            if count > 0:
                features[cursor : cursor + count] = flat_embedding[selected[:count]]
                out_labels[cursor : cursor + count] = class_idx
                out_events[cursor : cursor + count] = events[batch_idx]
                valid[cursor : cursor + count] = True
            cursor += samples_per_image_class
    return features, out_labels, out_events, valid, unique_ids


def cross_event_supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    event_ids: torch.Tensor,
    main_logits_feat: torch.Tensor,
    samples_per_image_class: int,
    temperature: float,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    local = _select_metric_samples(
        embedding,
        labels,
        event_ids,
        main_logits_feat,
        samples_per_image_class,
        rank,
    )
    local_features, local_labels, local_events, local_valid, local_ids = local

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        if world_size == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.detach())
        return torch.cat(gathered, dim=0)

    key_features = gather(local_features)
    key_labels = gather(local_labels)
    key_events = gather(local_events)
    key_valid = gather(local_valid)
    key_ids = gather(local_ids)

    query_mask = local_valid
    if int(query_mask.sum().item()) == 0 or int(key_valid.sum().item()) < 2:
        return embedding.sum() * 0.0
    queries = F.normalize(local_features[query_mask], dim=1)
    query_labels = local_labels[query_mask]
    query_events = local_events[query_mask]
    query_ids = local_ids[query_mask]
    keys = F.normalize(key_features[key_valid], dim=1)
    labels_k = key_labels[key_valid]
    events_k = key_events[key_valid]
    ids_k = key_ids[key_valid]

    logits = queries @ keys.t() / temperature
    self_mask = query_ids[:, None] == ids_k[None, :]
    logits = logits.masked_fill(self_mask, -1.0e4)
    same_class = query_labels[:, None] == labels_k[None, :]
    cross_event_positive = same_class & (query_events[:, None] != events_k[None, :]) & (~self_mask)
    fallback_positive = same_class & (~self_mask)
    use_cross = cross_event_positive.any(dim=1, keepdim=True)
    positive = torch.where(use_cross, cross_event_positive, fallback_positive)
    valid_query = positive.any(dim=1)
    if not bool(valid_query.any().item()):
        return embedding.sum() * 0.0

    logits = logits[valid_query]
    positive = positive[valid_query]
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probability * positive.float()).sum(dim=1).div(positive.sum(dim=1)).mean()


def compute_losses(
    output_loc: torch.Tensor,
    output_damage: torch.Tensor,
    output_exist: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    labels_loc: torch.Tensor,
    labels_damage: torch.Tensor,
    event_ids: torch.Tensor,
    args: argparse.Namespace,
    stages: Sequence[int],
    rank: int,
    world_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
    lovasz_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
    ce_damage = F.cross_entropy(output_damage, labels_damage, ignore_index=255)
    lovasz_damage = L.lovasz_softmax(F.softmax(output_damage, dim=1), labels_damage, ignore=255)
    main = ce_loc + ce_damage + 0.75 * lovasz_damage + 0.5 * lovasz_loc

    normal = masked_normal_relation_loss(aux, labels_damage, stages)
    cmu = cmu_loss(aux, labels_damage, stages, args.cmu_temperature, args.cmu_max_samples)
    flow_smooth, flow_magnitude = flow_regularization(aux, stages)
    exist = damage_exist_loss(output_exist, labels_damage)
    anchor = anchor_distribution_loss(aux, labels_damage)
    xevent = cross_event_supervised_contrastive_loss(
        aux["metric_embedding"],
        labels_damage,
        event_ids,
        aux["main_damage_logits_feat"],
        args.contrast_samples_per_image_class,
        args.contrast_temperature,
        rank,
        world_size,
    )

    total = (
        main
        + args.normal_weight * normal
        + args.cmu_weight * cmu
        + args.flow_smooth_weight * flow_smooth
        + args.flow_magnitude_weight * flow_magnitude
        + args.damage_exist_weight * exist
        + args.anchor_weight * anchor
        + args.cross_event_contrast_weight * xevent
    )
    stats = {
        "total": float(total.detach().item()),
        "main": float(main.detach().item()),
        "ce_damage": float(ce_damage.detach().item()),
        "lovasz_damage": float(lovasz_damage.detach().item()),
        "normal": float(normal.detach().item()),
        "cmu": float(cmu.detach().item()),
        "flow_smooth": float(flow_smooth.detach().item()),
        "flow_magnitude": float(flow_magnitude.detach().item()),
        "exist": float(exist.detach().item()),
        "anchor": float(anchor.detach().item()),
        "xevent": float(xevent.detach().item()),
    }
    return total, stats


def fast_hist(gt: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid = (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    return np.bincount(classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64), minlength=classes ** 2).reshape(classes, classes)


def iou_from_hist(hist: np.ndarray) -> np.ndarray:
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(np.diag(hist), denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)


def metrics_from_hist(hist: np.ndarray) -> Dict[str, object]:
    iou = iou_from_hist(hist)
    tp = np.diag(hist).astype(np.float64)
    precision = np.divide(tp, hist.sum(0), out=np.zeros(4), where=hist.sum(0) > 0)
    recall = np.divide(tp, hist.sum(1), out=np.zeros(4), where=hist.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4), where=(precision + recall) > 0)
    damage_f1 = f1[1:4]
    harmonic = 3.0 / np.sum(1.0 / np.clip(damage_f1, 1e-12, None))
    return {
        "miou": float(iou.mean()),
        "target3_miou": float(iou[1:4].mean()),
        "iou": iou.tolist(),
        "damage_f1": damage_f1.tolist(),
        "damage_hf1": float(harmonic),
        "oa": float(tp.sum() / max(hist.sum(), 1)),
    }


def localization_f1(hist: np.ndarray) -> float:
    tp = float(hist[1, 1])
    fp = float(hist[0, 1])
    fn = float(hist[1, 0])
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return 2 * precision * recall / max(precision + recall, 1e-12)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset_path: str,
    names: Sequence[str],
    batch_size: int,
    workers: int,
    device: torch.device,
    split_name: str,
) -> Dict[str, object]:
    dataset = MultimodalDamageAssessmentDatset(dataset_path, list(names), 1024, None, "test")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    model.eval()
    active_hist = np.zeros((4, 4), dtype=np.int64)
    base_hist = np.zeros((4, 4), dtype=np.int64)
    loc_hist = np.zeros((2, 2), dtype=np.int64)
    event_hist = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    exist_hist = np.zeros((2, 2), dtype=np.int64)

    for pre, post, labels_loc, labels_damage, sample_names in tqdm(loader, desc=split_name):
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc = labels_loc.to(device, non_blocking=True).long()
        labels_damage = labels_damage.to(device, non_blocking=True).long()
        output_loc, output_damage, output_exist, aux = model(pre, post, return_aux=True)
        base_logits = F.interpolate(aux["main_damage_logits_feat"], size=output_damage.shape[-2:], mode="bilinear", align_corners=False)

        loc_pred = output_loc.argmax(1).cpu().numpy()
        active_pred = output_damage.argmax(1).cpu().numpy()
        base_pred = base_logits.argmax(1).cpu().numpy()
        damage_gt = labels_damage.cpu().numpy()
        loc_gt = labels_loc.cpu().numpy()
        valid_loc = loc_gt != 255
        loc_hist += fast_hist((loc_gt > 0)[valid_loc].astype(np.int64), (loc_pred > 0)[valid_loc].astype(np.int64), 2)
        active_hist += fast_hist(damage_gt, active_pred, 4)
        base_hist += fast_hist(damage_gt, base_pred, 4)

        exist_gt = np.full_like(damage_gt, 255)
        exist_gt[damage_gt == 1] = 0
        exist_gt[(damage_gt == 2) | (damage_gt == 3)] = 1
        exist_pred = output_exist.argmax(1).cpu().numpy()
        valid_exist = exist_gt != 255
        exist_hist += fast_hist(exist_gt[valid_exist], exist_pred[valid_exist], 2)

        for idx, name in enumerate(sample_names):
            event_hist[infer_event_name(str(name))] += fast_hist(damage_gt[idx], active_pred[idx], 4)

    active = metrics_from_hist(active_hist)
    base = metrics_from_hist(base_hist)
    event_scores = {}
    for event, hist in event_hist.items():
        iou = iou_from_hist(hist)
        present = (hist.sum(1) + hist.sum(0) - np.diag(hist))[1:4] > 0
        event_scores[event] = float(iou[1:4][present].mean()) if present.any() else float("nan")
    finite = {k: v for k, v in event_scores.items() if np.isfinite(v)}
    exist_metrics = metrics_from_hist(np.pad(exist_hist, ((0, 2), (0, 2)))) if False else None
    tp = float(exist_hist[1, 1]); fp = float(exist_hist[0, 1]); fn = float(exist_hist[1, 0])
    exist_precision = tp / max(tp + fp, 1.0); exist_recall = tp / max(tp + fn, 1.0)
    exist_f1 = 2 * exist_precision * exist_recall / max(exist_precision + exist_recall, 1e-12)

    result = {
        "loc_f1": localization_f1(loc_hist),
        "active": active,
        "base": base,
        "active_minus_base_miou": float(active["miou"] - base["miou"]),
        "event_macro_target3": float(np.mean(list(finite.values()))) if finite else float("nan"),
        "worst_event": min(finite, key=finite.get) if finite else "N/A",
        "worst_event_target3": min(finite.values()) if finite else float("nan"),
        "event_target3": event_scores,
        "damage_exist_f1": float(exist_f1),
    }
    print(
        f"{split_name}: locF1={100*result['loc_f1']:.4f}, "
        f"active mIoU={100*active['miou']:.4f}, base mIoU={100*base['miou']:.4f}, "
        f"delta={100*result['active_minus_base_miou']:+.4f}, "
        f"IoU={100*np.asarray(active['iou'])}, existF1={100*exist_f1:.4f}, "
        f"eventMacro={100*result['event_macro_target3']:.4f}, "
        f"worst={result['worst_event']}:{100*result['worst_event_target3']:.4f}",
        flush=True,
    )
    model.train()
    return result


def save_training_state(path: Path, model: nn.Module, optimizer: optim.Optimizer, optimizer_step: int, best_values: Dict[str, float], args: argparse.Namespace, event_vocabulary: Sequence[str]) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_step": int(optimizer_step),
            "best_values": dict(best_values),
            "args": vars(args),
            "event_vocabulary": list(event_vocabulary),
        },
        path,
    )


def maybe_save_best(save_dir: Path, key: str, value: float, model: nn.Module, optimizer_step: int, best_values: Dict[str, float], metadata: Dict[str, object]) -> None:
    if not np.isfinite(value) or value <= best_values[key]:
        return
    best_values[key] = float(value)
    filename = {
        "val_miou": "best_val_miou.pth",
        "val_damage_hf1": "best_val_damage_hf1.pth",
        "val_event_macro_target3": "best_val_event_macro_target3.pth",
    }[key]
    atomic_torch_save(model.state_dict(), save_dir / filename)
    (save_dir / filename.replace(".pth", ".json")).write_text(
        json.dumps({"optimizer_step": optimizer_step, "selection_key": key, "selection_value": value, **metadata}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"NEW BEST {key}: {value:.6f} at step {optimizer_step}", flush=True)


def main() -> None:
    args = parse_args()
    apply_preset(args)
    stages = parse_stages(args.residual_stages)
    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
    set_seed(args.seed, rank)
    try:
        effective_bs = args.train_batch_size_per_gpu * world_size * args.accumulation_steps
        if effective_bs != 16:
            raise ValueError(f"Expected effective batch size 16, got {effective_bs}")
        if args.max_iters != args.optimizer_steps * effective_bs:
            raise ValueError(
                f"max_iters must equal optimizer_steps*EBS={args.optimizer_steps*effective_bs}, got {args.max_iters}"
            )

        train_names = read_list(args.train_data_list_path)
        val_names = read_list(args.val_data_list_path)
        test_names = read_list(args.test_data_list_path)
        event_vocabulary, event_to_id = build_event_vocabulary(train_names)
        args.num_events = len(event_vocabulary)
        gaussian_micro_momentum = args.gaussian_step_momentum ** (1.0 / args.accumulation_steps)

        model = DamageFormer_IARM(
            roi_alpha=args.roi_alpha,
            detach_building_prior=True,
            use_pretrained_swin=not bool(args.resume or args.init_checkpoint),
            residual_stages=stages,
            residual_dim=args.residual_dim,
            flow_hidden_dim=args.flow_hidden_dim,
            max_offset_pixels=args.max_offset_pixels,
            detach_backbone_for_residual=args.detach_backbone_for_residual,
            enable_residual_fusion=args.enable_residual_fusion,
            residual_gamma_max=args.residual_gamma_max,
            num_events=len(event_vocabulary),
            gaussian_step_momentum=gaussian_micro_momentum,
            gaussian_variance_floor=args.gaussian_variance_floor,
            max_pixels_per_image_class=args.max_pixels_per_image_class,
        ).to(device)
        disable_unused_swin_heads(model)

        optimizer_step = 0
        best_values = {"val_miou": -1.0, "val_damage_hf1": -1.0, "val_event_macro_target3": -1.0}
        resume_obj: Dict[str, object] = {}
        if args.resume:
            resume_obj = load_resume(model, args.resume)
        elif args.init_checkpoint:
            load_initial_weights(model, args.init_checkpoint)

        # Buffers and randomly initialized new heads must begin identically.
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor):
                dist.broadcast(tensor, src=0)

        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        optimizer = optim.AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False)
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(args.model_param_path) / args.dataset / f"{args.model_type}_{args.experiment}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
            (save_dir / "event_vocabulary.json").write_text(json.dumps(event_vocabulary, indent=2, ensure_ascii=False), encoding="utf-8")
            print("IARM training configuration", flush=True)
            print(f"  experiment               : {args.experiment}", flush=True)
            print(f"  residual stages          : {stages}", flush=True)
            print(f"  residual fusion          : {args.enable_residual_fusion}", flush=True)
            print(f"  backbone detached branch : {args.detach_backbone_for_residual}", flush=True)
            print(f"  EBS / steps / crops      : {effective_bs} / {args.optimizer_steps} / {args.max_iters}", flush=True)
            print(f"  loss weights             : normal={args.normal_weight}, cmu={args.cmu_weight}, exist={args.damage_exist_weight}, anchor={args.anchor_weight}, xevent={args.cross_event_contrast_weight}", flush=True)
        else:
            save_dir = None

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        running = 0.0
        last_stats: Dict[str, float] = {}
        iterator = tqdm(loader, total=len(loader), disable=not is_main)

        for pre, post, labels_loc, labels_damage, sample_names in iterator:
            if optimizer_step >= args.optimizer_steps:
                break
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            labels_loc = labels_loc.to(device, non_blocking=True).long()
            labels_damage = labels_damage.to(device, non_blocking=True).long()
            event_ids = event_ids_from_batch(sample_names, event_to_id, device)

            sync_now = accumulation + 1 == args.accumulation_steps
            sync_context = nullcontext() if sync_now else ddp_model.no_sync()
            with sync_context:
                output_loc, output_damage, output_exist, aux = ddp_model(
                    pre,
                    post,
                    damage_target=labels_damage,
                    event_ids=event_ids,
                    current_iter=optimizer_step,
                    return_aux=True,
                )
                raw_loss, last_stats = compute_losses(
                    output_loc,
                    output_damage,
                    output_exist,
                    aux,
                    labels_loc,
                    labels_damage,
                    event_ids,
                    args,
                    stages,
                    rank,
                    world_size,
                )
                (raw_loss / args.accumulation_steps).backward()

            running += float(raw_loss.detach().item())
            accumulation += 1
            if accumulation < args.accumulation_steps:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            optimizer_step += 1

            if optimizer_step % args.log_every == 0:
                local_avg = running / (args.log_every * args.accumulation_steps)
                avg_tensor = torch.tensor(local_avg, device=device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg = float((avg_tensor / world_size).item())
                running = 0.0
                if is_main:
                    base_model = unwrap(ddp_model)
                    print(
                        f"step {optimizer_step}/{args.optimizer_steps} | loss {avg:.4f} | "
                        f"main {last_stats['main']:.4f} normal {last_stats['normal']:.4f} "
                        f"cmu {last_stats['cmu']:.4f} exist {last_stats['exist']:.4f} "
                        f"anchor {last_stats['anchor']:.4f} xevent {last_stats['xevent']:.4f} | "
                        f"gamma {float(base_model.residual_gamma().detach().item()):.6f} | "
                        f"gauss_ready {base_model.class_distribution_memory.ready()}",
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
                        max(1, args.num_workers // 2),
                        device,
                        "validation",
                    )
                    save_training_state(save_dir / "latest.pth", base_model, optimizer, optimizer_step, best_values, args, event_vocabulary)
                    metadata = {"validation": val_result}
                    maybe_save_best(save_dir, "val_miou", float(val_result["active"]["miou"]), base_model, optimizer_step, best_values, metadata)
                    maybe_save_best(save_dir, "val_damage_hf1", float(val_result["active"]["damage_hf1"]), base_model, optimizer_step, best_values, metadata)
                    maybe_save_best(save_dir, "val_event_macro_target3", float(val_result["event_macro_target3"]), base_model, optimizer_step, best_values, metadata)
                    (save_dir / "latest_metrics.json").write_text(json.dumps({"optimizer_step": optimizer_step, "best_values": best_values, "validation": val_result}, indent=2, ensure_ascii=False), encoding="utf-8")
                    if args.eval_test_during_training:
                        evaluate_split(base_model, args.test_dataset_path, test_names, args.eval_batch_size, max(1, args.num_workers // 2), device, "testing")
                dist.barrier()

        dist.barrier()
        if is_main:
            base_model = unwrap(ddp_model)
            primary = save_dir / "best_val_event_macro_target3.pth"
            if not primary.exists():
                primary = save_dir / "best_val_miou.pth"
            if primary.exists():
                state = torch.load(primary, map_location="cpu")
                base_model.load_state_dict(clean_state_dict(state), strict=True)
            final_test = evaluate_split(base_model, args.test_dataset_path, test_names, args.eval_batch_size, max(1, args.num_workers // 2), device, "final testing")
            (save_dir / "final_test.json").write_text(json.dumps(final_test, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Training completed. Output: {save_dir}", flush=True)
        dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
