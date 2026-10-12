# -*- coding: utf-8 -*-
"""Train the teacher-free heterogeneous RGB/SAR DamageFormer variants."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.dataset.event_balanced_sampling import (  # noqa: E402
    EventBalancedBatchSampler,
    event_balanced_weights,
    event_from_sample_name,
    expected_event_mass,
)
from bda_benchmark.model.DamageFormer_RGBSAR_Heterogeneous import (  # noqa: E402
    VALID_VARIANTS,
    VARIANTS,
    build_heterogeneous_damageformer,
    common_unique_regularization,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (  # noqa: E402
    fast_hist,
    lovasz_softmax,
    metrics_from_hist,
    seed_worker,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        # A failed PyTorch zip write otherwise leaves a large, unusable .tmp
        # file behind and can make a low-space failure progressively worse.
        temporary.unlink(missing_ok=True)
        raise


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_metrics: Mapping[str, float],
    args,
) -> Dict[str, object]:
    return {
        "protocol": getattr(
            model, "resume_protocol", "HeterogeneousDamageFormer_resume_v2"
        ),
        "experiment": getattr(
            model, "experiment_name", "DamageFormer_RGBSAR_Heterogeneous"
        ),
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_metrics": dict(best_metrics),
        "args": vars(args),
        "model_contract": (
            model.checkpoint_contract() if hasattr(model, "checkpoint_contract") else {}
        ),
    }


def best_weights_payload(
    model: nn.Module,
    step: int,
    best_metrics: Mapping[str, float],
    args,
) -> Dict[str, object]:
    """Model-selection artifact without AdamW/GradScaler training state."""

    return {
        "protocol": getattr(
            model,
            "best_weights_protocol",
            "HeterogeneousDamageFormer_best_weights_v2",
        ),
        "experiment": getattr(
            model, "experiment_name", "DamageFormer_RGBSAR_Heterogeneous"
        ),
        "variant": args.variant,
        "step": int(step),
        "model": model.state_dict(),
        "best_metrics": dict(best_metrics),
        "args": vars(args),
        "model_contract": (
            model.checkpoint_contract() if hasattr(model, "checkpoint_contract") else {}
        ),
    }


def segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    ce_weight: float,
    lovasz_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.float()
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    lovasz = lovasz_softmax(
        F.softmax(logits, dim=1),
        labels,
        classes="present",
        ignore=ignore_index,
    )
    return ce_weight * ce + lovasz_weight * lovasz, ce, lovasz


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    damage_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hist: Dict[str, torch.Tensor] = {}
    reliability_sum = 0.0
    reliability_count = 0
    auxiliary_metric_sum: Dict[str, float] = {}
    auxiliary_metric_count: Dict[str, int] = {}
    processed = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rgb, sar, location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()
        if hasattr(model, "set_batch_events"):
            model.set_batch_events([event_from_sample_name(name) for name in names])
        outputs = model(rgb, sar, return_dict=True)
        # SRAC partition quality cannot be inferred from four-class metrics:
        # SR1PA deliberately leaves the prediction exactly equal to P3.  Run
        # the read-only objective during validation so the checkpoint audit
        # records whether the learned partition is actually boundary aware.
        # The callback is side-effect free in eval mode (in particular, SR2
        # memory updates remain training-only).
        if (
            "srac_assignment" in outputs
            or "srac_affinity_logits" in outputs
        ):
            auxiliary = common_unique_regularization(
                outputs,
                damage,
                unchanged_weight=1.0,
                orthogonality_weight=0.0,
            )
            for key, value in auxiliary.items():
                if not key.startswith("srac_"):
                    continue
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    scalar = float(value.detach().float().item())
                    if math.isfinite(scalar):
                        auxiliary_metric_sum[key] = (
                            auxiliary_metric_sum.get(key, 0.0) + scalar
                        )
                        auxiliary_metric_count[key] = (
                            auxiliary_metric_count.get(key, 0) + 1
                        )
        loc_prediction = outputs["output_loc"].argmax(1)
        damage_prediction = outputs["output_dam"].argmax(1)
        loc_hist += fast_hist(loc_prediction, location, 2, ignore_index)
        damage_hist += fast_hist(damage_prediction, damage, 4, ignore_index)
        for key, value in outputs.items():
            if key.endswith("_reliability"):
                reliability_sum += float(value.float().mean().item())
                reliability_count += 1
        for sample_index in range(rgb.shape[0]):
            event = re.sub(r"_\d+$", "", str(names[sample_index]))
            sample_hist = fast_hist(
                damage_prediction[sample_index : sample_index + 1],
                damage[sample_index : sample_index + 1],
                4,
                ignore_index,
            )
            event_hist[event] = event_hist.get(
                event, torch.zeros_like(sample_hist)
            ) + sample_hist
        processed += int(rgb.shape[0])

    metrics = metrics_from_hist(loc_hist, damage_hist)
    location_tp = loc_hist[1, 1]
    metrics["loc_precision"] = float(
        (location_tp / loc_hist[:, 1].sum().clamp_min(1.0)).item()
    )
    metrics["loc_recall"] = float(
        (location_tp / loc_hist[1, :].sum().clamp_min(1.0)).item()
    )
    intact_total = damage_hist[1].sum().clamp_min(1.0)
    metrics["intact_to_damaged"] = float(
        (damage_hist[1, 2] / intact_total).item()
    )
    metrics["intact_to_destroyed"] = float(
        (damage_hist[1, 3] / intact_total).item()
    )
    damaged_total = damage_hist[2].sum().clamp_min(1.0)
    affected_total = damage_hist[2:4].sum().clamp_min(1.0)
    metrics["damaged_gate_miss"] = float(
        ((damage_hist[2, 0] + damage_hist[2, 1]) / damaged_total).item()
    )
    metrics["affected_recall"] = float(
        (damage_hist[2:4, 2:4].sum() / affected_total).item()
    )
    metrics["processed_images"] = processed
    event_values = []
    event_scores: Dict[str, float] = {}
    for event, histogram in event_hist.items():
        intersection = torch.diag(histogram)
        union = histogram.sum(1) + histogram.sum(0) - intersection
        score = float((intersection / union.clamp_min(1.0)).mean().item())
        event_values.append(score)
        event_scores[event] = score
    metrics["event_count"] = len(event_values)
    metrics["event_mean_official_style_4class_miou"] = (
        sum(event_values) / len(event_values) if event_values else 0.0
    )
    metrics["event_worst_official_style_4class_miou"] = (
        min(event_values) if event_values else 0.0
    )
    metrics["event_miou4"] = event_scores
    metrics["local_correspondence_reliability"] = (
        reliability_sum / reliability_count if reliability_count else 1.0
    )
    for key, total in auxiliary_metric_sum.items():
        metrics[key] = total / max(auxiliary_metric_count.get(key, 0), 1)
    candidate_nodes = auxiliary_metric_sum.get("srac_candidate_node_count", 0.0)
    if candidate_nodes > 0.0:
        # Keep the denominator itself as a dataset-level count.  Reporting
        # the per-batch mean here would make audits with different crop
        # compositions look artificially comparable.
        metrics["srac_candidate_node_count"] = candidate_nodes
        metrics["srac_background_leakage_fraction"] = (
            auxiliary_metric_sum.get("srac_background_leakage_sum", 0.0)
            / candidate_nodes
        )
        metrics["srac_candidate_building_fraction"] = (
            auxiliary_metric_sum.get("srac_candidate_building_sum", 0.0)
            / candidate_nodes
        )
    if "srac_boundary_f1" in metrics and "srac_building_iou" in metrics:
        # Human-readable primary value. checkpoint_selection_key applies the
        # full lexicographic structural policy declared above.
        metrics["srac_partition_quality"] = metrics["srac_boundary_f1"]
    affinity_same_count = auxiliary_metric_sum.get(
        "srac_affinity_same_count", 0.0
    )
    affinity_different_count = auxiliary_metric_sum.get(
        "srac_affinity_different_count", 0.0
    )
    affinity_damage_count = auxiliary_metric_sum.get(
        "srac_affinity_damage_boundary_count", 0.0
    )
    if affinity_same_count > 0.0:
        metrics["srac_affinity_same_accuracy"] = (
            auxiliary_metric_sum.get("srac_affinity_same_correct", 0.0)
            / affinity_same_count
        )
    if affinity_different_count > 0.0:
        metrics["srac_affinity_different_recall"] = (
            auxiliary_metric_sum.get("srac_affinity_different_correct", 0.0)
            / affinity_different_count
        )
    if affinity_same_count > 0.0 and affinity_different_count > 0.0:
        metrics["srac_affinity_balanced_accuracy"] = 0.5 * (
            metrics["srac_affinity_same_accuracy"]
            + metrics["srac_affinity_different_recall"]
        )
    if affinity_damage_count > 0.0:
        metrics["srac_affinity_damage_boundary_recall"] = (
            auxiliary_metric_sum.get(
                "srac_affinity_damage_boundary_correct", 0.0
            ) / affinity_damage_count
        )
    model.train()
    return metrics


def metrics_line(prefix: str, metrics: Mapping[str, float]) -> str:
    line = (
        f"{prefix} | valid_mIoU={100*metrics['valid_miou']:.4f} | "
        f"mIoU4={100*metrics['miou4']:.4f} | "
        f"Loc P/R/F1={100*metrics['loc_precision']:.2f}/"
        f"{100*metrics['loc_recall']:.2f}/{100*metrics['loc_f1']:.2f} | "
        f"IoU(int/dmg/des)={100*metrics['intact_iou']:.2f}/"
        f"{100*metrics['damaged_iou']:.2f}/{100*metrics['destroyed_iou']:.2f} | "
        f"Dmg->Int={100*metrics['damaged_to_intact']:.2f}% | "
        f"Dmg gate-miss={100*metrics['damaged_gate_miss']:.2f}% | "
        f"Des->Int={100*metrics['destroyed_to_intact']:.2f}% | "
        f"Int->Dmg={100*metrics['intact_to_damaged']:.2f}% | "
        f"Event mean/worst="
        f"{100*metrics['event_mean_official_style_4class_miou']:.2f}/"
        f"{100*metrics['event_worst_official_style_4class_miou']:.2f}"
    )
    if "srac_partition_entropy" in metrics:
        line += (
            " | SP ent/cand/leak="
            f"{metrics['srac_partition_entropy']:.3f}/"
            f"{metrics.get('srac_effective_candidates', 0.0):.2f}/"
            f"{100*metrics.get('srac_background_leakage_fraction', 0.0):.1f}%"
            " | SP building-IoU/boundary-F1="
            f"{100*metrics.get('srac_building_iou', 0.0):.2f}/"
            f"{100*metrics.get('srac_boundary_f1', 0.0):.2f}"
        )
    return line


def checkpoint_selection_key(
    metrics: Mapping[str, float], primary_metric: str
) -> tuple[float, ...]:
    """Deterministic validation-only selection with declared tie breakers."""
    if primary_metric == "event_worst_official_style_4class_miou":
        return (
            float(metrics.get(primary_metric, -1.0)),
            float(metrics.get("event_mean_official_style_4class_miou", -1.0)),
            float(metrics.get("damaged_iou", -1.0)),
            float(metrics.get("valid_miou", -1.0)),
        )
    if primary_metric == "srac_partition_quality":
        # Partition-only variants deliberately leave the P3 prediction
        # unchanged, so four-class/event metrics are all tied. Select the
        # structural checkpoint by boundary/footprint quality first and use
        # leakage/entropy only as lower-is-better tie breakers.
        return (
            float(metrics.get("srac_boundary_f1", -1.0)),
            float(metrics.get("srac_building_iou", -1.0)),
            -float(metrics.get("srac_background_leakage_fraction", 1.0)),
            -float(metrics.get("srac_partition_entropy", 1.0)),
        )
    if primary_metric == "srac_affinity_balanced_accuracy":
        return (
            float(metrics.get(primary_metric, -1.0)),
            float(metrics.get("srac_affinity_damage_boundary_recall", -1.0)),
            float(metrics.get("srac_affinity_different_recall", -1.0)),
            float(metrics.get("srac_affinity_same_accuracy", -1.0)),
        )
    return (float(metrics.get(primary_metric, -1.0)),)


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    rgb_parameters = []
    sar_parameters = []
    joint_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("rgb_encoder"):
            rgb_parameters.append(parameter)
        elif name.startswith("sar_encoder.style_blocks"):
            # Event-style adapters are newly introduced fusion parameters,
            # even though their execution point is inside the SAR encoder.
            joint_parameters.append(parameter)
        elif name.startswith("sar_encoder"):
            sar_parameters.append(parameter)
        else:
            joint_parameters.append(parameter)
    groups = []
    if rgb_parameters:
        groups.append({"params": rgb_parameters, "lr": args.rgb_learning_rate})
    if sar_parameters:
        groups.append({"params": sar_parameters, "lr": args.sar_learning_rate})
    if joint_parameters:
        groups.append({"params": joint_parameters, "lr": args.fusion_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def load_resume(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    require_optimizer: bool,
) -> tuple[int, Dict[str, float]]:
    if not path:
        return 0, {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != model.variant:
        raise RuntimeError(
            f"resume variant {checkpoint.get('variant')} != model {model.variant}"
        )
    observed_contract = checkpoint.get("model_contract", {})
    expected_contract = (
        model.checkpoint_contract() if hasattr(model, "checkpoint_contract") else {}
    )
    if observed_contract and observed_contract != expected_contract:
        raise RuntimeError(
            "resume model contract mismatch: "
            f"checkpoint={observed_contract}, model={expected_contract}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
    elif require_optimizer:
        raise RuntimeError(
            f"{path} is a weights-only best checkpoint; resume training from "
            "latest_resume.pth instead"
        )
    return int(checkpoint.get("step", 0)), dict(checkpoint.get("best_metrics", {}))


def main() -> None:
    parser = argparse.ArgumentParser("Train heterogeneous RGB/SAR DamageFormer")
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="h3")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=8000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument(
        "--keep_milestone_checkpoints",
        action="store_true",
        help="also retain checkpoint_step_N.pth files; disabled by default",
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--rgb_learning_rate", type=float, default=1e-5)
    parser.add_argument("--sar_learning_rate", type=float, default=1e-4)
    parser.add_argument("--fusion_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--loc_ce_weight", type=float, default=1.0)
    parser.add_argument("--loc_lovasz_weight", type=float, default=0.5)
    parser.add_argument("--damage_ce_weight", type=float, default=1.0)
    parser.add_argument("--damage_lovasz_weight", type=float, default=0.75)
    parser.add_argument("--sar_aux_weight", type=float, default=0.25)
    parser.add_argument("--representation_weight", type=float, default=0.10)
    parser.add_argument("--common_unchanged_weight", type=float, default=1.0)
    parser.add_argument("--orthogonality_weight", type=float, default=0.05)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--event_balanced_sampling",
        action="store_true",
        help=(
            "Train-only inverse-event-frequency sampling. Each disaster event "
            "has equal expected image mass; validation and test remain unchanged."
        ),
    )
    parser.add_argument("--no_pretrained_rgb", action="store_true")
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument(
        "--selection_metric",
        choices=(
            "valid_miou",
            "event_mean_official_style_4class_miou",
            "event_worst_official_style_4class_miou",
            "damaged_iou",
            "srac_partition_quality",
            "srac_affinity_balanced_accuracy",
        ),
        default="valid_miou",
        help="Validation-only primary checkpoint selector; test is never eligible.",
    )
    parser.add_argument(
        "--select_initial_checkpoint",
        action="store_true",
        help=(
            "Allow the strict step-0 validation state to compete in model "
            "selection. Required by zero-gated safe-initialisation ablations."
        ),
    )
    parser.add_argument(
        "--error_on_nonfinite_grad",
        action="store_true",
        help="Stop immediately if gradient clipping observes NaN or Inf.",
    )
    args = parser.parse_args()

    if args.variant != "h0" and not args.sar_stats_path:
        parser.error("H1-H3 require --sar_stats_path")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_heterogeneous_damageformer(
        args.variant, pretrained_rgb=not args.no_pretrained_rgb
    ).to(device)
    optimizer = make_optimizer(model, args)
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and args.amp == "fp16"
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and args.amp == "fp16"
        )
    start_step, best_metrics = load_resume(
        args.resume,
        model,
        optimizer,
        scaler,
        require_optimizer=not args.validation_only,
    )
    best_step = None
    best_checkpoint_name = {
        "valid_miou": "best_val_miou.pth",
        "event_mean_official_style_4class_miou": "best_event_mean.pth",
        "event_worst_official_style_4class_miou": "best_event_worst.pth",
        "damaged_iou": "best_damaged_iou.pth",
        "srac_partition_quality": "best_partition.pth",
        "srac_affinity_balanced_accuracy": "best_affinity.pth",
    }[args.selection_metric]
    best_checkpoint_path = output_dir / best_checkpoint_name
    if best_checkpoint_path.is_file():
        existing_best = torch.load(
            best_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        best_step = int(existing_best.get("step", start_step))

    total_crops = max(
        args.train_batch_size,
        args.optimizer_steps
        * args.train_batch_size
        * args.accumulation_steps,
    )
    train_dataset = HeterogeneousRGBSARDataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        max_iters=total_crops,
        split="train",
        variant=args.variant,
        sar_stats_path=args.sar_stats_path,
    )
    val_dataset = HeterogeneousRGBSARDataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        split="val",
        variant=args.variant,
        sar_stats_path=args.sar_stats_path,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_batch_sampler = None
    sampling_report = {
        "mode": "image_uniform",
        "event_counts": {},
        "expected_event_mass": {},
    }
    if args.event_balanced_sampling:
        _, event_counts = event_balanced_weights(train_dataset.data_list)
        train_batch_sampler = EventBalancedBatchSampler(
            train_dataset.data_list,
            batch_size=args.train_batch_size,
            num_batches=len(train_dataset) // args.train_batch_size,
            seed=args.seed,
        )
        sampling_report = {
            "mode": "event_balanced",
            "event_counts": event_counts,
            "expected_event_mass": expected_event_mass(train_dataset.data_list),
        }
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print("=" * 80, flush=True)
    print(f"Experiment       : {VARIANTS[args.variant].name}", flush=True)
    print(f"Variant          : {args.variant}", flush=True)
    print(f"Train sampling   : {sampling_report['mode']}", flush=True)
    if args.event_balanced_sampling:
        print(
            "Train events     : "
            + ", ".join(
                f"{event}={count}"
                for event, count in sampling_report["event_counts"].items()
            ),
            flush=True,
        )
    print("Teacher          : NONE", flush=True)
    print("RGB localization : RGB branch only", flush=True)
    print(
        f"Steps/EBS/crops  : {args.optimizer_steps}/"
        f"{args.train_batch_size*args.accumulation_steps}/{total_crops}",
        flush=True,
    )
    print(
        f"LR RGB/SAR/fusion: {args.rgb_learning_rate}/"
        f"{args.sar_learning_rate}/{args.fusion_learning_rate}",
        flush=True,
    )
    print(
        "Checkpoint policy: best weights + latest resume"
        + (" + milestones" if args.keep_milestone_checkpoints else ""),
        flush=True,
    )
    print(f"Output           : {output_dir}", flush=True)
    print("=" * 80, flush=True)

    initial_metrics = evaluate(
        model, val_loader, device, args.ignore_index, args.val_max_batches
    )
    max_observed_valid_miou = float(initial_metrics.get("valid_miou", -1.0))
    max_observed_valid_step = int(start_step)
    print(metrics_line("Initial validation", initial_metrics), flush=True)
    if args.validation_only:
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8"
        )
        return

    if args.select_initial_checkpoint and (
        not best_metrics
        or checkpoint_selection_key(initial_metrics, args.selection_metric)
        > checkpoint_selection_key(best_metrics, args.selection_metric)
    ):
        best_metrics = dict(initial_metrics)
        best_step = int(start_step)
        atomic_save(
            best_weights_payload(model, start_step, best_metrics, args),
            best_checkpoint_path,
        )
        print(
            "Initial state admitted to selection | "
            f"step={start_step} | {args.selection_metric}="
            f"{100*best_metrics[args.selection_metric]:.4f}",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    running: Dict[str, float] = {}
    window_start = time.time()
    train_iterator = iter(train_loader)
    while step < args.optimizer_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        rgb, sar, location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        location = location.to(device, non_blocking=True).long()
        damage = damage.to(device, non_blocking=True).long()
        if hasattr(model, "set_optimizer_progress"):
            model.set_optimizer_progress(step, args.optimizer_steps)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            if hasattr(model, "set_batch_events"):
                model.set_batch_events(
                    [event_from_sample_name(name) for name in names]
                )
            outputs = model(rgb, sar, return_dict=True)
            loc_loss, loc_ce, loc_lovasz = segmentation_loss(
                outputs["output_loc"],
                location,
                args.ignore_index,
                args.loc_ce_weight,
                args.loc_lovasz_weight,
            )
            damage_loss, damage_ce, damage_lovasz = segmentation_loss(
                outputs["output_dam"],
                damage,
                args.ignore_index,
                args.damage_ce_weight,
                args.damage_lovasz_weight,
            )
            sar_aux_loss = outputs["output_dam"].sum() * 0.0
            if "sar_aux_logits" in outputs:
                sar_aux_loss = F.cross_entropy(
                    outputs["sar_aux_logits"].float(),
                    damage,
                    ignore_index=args.ignore_index,
                )
            representation = common_unique_regularization(
                outputs,
                damage,
                unchanged_weight=args.common_unchanged_weight,
                orthogonality_weight=args.orthogonality_weight,
            )
            total_loss = (
                loc_loss
                + damage_loss
                + args.sar_aux_weight * sar_aux_loss
                + args.representation_weight * representation["representation_loss"]
            )
            scaled_loss = total_loss / args.accumulation_steps
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"non-finite loss before optimizer step {step + 1}: "
                f"{float(total_loss.detach().item())}"
            )
        scaler.scale(scaled_loss).backward()

        micro_step = int(running.get("_micro_step", 0.0)) + 1
        running["_micro_step"] = float(micro_step)
        values = {
            "total_loss": total_loss,
            "loc_loss": loc_loss,
            "damage_loss": damage_loss,
            "loc_ce_loss": loc_ce,
            "loc_lovasz_loss": loc_lovasz,
            "damage_ce_loss": damage_ce,
            "damage_lovasz_loss": damage_lovasz,
            "sar_aux_loss": sar_aux_loss,
            **representation,
        }
        for key, value in values.items():
            running[key] = running.get(key, 0.0) + float(value.detach().item())
        if micro_step % args.accumulation_steps:
            continue

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
            error_if_nonfinite=args.error_on_nonfinite_grad,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            denominator = args.log_every * args.accumulation_steps
            elapsed = max(time.time() - window_start, 1e-6)
            payload = {
                "step": step,
                "variant": args.variant,
                **{
                    key: value / denominator
                    for key, value in running.items()
                    if not key.startswith("_")
                },
                "optimizer_steps_per_second": args.log_every / elapsed,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            print(json.dumps(payload, allow_nan=False), flush=True)
            running = {"_micro_step": float(micro_step)}
            window_start = time.time()

        should_evaluate = step % args.eval_every == 0 or step == args.optimizer_steps
        if should_evaluate:
            metrics = evaluate(
                model, val_loader, device, args.ignore_index, args.val_max_batches
            )
            print(metrics_line("Validation", metrics), flush=True)
            if float(metrics.get("valid_miou", -1.0)) > max_observed_valid_miou:
                max_observed_valid_miou = float(metrics["valid_miou"])
                max_observed_valid_step = int(step)
            if (
                not best_metrics
                or checkpoint_selection_key(metrics, args.selection_metric)
                > checkpoint_selection_key(best_metrics, args.selection_metric)
            ):
                best_metrics = metrics
                best_step = int(step)
                atomic_save(
                    best_weights_payload(model, step, best_metrics, args),
                    best_checkpoint_path,
                )
                print(
                    f"New best {args.selection_metric}="
                    f"{100*metrics[args.selection_metric]:.4f}",
                    flush=True,
                )
            (output_dir / "latest_validation.json").write_text(
                json.dumps({"step": step, "metrics": metrics}, indent=2) + "\n",
                encoding="utf-8",
            )

        if step % args.save_every == 0 or step == args.optimizer_steps:
            payload = checkpoint_payload(
                model, optimizer, scaler, step, best_metrics, args
            )
            atomic_save(payload, output_dir / "latest_resume.pth")
            if args.keep_milestone_checkpoints:
                atomic_save(
                    payload, output_dir / f"checkpoint_step_{step}.pth"
                )

    summary = {
        "variant": args.variant,
        "completed_step": step,
        "selected_step": best_step,
        "selection_metric": args.selection_metric,
        # Kept for backward compatibility: this is the valid mIoU belonging
        # to the checkpoint selected by selection_metric, not necessarily the
        # maximum valid mIoU observed anywhere on the curve.
        "best_valid_miou": best_metrics.get("valid_miou"),
        "selected_valid_miou": best_metrics.get("valid_miou"),
        "max_observed_valid_miou": max_observed_valid_miou,
        "max_observed_valid_step": max_observed_valid_step,
        "best_metrics": best_metrics,
        "training_sampling": sampling_report,
        "model_contract": (
            model.checkpoint_contract() if hasattr(model, "checkpoint_contract") else {}
        ),
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Training complete | "
        f"selected_step={best_step} | selection_metric={args.selection_metric} | "
        f"selected_score={100*best_metrics.get(args.selection_metric, 0.0):.4f} | "
        f"selected_valid_mIoU={100*best_metrics.get('valid_miou', 0.0):.4f} | "
        f"max_observed_valid_mIoU={100*max_observed_valid_miou:.4f} "
        f"@step={max_observed_valid_step}",
        flush=True,
    )


if __name__ == "__main__":
    main()
