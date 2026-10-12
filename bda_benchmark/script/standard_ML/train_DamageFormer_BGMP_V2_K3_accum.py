# -*- coding: utf-8 -*-
"""Experiment-locked BGMP-V2 K=3 hard-assignment training.

This script keeps the same BGMP-V2 model, CE+Lovasz objective, optimizer,
gradient accumulation, warm-up and checkpoint policy as the K=1 run.
The only experiment change is K=3 with spherical hard assignment and
empty-prototype reinitialization.

Validation-selected checkpoints only; test is never used for selection.

Original notes:

Primary checkpoint selection:
- best_val_miou.pth: highest validation mIoU.

Additional analysis checkpoint:
- best_val_destroyed_iou.pth: highest validation destroyed IoU.

The test split is NOT evaluated every validation interval and is NOT used to
select checkpoints. At the end, validation-selected checkpoints are evaluated
on test. This avoids test-set model selection.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

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
from model.DamageFormer_BGMP_V2 import DamageFormer_BGMP_V2
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L

CLASS_NAMES = ["background", "intact", "damaged", "destroyed"]
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DamageFormer BGMP-V2 K3 hard-assignment training")
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
    p.add_argument("--max_iters", type=int, default=800000)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=0)

    p.add_argument("--roi_alpha", type=float, default=0.1)
    p.add_argument("--proto_k", type=int, default=3, help="Locked to 3 in this S3 experiment")
    p.add_argument("--proto_step_momentum", type=float, default=0.99)
    p.add_argument("--proto_tau", type=float, default=0.1)
    p.add_argument("--proto_warmup_iters", type=int, default=5000)
    p.add_argument(
        "--proto_assignment_mode",
        choices=["auto", "mean", "hard"],
        default="hard",
        help="Locked to hard for K=3",
    )

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_type", type=str, default="S3V2_BGMP_K3_direct256_hard")
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


def compute_loss(output_loc, output_clf, labels_loc, labels_clf):
    ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
    lovasz_loss_loc = L.lovasz_softmax(
        F.softmax(output_loc, dim=1), labels_loc, ignore=255
    )
    ce_loss_clf = F.cross_entropy(output_clf, labels_clf)
    lovasz_loss_clf = L.lovasz_softmax(
        F.softmax(output_clf, dim=1), labels_clf, ignore=255
    )
    total = ce_loss_loc + ce_loss_clf + 0.75 * lovasz_loss_clf + 0.5 * lovasz_loss_loc
    return total, {
        "ce_loc": float(ce_loss_loc.detach().item()),
        "lovasz_loc": float(lovasz_loss_loc.detach().item()),
        "ce_clf": float(ce_loss_clf.detach().item()),
        "lovasz_clf": float(lovasz_loss_clf.detach().item()),
    }


def harmonic_damage_f1(evaluator: Evaluator) -> float:
    values = np.asarray(evaluator.Damage_F1_score(), dtype=np.float64)
    if np.any(values <= 0):
        return 0.0
    return float(len(values) / np.sum(1.0 / values))


def row_normalized_confusion(cm: np.ndarray) -> np.ndarray:
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0


def proto_bank_diagnostics(model: DamageFormer_BGMP_V2) -> Dict[str, Any]:
    memory = model.damage_proto_memory
    proto = memory.prototypes.detach()
    normalized = F.normalize(proto, dim=-1)
    norms = proto.norm(dim=-1)
    class_mean = F.normalize(normalized.mean(dim=1), dim=-1)
    interclass = class_mean @ class_mean.t()

    same_class_cos = []
    for class_idx in range(proto.shape[0]):
        if proto.shape[1] < 2:
            same_class_cos.append(None)
            continue
        matrix = normalized[class_idx] @ normalized[class_idx].t()
        mask = ~torch.eye(proto.shape[1], device=proto.device, dtype=torch.bool)
        same_class_cos.append(float(matrix[mask].mean().item()))

    usage = memory.prototype_usage.detach()
    usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return {
        "shape": list(proto.shape),
        "assignment_mode": memory.resolved_assignment_mode(),
        "initialized": memory.prototype_initialized.detach().cpu().tolist(),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "interclass_cosine": interclass.detach().cpu().tolist(),
        "same_class_cosine_mean": same_class_cos,
        "usage_ratio": usage_ratio.detach().cpu().tolist(),
        "empty_reinit_count": memory.empty_reinit_count.detach().cpu().tolist(),
        "gamma": float(memory.gamma().detach().item()),
    }


@torch.no_grad()
def evaluate_split(
    model: DamageFormer_BGMP_V2,
    dataset_path: str,
    data_names: list[str],
    eval_batch_size: int,
    device: torch.device,
    num_workers: int,
    split_name: str,
    current_iter: int,
) -> Dict[str, Any]:
    print(f"---------starting {split_name}-----------", flush=True)
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    proto_cm = np.zeros((3, 3), dtype=np.int64)

    query_margin_sum = 0.0
    attention_entropy_sum = 0.0
    effect_ratio_sum = 0.0
    query_batches = 0

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

        output_loc, output_clf, aux = model(
            pre,
            post,
            current_iter=current_iter,
            return_aux=True,
        )
        pred_loc = output_loc.argmax(dim=1).cpu().numpy()
        pred_clf = output_clf.argmax(dim=1).cpu().numpy()
        gt_loc = labels_loc_t.cpu().numpy()
        gt_clf = labels_clf_t.cpu().numpy()

        evaluator_loc.add_batch(gt_loc, pred_loc)
        evaluator_clf.add_batch(gt_clf[gt_loc > 0], pred_clf[gt_loc > 0])
        evaluator_total.add_batch(gt_clf, pred_clf)

        sim = aux.get("proto_class_similarity")
        if sim is not None:
            target = F.interpolate(
                labels_clf_t.unsqueeze(1).float(),
                size=aux["final_feat_fused_proto"].shape[-2:],
                mode="nearest",
            )[:, 0].long().reshape(-1)
            valid = (target >= 1) & (target <= 3)
            if bool(valid.any().item()):
                pred_proto = sim.argmax(dim=1) + 1
                gt_proto = target[valid] - 1
                pred_proto_valid = pred_proto[valid] - 1
                encoded = 3 * gt_proto + pred_proto_valid
                proto_cm += torch.bincount(
                    encoded, minlength=9
                ).reshape(3, 3).cpu().numpy()

            margin = float(aux["proto_query_margin"].item())
            entropy = float(aux["proto_attention_entropy"].item())
            effect = float(aux["proto_effect_ratio"].item())
            if math.isfinite(margin) and math.isfinite(entropy):
                query_margin_sum += margin
                attention_entropy_sum += entropy
                effect_ratio_sum += effect
                query_batches += 1

    loc_f1 = float(evaluator_loc.Pixel_F1_score())
    clf_f1 = harmonic_damage_f1(evaluator_clf)
    oa = float(evaluator_total.Pixel_Accuracy())
    ious = np.asarray(evaluator_total.Intersection_over_Union(), dtype=np.float64)
    miou = float(evaluator_total.Mean_Intersection_over_Union())
    cm = evaluator_total.confusion_matrix.astype(np.int64)
    cm_row = row_normalized_confusion(cm)

    proto_total = proto_cm.sum()
    proto_acc = float(np.trace(proto_cm) / proto_total) if proto_total > 0 else float("nan")
    proto_row = row_normalized_confusion(proto_cm) if proto_total > 0 else np.zeros((3, 3))

    result: Dict[str, Any] = {
        "loc_f1_percent": 100.0 * loc_f1,
        "clf_f1_percent": 100.0 * clf_f1,
        "oa_percent": 100.0 * oa,
        "miou_percent": 100.0 * miou,
        "iou_percent": {
            name: 100.0 * float(value) for name, value in zip(CLASS_NAMES, ious)
        },
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_normalized_percent": cm_row.tolist(),
        "gt_destroyed_prediction_percent": {
            name: float(cm_row[3, idx]) for idx, name in enumerate(CLASS_NAMES)
        },
        "prototype_class_accuracy_percent": 100.0 * proto_acc if math.isfinite(proto_acc) else None,
        "prototype_class_confusion_counts": proto_cm.tolist(),
        "prototype_class_confusion_row_normalized_percent": proto_row.tolist(),
        "proto_query_margin_mean": query_margin_sum / query_batches if query_batches else None,
        "proto_attention_entropy_mean": attention_entropy_sum / query_batches if query_batches else None,
        "proto_effect_ratio_mean": effect_ratio_sum / query_batches if query_batches else None,
        "prototype_bank": proto_bank_diagnostics(model),
    }
    print(
        f"{split_name}: loc_F1={result['loc_f1_percent']:.4f}, "
        f"clf_F1={result['clf_f1_percent']:.4f}, OA={result['oa_percent']:.4f}, "
        f"mIoU={result['miou_percent']:.4f}",
        flush=True,
    )
    print("IoU:", result["iou_percent"], flush=True)
    print("GT destroyed prediction row (%):", result["gt_destroyed_prediction_percent"], flush=True)
    print(
        "Prototype diagnostics: "
        f"cls_acc={result['prototype_class_accuracy_percent']} | "
        f"margin={result['proto_query_margin_mean']} | "
        f"attn_entropy={result['proto_attention_entropy_mean']} | "
        f"effect_ratio={result['proto_effect_ratio_mean']}",
        flush=True,
    )
    return result


def save_state(model: DamageFormer_BGMP_V2, path: Path) -> None:
    torch.save(model.state_dict(), path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.proto_k != 3:
        raise ValueError(f"This experiment is locked to proto_k=3, got {args.proto_k}")
    if args.proto_assignment_mode != "hard":
        raise ValueError(
            "This experiment is locked to --proto_assignment_mode hard"
        )
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    effective_bs = args.train_batch_size * args.accumulation_steps
    expected_crops = effective_bs * args.optimizer_steps
    if effective_bs != 16:
        raise ValueError(
            f"Paper-equivalent run requires effective batch size 16, got {effective_bs}."
        )
    if args.max_iters != expected_crops:
        raise ValueError(
            f"Schedule mismatch: max_iters must be {expected_crops}, got {args.max_iters}."
        )
    if not 0.0 < args.proto_step_momentum < 1.0:
        raise ValueError("proto_step_momentum must be in (0,1)")

    proto_micro_momentum = args.proto_step_momentum ** (
        1.0 / args.accumulation_steps
    )

    train_names = read_list(args.train_data_list_path)
    val_names = read_list(args.val_data_list_path)
    test_names = read_list(args.test_data_list_path)

    model = DamageFormer_BGMP_V2(
        roi_alpha=args.roi_alpha,
        proto_k=args.proto_k,
        proto_momentum=proto_micro_momentum,
        proto_tau=args.proto_tau,
        proto_warmup_iters=args.proto_warmup_iters,
        proto_assignment_mode=args.proto_assignment_mode,
    ).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

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
    save_dir = (
        Path(args.model_param_path)
        / args.dataset
        / f"{args.model_type}_{now}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "validation_history.jsonl"

    print("Training schedule", flush=True)
    print("  GPU                    : 1", flush=True)
    print(f"  micro batch            : {args.train_batch_size}", flush=True)
    print(f"  accumulation_steps     : {args.accumulation_steps}", flush=True)
    print(f"  effective batch size   : {effective_bs}", flush=True)
    print(f"  optimizer steps        : {args.optimizer_steps}", flush=True)
    print(f"  training crops         : {args.max_iters}", flush=True)
    print(f"  proto K / dim          : {args.proto_k} / 256 (direct feature)", flush=True)
    print(f"  proto assignment       : {model.damage_proto_memory.resolved_assignment_mode()}", flush=True)
    print(f"  proto tau              : {args.proto_tau}", flush=True)
    print(f"  proto warmup opt steps : {args.proto_warmup_iters}", flush=True)
    print(f"  proto EMA m_step       : {args.proto_step_momentum:.8f}", flush=True)
    print(f"  proto EMA m_micro      : {proto_micro_momentum:.8f}", flush=True)
    print(f"  save directory         : {save_dir}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    accum_counter = 0
    best_val_miou = -1.0
    best_val_destroyed = -1.0
    best_val_miou_step = -1
    best_val_destroyed_step = -1
    no_improve = 0
    running_raw_loss = 0.0

    for pre, post, labels_loc, labels_clf, _ in tqdm(loader, total=len(loader)):
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
        raw_loss, loss_parts = compute_loss(
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
            avg_loss = running_raw_loss / (
                args.log_every * args.accumulation_steps
            )
            bank = proto_bank_diagnostics(model)
            print(
                f"opt_step {optimizer_step}/{args.optimizer_steps} | "
                f"loss {avg_loss:.4f} | proto_active {aux['proto_active']} | "
                f"proto_init {aux['proto_initialized'].tolist()} | "
                f"mode {aux['proto_assignment_mode']} | gamma {float(aux['proto_gamma']):.4f} | "
                f"margin {float(aux['proto_query_margin']):.6f} | "
                f"attn_entropy {float(aux['proto_attention_entropy']):.6f} | "
                f"effect_ratio {float(aux['proto_effect_ratio']):.6f} | "
                f"usage {aux['proto_usage_ratio'].detach().cpu().tolist()} | "
                f"same_class_cos {bank['same_class_cosine_mean']}",
                flush=True,
            )
            running_raw_loss = 0.0

        stop_now = False
        if optimizer_step % args.eval_every == 0:
            val = evaluate_split(
                model,
                args.val_dataset_path,
                val_names,
                args.eval_batch_size,
                device,
                max(1, args.num_workers // 2),
                "validation/active_state",
                current_iter=optimizer_step,
            )
            record = {"optimizer_step": optimizer_step, **val}
            append_jsonl(history_path, record)

            improved_miou = val["miou_percent"] > best_val_miou
            if improved_miou:
                best_val_miou = float(val["miou_percent"])
                best_val_miou_step = optimizer_step
                no_improve = 0
                save_state(model, save_dir / "best_val_miou.pth")
                (save_dir / "best_val_miou_info.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"NEW BEST VAL mIoU={best_val_miou:.4f} at step={optimizer_step}",
                    flush=True,
                )
            else:
                no_improve += 1
                print(
                    f"No new val mIoU best: {no_improve} validation(s); "
                    f"best={best_val_miou:.4f} at step={best_val_miou_step}",
                    flush=True,
                )

            destroyed_iou = float(val["iou_percent"]["destroyed"])
            if destroyed_iou > best_val_destroyed:
                best_val_destroyed = destroyed_iou
                best_val_destroyed_step = optimizer_step
                save_state(model, save_dir / "best_val_destroyed_iou.pth")
                (save_dir / "best_val_destroyed_iou_info.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"NEW BEST VAL destroyed IoU={best_val_destroyed:.4f} "
                    f"at step={optimizer_step}",
                    flush=True,
                )

            if (
                args.early_stop_patience > 0
                and no_improve >= args.early_stop_patience
            ):
                stop_now = True
                print(
                    f"EARLY STOP: validation mIoU did not improve for "
                    f"{args.early_stop_patience} validations.",
                    flush=True,
                )
            model.train()

        if stop_now or optimizer_step >= args.optimizer_steps:
            break

    save_state(model, save_dir / "last_model.pth")

    if best_val_miou_step < 0:
        raise RuntimeError("No validation checkpoint was saved")

    final_summary: Dict[str, Any] = {
        "selection_policy": (
            "Primary model is selected only by validation mIoU. "
            "Test is evaluated after training and is not a checkpoint-selection criterion."
        ),
        "best_val_miou_step": best_val_miou_step,
        "best_val_miou_percent": best_val_miou,
        "best_val_destroyed_iou_step": best_val_destroyed_step,
        "best_val_destroyed_iou_percent": best_val_destroyed,
        "stopped_at_optimizer_step": optimizer_step,
    }

    # Primary paper checkpoint: selected by validation mIoU.
    state = torch.load(save_dir / "best_val_miou.pth", map_location="cpu")
    model.load_state_dict(state, strict=True)
    primary_selected_state = evaluate_split(
        model,
        args.test_dataset_path,
        test_names,
        args.eval_batch_size,
        device,
        max(1, args.num_workers // 2),
        "testing/best_val_miou/selected_state",
        current_iter=best_val_miou_step,
    )
    primary_active = evaluate_split(
        model,
        args.test_dataset_path,
        test_names,
        args.eval_batch_size,
        device,
        max(1, args.num_workers // 2),
        "testing/best_val_miou/forced_active",
        current_iter=args.proto_warmup_iters,
    )
    primary_inactive = evaluate_split(
        model,
        args.test_dataset_path,
        test_names,
        args.eval_batch_size,
        device,
        max(1, args.num_workers // 2),
        "testing/best_val_miou/forced_inactive",
        current_iter=0,
    )
    final_summary["best_val_miou_test_selected_state"] = primary_selected_state
    final_summary["best_val_miou_test_forced_active"] = primary_active
    final_summary["best_val_miou_test_forced_inactive"] = primary_inactive

    # Secondary targeted checkpoint: selected by VALIDATION destroyed IoU.
    state = torch.load(
        save_dir / "best_val_destroyed_iou.pth", map_location="cpu"
    )
    model.load_state_dict(state, strict=True)
    destroyed_selected_test = evaluate_split(
        model,
        args.test_dataset_path,
        test_names,
        args.eval_batch_size,
        device,
        max(1, args.num_workers // 2),
        "testing/best_val_destroyed_iou/selected_state",
        current_iter=best_val_destroyed_step,
    )
    final_summary["best_val_destroyed_iou_test_selected_state"] = destroyed_selected_test

    (save_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("FINAL SUMMARY", json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
