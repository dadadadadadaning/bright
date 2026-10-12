# -*- coding: utf-8 -*-
"""Evaluate saved BRIGHT DamageFormer checkpoints on validation/test splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"
for import_root in (PROJECT_ROOT, BDA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BG import DamageFormer_BG
from model.DamageFormer_BGMP import DamageFormer_BGMP
from util_func.metrics import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BRIGHT DamageFormer checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_variant", choices=["auto", "s1", "s2", "s1c"], default="auto")
    p.add_argument("--roi_alpha", type=float, default=0.1)
    p.add_argument("--guidance_beta", type=float, default=1.0)
    p.add_argument("--proto_tau", type=float, default=0.1)
    p.add_argument("--proto_warmup_iters", type=int, default=5000)
    p.add_argument("--proto_eval_mode", choices=["active", "inactive", "both"], default="both")
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--val_dataset_path", type=str, required=True)
    p.add_argument("--val_data_list_path", type=str, required=True)
    p.add_argument("--test_dataset_path", type=str, required=True)
    p.add_argument("--test_data_list_path", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def read_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_state(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


def infer_variant(state: Dict[str, torch.Tensor], requested: str) -> str:
    if requested != "auto":
        return requested
    return "s2" if "damage_proto_memory.prototypes" in state else "s1"


def build_model(args: argparse.Namespace, state: Dict[str, torch.Tensor], variant: str) -> torch.nn.Module:
    if variant == "s1":
        model = DamageFormer_BG(roi_alpha=args.roi_alpha, use_pretrained_swin=False)
    elif variant == "s1c":
        from model.DamageFormer_S1C import DamageFormer_S1C
        model = DamageFormer_S1C(guidance_beta=args.guidance_beta, use_pretrained_swin=False)
    elif variant == "s2":
        proto = state.get("damage_proto_memory.prototypes")
        if proto is None or proto.ndim != 3:
            raise KeyError("BGMP checkpoint does not contain damage_proto_memory.prototypes")
        proto_k = int(proto.shape[1])
        proto_dim = int(proto.shape[2])
        model = DamageFormer_BGMP(
            roi_alpha=args.roi_alpha,
            use_pretrained_swin=False,
            proto_k=proto_k,
            proto_dim=proto_dim,
            proto_momentum=0.99,
            proto_tau=args.proto_tau,
            proto_warmup_iters=args.proto_warmup_iters,
        )
        print(f"Detected BGMP prototype bank: classes={proto.shape[0]}, K={proto_k}, dim={proto_dim}", flush=True)
    else:
        raise ValueError(variant)
    model.load_state_dict(state, strict=True)
    return model


def harmonic_damage_f1(evaluator: Evaluator) -> float:
    values = np.asarray(evaluator.Damage_F1_score(), dtype=np.float64)
    if np.any(values <= 0):
        return 0.0
    return float(len(values) / np.sum(1.0 / values))


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataset_path: str,
    data_names: list[str],
    eval_batch_size: int,
    device: torch.device,
    num_workers: int,
    split_name: str,
    current_iter: int | None = None,
    is_bgmp: bool = False,
) -> Dict[str, Any]:
    print(f"--------- starting {split_name} ---------", flush=True)
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    dataset = MultimodalDamageAssessmentDatset(dataset_path, data_names, 1024, None, "test")
    loader = DataLoader(dataset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True, drop_last=False)

    model.eval()
    for pre, post, labels_loc, labels_clf, _ in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc_t = labels_loc.to(device, non_blocking=True).long()
        labels_clf_t = labels_clf.to(device, non_blocking=True).long()
        if is_bgmp:
            output_loc, output_clf = model(pre, post, current_iter=current_iter)
        else:
            output_loc, output_clf = model(pre, post)

        pred_loc = output_loc.argmax(dim=1).cpu().numpy()
        pred_clf = output_clf.argmax(dim=1).cpu().numpy()
        gt_loc = labels_loc_t.cpu().numpy()
        gt_clf = labels_clf_t.cpu().numpy()
        evaluator_loc.add_batch(gt_loc, pred_loc)
        evaluator_clf.add_batch(gt_clf[gt_loc > 0], pred_clf[gt_loc > 0])
        evaluator_total.add_batch(gt_clf, pred_clf)

    loc_f1 = float(evaluator_loc.Pixel_F1_score())
    clf_f1 = harmonic_damage_f1(evaluator_clf)
    oa = float(evaluator_total.Pixel_Accuracy())
    ious = np.asarray(evaluator_total.Intersection_over_Union(), dtype=np.float64)
    miou = float(evaluator_total.Mean_Intersection_over_Union())
    names = ["background", "intact", "damaged", "destroyed"]
    result = {
        "loc_f1": 100.0 * loc_f1,
        "clf_f1": 100.0 * clf_f1,
        "oa": 100.0 * oa,
        "miou": 100.0 * miou,
        "iou": {name: 100.0 * float(value) for name, value in zip(names, ious)},
    }
    print(
        f"{split_name}: loc_F1={result['loc_f1']:.4f}, clf_F1={result['clf_f1']:.4f}, "
        f"OA={result['oa']:.4f}, mIoU={result['miou']:.4f}",
        flush=True,
    )
    print("IoU: " + " | ".join(f"{name}={result['iou'][name]:.4f}" for name in names), flush=True)
    return result


def print_prototype_diagnostics(model: DamageFormer_BGMP) -> Dict[str, Any]:
    memory = model.damage_proto_memory
    proto = memory.prototypes.detach().cpu()
    norms = proto.norm(dim=-1)
    normalized = F.normalize(proto, dim=-1)
    class_mean = F.normalize(normalized.mean(dim=1), dim=-1)
    interclass = class_mean @ class_mean.t()
    usage = memory.prototype_usage.detach().cpu()
    usage_ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
    result = {
        "initialized": memory.prototype_initialized.detach().cpu().tolist(),
        "shape": list(proto.shape),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "usage_ratio": usage_ratio.tolist(),
        "interclass_cosine": interclass.tolist(),
        "gamma": float(memory.gamma().detach().cpu().item()),
    }
    print("Prototype diagnostics:", json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    state = load_state(str(checkpoint))
    variant = infer_variant(state, args.model_variant)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Model variant: {variant}", flush=True)

    device = torch.device("cuda:0")
    model = build_model(args, state, variant).to(device)
    val_names = read_list(args.val_data_list_path)
    test_names = read_list(args.test_data_list_path)
    output: Dict[str, Any] = {"checkpoint": str(checkpoint), "variant": variant}

    if variant == "s2":
        output["prototype"] = print_prototype_diagnostics(model)
        modes = [args.proto_eval_mode] if args.proto_eval_mode != "both" else ["inactive", "active"]
        for mode in modes:
            current_iter = 0 if mode == "inactive" else args.proto_warmup_iters
            print(f"\n===== BGMP prototype mode: {mode} =====", flush=True)
            output[mode] = {
                "validation": evaluate_split(
                    model, args.val_dataset_path, val_names, args.eval_batch_size, device,
                    args.num_workers, f"validation/{mode}", current_iter=current_iter, is_bgmp=True
                ),
                "testing": evaluate_split(
                    model, args.test_dataset_path, test_names, args.eval_batch_size, device,
                    args.num_workers, f"testing/{mode}", current_iter=current_iter, is_bgmp=True
                ),
            }
    else:
        output["validation"] = evaluate_split(
            model, args.val_dataset_path, val_names, args.eval_batch_size, device, args.num_workers, "validation"
        )
        output["testing"] = evaluate_split(
            model, args.test_dataset_path, test_names, args.eval_batch_size, device, args.num_workers, "testing"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {out_path}", flush=True)


if __name__ == "__main__":
    main()
