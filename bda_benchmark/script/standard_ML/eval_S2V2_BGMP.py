#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone evaluator for DamageFormer_BGMP / S2V2.

Reports:
1. Localization F1
2. Damage harmonic F1
3. Damage class F1
4. OA
5. Four-class IoU
6. Four-class mIoU
7. Damage confusion matrix
8. Damaged -> Intact / Destroyed
9. Destroyed -> Intact / Damaged
10. Prototype diagnostics
11. Optional active-vs-bypass prototype comparison

The model hyperparameters proto_k / proto_dim are inferred directly
from the checkpoint prototype buffer.
"""

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
from tqdm import tqdm


# -------------------------------------------------------------------------
# Repository paths
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"

for import_root in (PROJECT_ROOT, BDA_ROOT, SCRIPT_DIR):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)


from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BGMP import DamageFormer_BGMP
from util_func.metrics import Evaluator


CLASS_NAMES = [
    "Background",
    "Intact",
    "Damaged",
    "Destroyed",
]


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate S2V2 DamageFormer_BGMP checkpoint"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--val_data_list_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--test_data_list_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--split",
        default="both",
        choices=["val", "test", "both"],
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--roi_alpha",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--proto_tau",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--proto_warmup_iters",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--compare_proto_bypass",
        action="store_true",
        help=(
            "Also evaluate with current_iter=-1 so prototype retrieval "
            "is bypassed. This does not change checkpoint buffers."
        ),
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Checkpoint utilities
# -------------------------------------------------------------------------

def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dict or state_dict-like object."
        )

    for key in (
        "model",
        "state_dict",
        "model_state_dict",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    if not all(
        isinstance(k, str)
        for k in checkpoint.keys()
    ):
        raise TypeError(
            "Could not identify a valid model state_dict."
        )

    state: Dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():
        clean_key = (
            key[7:]
            if key.startswith("module.")
            else key
        )
        state[clean_key] = value

    return state


def find_prototype_key(
    state: Dict[str, torch.Tensor],
) -> str:
    exact = "damage_proto_memory.prototypes"

    if exact in state:
        return exact

    matches = [
        key
        for key in state
        if key.endswith(
            "damage_proto_memory.prototypes"
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely locate prototype buffer. "
            f"Matches: {matches}"
        )

    return matches[0]


# -------------------------------------------------------------------------
# Confusion matrix
# -------------------------------------------------------------------------

def update_hist(
    hist: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
) -> None:
    valid = (
        (gt != 255)
        & (gt >= 0)
        & (gt < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )

    gt_valid = gt[valid].astype(np.int64)
    pred_valid = pred[valid].astype(np.int64)

    encoded = (
        num_classes * gt_valid
        + pred_valid
    )

    hist += np.bincount(
        encoded,
        minlength=num_classes ** 2,
    ).reshape(
        num_classes,
        num_classes,
    )


def row_percent(hist: np.ndarray) -> np.ndarray:
    row_sum = hist.sum(
        axis=1,
        keepdims=True,
    )

    return (
        hist
        / np.maximum(row_sum, 1)
        * 100.0
    )


# -------------------------------------------------------------------------
# Prototype diagnostics
# -------------------------------------------------------------------------

@torch.no_grad()
def prototype_diagnostics(
    model: DamageFormer_BGMP,
) -> Dict[str, Any]:
    memory = model.damage_proto_memory

    proto = memory.prototypes.detach().float()
    proto_norm = proto.norm(dim=-1)

    normalized = F.normalize(
        proto,
        dim=-1,
    )

    class_center = F.normalize(
        normalized.mean(dim=1),
        dim=-1,
    )

    class_cosine = (
        class_center
        @ class_center.t()
    )

    usage = (
        memory.prototype_usage
        .detach()
        .float()
    )

    usage_ratio = (
        usage
        / usage.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
    )

    return {
        "prototype_shape": list(proto.shape),
        "initialized": (
            memory.prototype_initialized
            .detach()
            .cpu()
            .tolist()
        ),
        "gamma": float(
            memory.gamma().detach().cpu().item()
        ),
        "norm_min": float(
            proto_norm.min().cpu().item()
        ),
        "norm_max": float(
            proto_norm.max().cpu().item()
        ),
        "usage_ratio": (
            usage_ratio.cpu().tolist()
        ),
        "interclass_cosine": (
            class_cosine.cpu().tolist()
        ),
    }


def print_prototype_diagnostics(
    diagnostics: Dict[str, Any],
) -> None:
    print(
        "\n"
        "================ PROTOTYPE DIAGNOSTICS "
        "================"
    )

    print(
        "prototype shape :",
        diagnostics["prototype_shape"],
    )

    print(
        "initialized     :",
        diagnostics["initialized"],
    )

    print(
        "gamma           :",
        f"{diagnostics['gamma']:.6f}",
    )

    print(
        "norm range      :",
        f"[{diagnostics['norm_min']:.6f}, "
        f"{diagnostics['norm_max']:.6f}]",
    )

    print(
        "usage ratio     :",
        diagnostics["usage_ratio"],
    )

    cosine = np.asarray(
        diagnostics["interclass_cosine"],
        dtype=np.float64,
    )

    print("\nPrototype inter-class cosine:")
    print(
        "                 Intact      Damaged    Destroyed"
    )

    for idx, class_name in enumerate(
        ("Intact", "Damaged", "Destroyed")
    ):
        values = "  ".join(
            f"{value:10.6f}"
            for value in cosine[idx]
        )

        print(
            f"{class_name:<12s} {values}"
        )


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model: DamageFormer_BGMP,
    dataset_path: str,
    data_names: list[str],
    eval_batch_size: int,
    device: torch.device,
    num_workers: int,
    split_name: str,
    prototype_active: bool,
) -> Dict[str, Any]:

    mode_name = (
        "PROTO_ACTIVE"
        if prototype_active
        else "PROTO_BYPASS"
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        f"EVALUATING {split_name.upper()} | {mode_name}"
    )

    print(
        "============================================================"
    )

    evaluator_loc = Evaluator(
        num_class=2,
    )

    evaluator_clf = Evaluator(
        num_class=4,
    )

    evaluator_total = Evaluator(
        num_class=4,
    )

    damage_hist = np.zeros(
        (4, 4),
        dtype=np.int64,
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
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model.eval()

    proto_correct = 0
    proto_total = 0

    for data in tqdm(
        loader,
        desc=f"{split_name}-{mode_name}",
    ):
        (
            pre,
            post,
            labels_loc,
            labels_clf,
            _,
        ) = data

        pre = pre.to(
            device,
            non_blocking=True,
        )

        post = post.to(
            device,
            non_blocking=True,
        )

        labels_loc_t = labels_loc.to(
            device,
            non_blocking=True,
        ).long()

        labels_clf_t = labels_clf.to(
            device,
            non_blocking=True,
        ).long()

        current_iter = (
            None
            if prototype_active
            else -1
        )

        (
            output_loc,
            output_clf,
            aux,
        ) = model(
            pre,
            post,
            current_iter=current_iter,
            return_aux=True,
        )

        pred_loc = (
            output_loc
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        pred_clf = (
            output_clf
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        gt_loc = (
            labels_loc_t
            .cpu()
            .numpy()
        )

        gt_clf = (
            labels_clf_t
            .cpu()
            .numpy()
        )

        # Same metric organization as the training evaluator.
        evaluator_loc.add_batch(
            gt_loc,
            pred_loc,
        )

        evaluator_clf.add_batch(
            gt_clf[gt_loc > 0],
            pred_clf[gt_loc > 0],
        )

        evaluator_total.add_batch(
            gt_clf,
            pred_clf,
        )

        update_hist(
            damage_hist,
            gt_clf,
            pred_clf,
            num_classes=4,
        )

        proto_similarity = aux.get(
            "proto_class_similarity"
        )

        if proto_similarity is not None:
            feature = aux[
                "final_feat_fused_proto"
            ]

            feature_target = F.interpolate(
                labels_clf_t
                .unsqueeze(1)
                .float(),
                size=feature.shape[-2:],
                mode="nearest",
            )[:, 0].long().reshape(-1)

            valid_proto = (
                (feature_target >= 1)
                & (feature_target <= 3)
            )

            if bool(valid_proto.any().item()):
                proto_pred = (
                    proto_similarity.argmax(dim=1)
                    + 1
                )

                proto_correct += int(
                    (
                        proto_pred[valid_proto]
                        == feature_target[valid_proto]
                    )
                    .sum()
                    .item()
                )

                proto_total += int(
                    valid_proto.sum().item()
                )

    loc_f1 = float(
        evaluator_loc.Pixel_F1_score()
    )

    damage_f1 = np.asarray(
        evaluator_clf.Damage_F1_score(),
        dtype=np.float64,
    )

    if (
        len(damage_f1) > 0
        and np.all(damage_f1 > 0)
    ):
        harmonic_f1 = float(
            len(damage_f1)
            / np.sum(
                1.0 / damage_f1
            )
        )
    else:
        harmonic_f1 = 0.0

    oa = float(
        evaluator_total.Pixel_Accuracy()
    )

    ious = np.asarray(
        evaluator_total
        .Intersection_over_Union(),
        dtype=np.float64,
    )

    miou = float(
        evaluator_total
        .Mean_Intersection_over_Union()
    )

    damage_row_pct = row_percent(
        damage_hist,
    )

    damaged_total = max(
        int(damage_hist[2].sum()),
        1,
    )

    destroyed_total = max(
        int(damage_hist[3].sum()),
        1,
    )

    damaged_to_intact = float(
        damage_hist[2, 1]
        / damaged_total
    )

    damaged_to_destroyed = float(
        damage_hist[2, 3]
        / damaged_total
    )

    destroyed_to_intact = float(
        damage_hist[3, 1]
        / destroyed_total
    )

    destroyed_to_damaged = float(
        damage_hist[3, 2]
        / destroyed_total
    )

    proto_cls_acc = (
        float(proto_correct / proto_total)
        if proto_total > 0
        else float("nan")
    )

    result = {
        "split": split_name,
        "prototype_mode": mode_name,
        "localization_f1": loc_f1,
        "damage_harmonic_f1": harmonic_f1,
        "damage_class_f1": damage_f1.tolist(),
        "oa": oa,
        "miou": miou,
        "class_iou": ious.tolist(),
        "confusion_matrix": damage_hist.tolist(),
        "confusion_row_percent": damage_row_pct.tolist(),
        "damaged_to_intact": damaged_to_intact,
        "damaged_to_destroyed": damaged_to_destroyed,
        "destroyed_to_intact": destroyed_to_intact,
        "destroyed_to_damaged": destroyed_to_damaged,
        "prototype_class_accuracy": proto_cls_acc,
    }

    print_results(
        result,
    )

    return result


def print_results(
    result: Dict[str, Any],
) -> None:

    print("\n---------------- SUMMARY ----------------")

    print(
        f"Localization F1       : "
        f"{100 * result['localization_f1']:.4f}"
    )

    print(
        f"Damage Harmonic F1    : "
        f"{100 * result['damage_harmonic_f1']:.4f}"
    )

    damage_f1 = result[
        "damage_class_f1"
    ]

    if len(damage_f1) >= 3:
        print(
            f"Damage F1 [I/D/Dest] : "
            f"{100 * damage_f1[0]:.4f} / "
            f"{100 * damage_f1[1]:.4f} / "
            f"{100 * damage_f1[2]:.4f}"
        )

    print(
        f"OA                    : "
        f"{100 * result['oa']:.4f}"
    )

    print(
        f"mIoU                  : "
        f"{100 * result['miou']:.4f}"
    )

    print("\nClass IoU:")

    for class_name, value in zip(
        CLASS_NAMES,
        result["class_iou"],
    ):
        print(
            f"  {class_name:<10s}: "
            f"{100 * value:.4f}"
        )

    print("\nKey damage errors:")

    print(
        f"  Damaged   -> Intact    : "
        f"{100 * result['damaged_to_intact']:.4f}%"
    )

    print(
        f"  Damaged   -> Destroyed : "
        f"{100 * result['damaged_to_destroyed']:.4f}%"
    )

    print(
        f"  Destroyed -> Intact    : "
        f"{100 * result['destroyed_to_intact']:.4f}%"
    )

    print(
        f"  Destroyed -> Damaged   : "
        f"{100 * result['destroyed_to_damaged']:.4f}%"
    )

    proto_acc = result[
        "prototype_class_accuracy"
    ]

    if np.isfinite(proto_acc):
        print(
            f"\nPrototype class accuracy: "
            f"{100 * proto_acc:.4f}%"
        )

    print(
        "\nDamage confusion row percentages "
        "(GT rows -> predicted columns):"
    )

    matrix = np.asarray(
        result["confusion_row_percent"],
        dtype=np.float64,
    )

    print(
        "GT \\ Pred         BG       Intact      "
        "Damaged    Destroyed"
    )

    for idx, class_name in enumerate(
        CLASS_NAMES
    ):
        values = "  ".join(
            f"{value:10.4f}"
            for value in matrix[idx]
        )

        print(
            f"{class_name:<12s} {values}"
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but "
                "torch.cuda.is_available() is False"
            )

    device = torch.device(
        args.device
    )

    print(
        "Loading checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = extract_state_dict(
        checkpoint,
    )

    proto_key = find_prototype_key(
        state,
    )

    proto_tensor = state[
        proto_key
    ]

    if proto_tensor.ndim != 3:
        raise RuntimeError(
            f"Expected prototype shape [3,K,D], "
            f"got {tuple(proto_tensor.shape)}"
        )

    num_classes, proto_k, proto_dim = (
        proto_tensor.shape
    )

    if int(num_classes) != 3:
        raise RuntimeError(
            "Expected 3 prototype damage classes, "
            f"got {num_classes}"
        )

    print("\nCheckpoint architecture inference:")
    print(
        "  prototype key :",
        proto_key,
    )
    print(
        "  prototype bank:",
        tuple(proto_tensor.shape),
    )
    print(
        "  proto_k       :",
        int(proto_k),
    )
    print(
        "  proto_dim     :",
        int(proto_dim),
    )

    feature_proj_key = (
        "damage_proto_memory.feature_proj.weight"
    )

    if feature_proj_key in state:
        print(
            "  feature_proj  :",
            tuple(
                state[
                    feature_proj_key
                ].shape
            ),
        )
    else:
        print(
            "  feature_proj  : "
            "NOT FOUND IN CHECKPOINT"
        )

    model = DamageFormer_BGMP(
        roi_alpha=args.roi_alpha,
        use_pretrained_swin=False,
        proto_k=int(proto_k),
        proto_dim=int(proto_dim),
        proto_momentum=0.99,
        proto_tau=args.proto_tau,
        proto_warmup_iters=(
            args.proto_warmup_iters
        ),
    ).to(device)

    # -----------------------------------------------------------------
    # S2V2 direct256 checkpoint compatibility
    # -----------------------------------------------------------------
    # direct256 checkpoints do not contain feature_proj.weight because
    # they directly use the 256-channel damage feature as prototype feature.
    if feature_proj_key not in state:
        if int(proto_dim) != 256:
            raise RuntimeError(
                "Checkpoint has no feature_proj.weight, but inferred "
                f"proto_dim={int(proto_dim)}. Identity projection is only "
                "safe for direct256 checkpoints."
            )
        print(
            "\nS2V2 direct256 compatibility: "
            "replace damage_proto_memory.feature_proj with nn.Identity()"
        )
        model.damage_proto_memory.feature_proj = torch.nn.Identity()

    # Some S2V2 checkpoints store an additional buffer used during training.
    # Register it here so strict=True loading remains valid.
    empty_reinit_key = "damage_proto_memory.empty_reinit_count"
    if empty_reinit_key in state and not hasattr(
        model.damage_proto_memory,
        "empty_reinit_count",
    ):
        print(
            "S2V2 compatibility: register empty_reinit_count buffer",
            tuple(state[empty_reinit_key].shape),
        )
        model.damage_proto_memory.register_buffer(
            "empty_reinit_count",
            torch.zeros_like(
                state[empty_reinit_key],
                device=device,
            ),
        )

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except RuntimeError as exc:
        print(
            "\n"
            "================ STRICT LOAD FAILED ================"
        )

        print(exc)

        print(
            "\nThis usually means the current "
            "model/DamageFormer_BGMP.py is not the exact "
            "architecture used to train this S2V2 checkpoint."
        )

        print(
            "In particular, check whether the direct256 "
            "experiment removed/replaced feature_proj."
        )

        raise

    print(
        "\nSTRICT CHECKPOINT LOAD: PASS"
    )

    diagnostics = prototype_diagnostics(
        model,
    )

    print_prototype_diagnostics(
        diagnostics,
    )

    val_names = []

    with open(
        args.val_data_list_path,
        "r",
        encoding="utf-8",
    ) as file:
        val_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    test_names = []

    with open(
        args.test_data_list_path,
        "r",
        encoding="utf-8",
    ) as file:
        test_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print(
        "\nValidation samples:",
        len(val_names),
    )

    print(
        "Test samples      :",
        len(test_names),
    )

    results: Dict[str, Any] = {
        "checkpoint": str(
            checkpoint_path
        ),
        "inferred_proto_k": int(
            proto_k
        ),
        "inferred_proto_dim": int(
            proto_dim
        ),
        "prototype_diagnostics": diagnostics,
        "results": {},
    }

    if args.split in (
        "val",
        "both",
    ):
        results["results"][
            "val_active"
        ] = evaluate_split(
            model=model,
            dataset_path=args.dataset_path,
            data_names=val_names,
            eval_batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
            split_name="validation",
            prototype_active=True,
        )

        if args.compare_proto_bypass:
            results["results"][
                "val_bypass"
            ] = evaluate_split(
                model=model,
                dataset_path=args.dataset_path,
                data_names=val_names,
                eval_batch_size=args.eval_batch_size,
                device=device,
                num_workers=args.num_workers,
                split_name="validation",
                prototype_active=False,
            )

    if args.split in (
        "test",
        "both",
    ):
        results["results"][
            "test_active"
        ] = evaluate_split(
            model=model,
            dataset_path=args.dataset_path,
            data_names=test_names,
            eval_batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
            split_name="testing",
            prototype_active=True,
        )

        if args.compare_proto_bypass:
            results["results"][
                "test_bypass"
            ] = evaluate_split(
                model=model,
                dataset_path=args.dataset_path,
                data_names=test_names,
                eval_batch_size=args.eval_batch_size,
                device=device,
                num_workers=args.num_workers,
                split_name="testing",
                prototype_active=False,
            )

    if args.compare_proto_bypass:
        print(
            "\n"
            "================ ACTIVE vs BYPASS ================"
        )

        for split_prefix in (
            "val",
            "test",
        ):
            active_key = (
                f"{split_prefix}_active"
            )

            bypass_key = (
                f"{split_prefix}_bypass"
            )

            if (
                active_key
                not in results["results"]
                or bypass_key
                not in results["results"]
            ):
                continue

            active = results[
                "results"
            ][active_key]

            bypass = results[
                "results"
            ][bypass_key]

            print(
                f"\n{split_prefix.upper()}:"
            )

            print(
                "  Delta mIoU active-bypass      : "
                f"{100 * (active['miou'] - bypass['miou']):+.4f}"
            )

            delta_damage_f1 = 100.0 * (
                active["damage_harmonic_f1"]
                - bypass["damage_harmonic_f1"]
            )

            print(
                "  Delta Damage F1 active-bypass : "
                f"{delta_damage_f1:+.4f}"
            )

            for idx, class_name in enumerate(
                CLASS_NAMES
            ):
                delta = (
                    active["class_iou"][idx]
                    - bypass["class_iou"][idx]
                )

                print(
                    f"  Delta {class_name:<10s} IoU    : "
                    f"{100 * delta:+.4f}"
                )

    output_path = (
        Path(args.output_json)
        if args.output_json
        else (
            checkpoint_path.parent
            / "eval_s2v2_full.json"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nEvaluation JSON saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
