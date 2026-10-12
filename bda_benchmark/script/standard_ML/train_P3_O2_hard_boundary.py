#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-O2: Train-only hard-region Occurrence boundary screen.

Unlike P3-O1, this final diagnostic does not supervise every easy building
pixel.  A frozen P3 Train cache only identifies two exact region pools:

* GT Damaged regions that frozen P3 predicts Intact;
* GT Intact regions that frozen P3 predicts Intact but assigns the highest
  Affected probability within each event.

No cached feature, neighbour, similarity, Memory label, validation target, or
Test sample enters training.  Inference remains the ordinary P3 forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.make_data_loader_p4_native_sar import (
    P4NativeSARDataset,
    build_p4_dataset,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache
from script.standard_ML.p3_direct_occurrence_runtime import (
    o1_validation_checks,
    region_occurrence_margin,
    select_hard_region_indices,
)
from script.standard_ML.p3_hard_occurrence_runtime import (
    match_cached_component,
    protected_spatial_kl,
)
from script.standard_ML.train_P3_O1_direct_occurrence import (
    O1_TRAINABLE_PREFIXES,
    atomic_save,
    configure_o1_trainable,
    o1_model_delta,
    set_o1_train_mode,
    sha256,
)
from script.standard_ML.train_S8_P4_native_sar_micb import (
    evaluate,
    format_metrics,
    parse_p4_batch,
)
from train_S5_S8_GPU import segmentation_loss, seed_worker, set_random_seed


class O2CachedRegionDataset(Dataset):
    """Full Train images with an exact frozen-P3 component identity."""

    def __init__(
        self,
        dataset_path: str,
        cache: Mapping[str, object],
        region_indices: torch.Tensor,
        crop_size: int,
    ) -> None:
        self.cache = cache
        self.region_indices = region_indices.long().flatten().cpu()
        self.region_image_ids = [
            str(cache["image_ids"][int(index)]) for index in self.region_indices
        ]
        unique_images = sorted(set(self.region_image_ids))
        self.image_to_index = {
            image_id: index for index, image_id in enumerate(unique_images)
        }
        self.base = P4NativeSARDataset(
            dataset_path=dataset_path,
            data_list=unique_images,
            crop_size=crop_size,
            split="cache",
            variant="a0",
        )

    def __len__(self) -> int:
        return int(self.region_indices.numel())

    def __getitem__(self, row: int):
        row = int(row)
        region_index = int(self.region_indices[row])
        image_id = self.region_image_ids[row]
        sample = dict(self.base[self.image_to_index[image_id]])
        sample.update(
            {
                "region_index": region_index,
                "expected_bbox": self.cache["bboxes"][region_index].clone(),
                "expected_area": int(self.cache["areas"][region_index]),
            }
        )
        return sample


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def exact_region_margin(
    student: DamageFormer_S8_P4_NativeSAR_MICB,
    teacher: DamageFormer_S8_P4_NativeSAR_MICB,
    batch: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Replay one cached region and return its differentiable O2 margin."""

    pre = batch["pre_img"].float().to(device)
    post = batch["teacher_post_img"].float().to(device)
    if pre.shape[0] != 1:
        raise RuntimeError("P3-O2 exact replay requires hard_batch_size=1")
    with torch.no_grad():
        teacher_location, _ = teacher(pre, post)
    _, student_damage = student(pre, post)
    building_mask = (
        teacher_location.argmax(dim=1)[0].detach().cpu().numpy() == 1
    )
    component = match_cached_component(
        building_mask,
        batch["expected_bbox"][0].tolist(),
        int(batch["expected_area"][0]),
    )
    if component is None:
        raise RuntimeError(
            "P3-O2 frozen-Teacher component replay failed for "
            f"{batch['name'][0]} | "
            f"bbox={batch['expected_bbox'][0].tolist()} | "
            f"area={int(batch['expected_area'][0])}"
        )
    return (
        region_occurrence_margin(student_damage, component, batch_index=0),
        int(component.sum()),
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "protocol": "P3_O2_hard_boundary_delta_checkpoint_v1",
        "base_p3_checkpoint": str(args.p3_checkpoint),
        "base_p3_checkpoint_sha256": sha256(args.p3_checkpoint),
        "model_delta": o1_model_delta(model),
        "model_delta_prefixes": list(O1_TRAINABLE_PREFIXES),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "metrics": dict(metrics),
        "config": vars(args),
        "inference_uses_memory": False,
        "uses_cached_features": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser("P3-O2 hard-region boundary screen")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--optimizer_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--hard_weight", type=float, default=0.05)
    parser.add_argument("--protect_weight", type=float, default=0.5)
    parser.add_argument("--required_margin", type=float, default=0.0)
    parser.add_argument("--cap_per_event", type=int, default=64)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--expected_valid_miou", type=float, default=60.5019)
    parser.add_argument("--baseline_tolerance_pp", type=float, default=0.10)
    parser.add_argument("--minimum_damaged_gain_pp", type=float, default=0.30)
    parser.add_argument("--maximum_guard_drop_pp", type=float, default=0.20)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    args = parser.parse_args()
    if args.optimizer_steps != 500 or args.eval_every != 250:
        raise ValueError("P3-O2 is locked to 500 steps with eval_every=250")
    if args.hard_weight <= 0 or args.protect_weight < 0:
        raise ValueError("invalid O2 loss weights")
    if args.required_margin != 0.0:
        raise ValueError("first P3-O2 screen locks required_margin to zero")
    if args.cap_per_event != 64:
        raise ValueError("first P3-O2 screen locks cap_per_event to 64")

    _, labels, events, _, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    metadata = cache["metadata"]
    if metadata.get("source_split") != "train" or bool(
        metadata.get("test_accessed", True)
    ):
        raise RuntimeError("P3-O2 accepts only a Test-free Train cache")
    if "base_predictions" not in cache or "base_probabilities" not in cache:
        raise RuntimeError("P3-O2 cache lacks frozen-P3 outputs")
    checkpoint_hash = sha256(args.p3_checkpoint)
    if checkpoint_hash != metadata.get("p3_checkpoint_sha256"):
        raise RuntimeError("P3-O2 checkpoint/cache identity mismatch")
    positive_indices, negative_indices, hard_audit = select_hard_region_indices(
        labels,
        cache["base_predictions"],
        cache["base_probabilities"],
        events,
        cap_per_event=args.cap_per_event,
        seed=args.seed,
    )
    if hard_audit["positive_events"] < 10 or hard_audit["negative_events"] < 10:
        raise RuntimeError("P3-O2 hard pools do not cover at least ten events")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = set_random_seed(args.seed, 0, False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    student = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    teacher = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(student, args.p3_checkpoint)
    load_p3_checkpoint(teacher, args.p3_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    parameter_counts = configure_o1_trainable(student)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_set = build_p4_dataset(
        args.train_dataset_path,
        args.train_data_list_path,
        args.crop_size,
        "train",
        "a0",
        "",
    )
    val_set = build_p4_dataset(
        args.val_dataset_path,
        args.val_data_list_path,
        args.crop_size,
        "val",
        "a0",
        "",
    )
    positive_set = O2CachedRegionDataset(
        args.train_dataset_path, cache, positive_indices, args.crop_size
    )
    negative_set = O2CachedRegionDataset(
        args.train_dataset_path, cache, negative_indices, args.crop_size
    )
    common_loader = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        **{**common_loader, "drop_last": True, "persistent_workers": args.num_workers > 0},
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **{**common_loader, "persistent_workers": args.num_workers > 0},
    )
    hard_generator = torch.Generator().manual_seed(args.seed + 991)
    positive_loader = DataLoader(
        positive_set,
        batch_size=1,
        shuffle=True,
        generator=hard_generator,
        **common_loader,
    )
    negative_loader = DataLoader(
        negative_set,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1991),
        **common_loader,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "hard_region_audit.json").write_text(
        json.dumps(hard_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline = evaluate(student, val_loader, device, args.ignore_index, 0)
    print("Initial validation | " + format_metrics(baseline), flush=True)
    if (
        abs(100.0 * baseline["valid_miou"] - args.expected_valid_miou)
        > args.baseline_tolerance_pp
    ):
        raise RuntimeError("P3-O2 Step-0 does not reproduce accepted P3")
    baseline_checks, baseline_diagnostics = o1_validation_checks(
        baseline,
        baseline,
        minimum_damaged_gain_pp=args.minimum_damaged_gain_pp,
        maximum_guard_drop_pp=args.maximum_guard_drop_pp,
        minimum_nonnegative_events=args.minimum_nonnegative_events,
    )
    history = [
        {
            "step": 0,
            "metrics": baseline,
            "checks": baseline_checks,
            "diagnostics": baseline_diagnostics,
            "passed": False,
        }
    ]
    (output_dir / "safe_step0.json").write_text(
        json.dumps(
            {
                "protocol": "P3_O2_safe_step0_reference_v1",
                "base_p3_checkpoint": str(args.p3_checkpoint),
                "base_p3_checkpoint_sha256": checkpoint_hash,
                "metrics": baseline,
                "inference_uses_memory": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    train_iterator = iter(train_loader)
    positive_iterator = iter(positive_loader)
    negative_iterator = iter(negative_loader)
    best_admitted = None
    start_time = time.perf_counter()
    exact_replays = 0

    for step in range(1, args.optimizer_steps + 1):
        natural_batch, train_iterator = next_or_restart(
            train_iterator, train_loader
        )
        positive_batch, positive_iterator = next_or_restart(
            positive_iterator, positive_loader
        )
        negative_batch, negative_iterator = next_or_restart(
            negative_iterator, negative_loader
        )
        set_o1_train_mode(student)
        optimizer.zero_grad(set_to_none=True)

        pre, sar, teacher_sar, _, damage, _ = parse_p4_batch(
            natural_batch, device
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _, teacher_damage, _ = teacher(
                pre, teacher_sar, return_aux=True
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _, student_damage, _ = student(pre, sar, return_aux=True)
            damage_loss, _, _ = segmentation_loss(
                student_damage,
                damage,
                ce,
                True,
                1.0,
                1.0,
                args.ignore_index,
            )
            protect_loss = protected_spatial_kl(
                student_damage,
                teacher_damage,
                damage,
                temperature=args.kd_temperature,
                protected_classes=(1, 3),
                ignore_index=args.ignore_index,
            )
            natural_loss = damage_loss + args.protect_weight * protect_loss
        natural_loss.backward()

        positive_margin, positive_area = exact_region_margin(
            student, teacher, positive_batch, device
        )
        positive_loss = F.softplus(args.required_margin - positive_margin)
        (0.5 * args.hard_weight * positive_loss).backward()
        positive_margin_value = float(positive_margin.detach())
        del positive_margin

        negative_margin, negative_area = exact_region_margin(
            student, teacher, negative_batch, device
        )
        negative_loss = F.softplus(args.required_margin + negative_margin)
        (0.5 * args.hard_weight * negative_loss).backward()
        negative_margin_value = float(negative_margin.detach())
        del negative_margin
        exact_replays += 2

        grad_norms = {
            prefix: sum(
                float(parameter.grad.detach().abs().sum())
                for name, parameter in student.named_parameters()
                if name.startswith(prefix + ".") and parameter.grad is not None
            )
            for prefix in ("fusion_layer_3", "clf_2")
        }
        if step == 1 and any(value <= 0.0 for value in grad_norms.values()):
            raise RuntimeError(f"P3-O2 gradient contract failed: {grad_norms}")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            5.0,
        )
        optimizer.step()

        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "damage_loss": float(damage_loss.detach()),
                        "protect_loss": float(protect_loss.detach()),
                        "hard_boundary_loss": 0.5
                        * (
                            float(positive_loss.detach())
                            + float(negative_loss.detach())
                        ),
                        "damaged_margin": positive_margin_value,
                        "intact_margin": negative_margin_value,
                        "damaged_area": positive_area,
                        "intact_area": negative_area,
                        "exact_region_replays": exact_replays,
                        "grad_norms": grad_norms,
                        "steps_per_second": step
                        / max(time.perf_counter() - start_time, 1e-6),
                    }
                ),
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics = evaluate(student, val_loader, device, args.ignore_index, 0)
            checks, diagnostics = o1_validation_checks(
                metrics,
                baseline,
                minimum_damaged_gain_pp=args.minimum_damaged_gain_pp,
                maximum_guard_drop_pp=args.maximum_guard_drop_pp,
                minimum_nonnegative_events=args.minimum_nonnegative_events,
            )
            result = {
                "step": step,
                "metrics": metrics,
                "checks": checks,
                "diagnostics": diagnostics,
                "passed": all(checks.values()),
            }
            history.append(result)
            print("Validation | " + format_metrics(metrics), flush=True)
            print(f"P3-O2 checks: {checks}", flush=True)
            atomic_save(
                checkpoint_payload(
                    student,
                    optimizer,
                    step=step,
                    metrics=metrics,
                    args=args,
                ),
                output_dir / "latest_resume.pth",
            )
            if result["passed"] and (
                best_admitted is None
                or diagnostics["damaged_gain_pp"]
                > best_admitted["diagnostics"]["damaged_gain_pp"]
            ):
                best_admitted = result
                atomic_save(
                    checkpoint_payload(
                        student,
                        optimizer,
                        step=step,
                        metrics=metrics,
                        args=args,
                    ),
                    output_dir / "best_admitted.pth",
                )

    decision = (
        "ADMIT_P3_O2_TO_1500_STEP_EXTENSION_NO_TEST"
        if best_admitted is not None
        else "STOP_P3_O2_HARD_BOUNDARY_SCREEN_DO_NOT_TEST"
    )
    summary = {
        "protocol": "P3_O2_hard_boundary_screen_summary_v1",
        "baseline": baseline,
        "hard_region_audit": hard_audit,
        "history": history,
        "best_admitted": best_admitted,
        "selected_checkpoint": (
            str(output_dir / "best_admitted.pth")
            if best_admitted
            else str(args.p3_checkpoint)
        ),
        "exact_region_replays": exact_replays,
        "parameter_counts": parameter_counts,
        "decision": decision,
        "inference_uses_memory": False,
        "uses_cached_features": False,
        "test_accessed": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Decision: {decision}")
    print(f"Selected: {summary['selected_checkpoint']}")
    print(f"Saved: {output_dir}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
