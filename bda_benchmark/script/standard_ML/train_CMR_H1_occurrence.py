#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMR-H1A: 500-step Train-only hard occurrence supervision on frozen P3.

Memory is used only to choose Train pairs.  Validation calls the ordinary P3
forward and never loads a Memory item or a retrieval module.
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
from torch.utils.data import DataLoader, Dataset

from dataset.make_data_loader_p4_native_sar import (
    P4NativeSARDataset,
    build_p4_dataset,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
    lock_frozen_batch_norm,
)
from model.cmr.occurrence import CMRTrainingOnlyOccurrenceProjector
from script.standard_ML.cmr_event_capped_sampler import build_event_capped_epoch
from script.standard_ML.cmr_metric_utils import load_cmr_cache
from script.standard_ML.p3_hard_occurrence_runtime import (
    differentiable_region_mean_key,
    h1_validation_checks,
    match_cached_component,
    occurrence_info_nce,
    protected_spatial_kl,
)
from script.standard_ML.train_S8_P4_native_sar_micb import (
    evaluate,
    format_metrics,
    parse_p4_batch,
)
from train_S5_S8_GPU import segmentation_loss, seed_worker, set_random_seed


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    except Exception:
        # A full disk, quota failure, or interrupted network filesystem write
        # must not leave a large corrupt temporary checkpoint behind.
        temporary.unlink(missing_ok=True)
        raise


H1_TRAINABLE_PREFIXES = ("fusion_layer_3.", "clf_2.")


def h1_model_delta(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only H1's inference-time trainable tensors.

    The accepted P3 is the immutable base checkpoint.  Re-saving its frozen
    encoders at every H1 validation wastes hundreds of megabytes and obscures
    the actual experimental change.
    """

    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(H1_TRAINABLE_PREFIXES)
    }


def load_h1_model_delta(
    model: nn.Module,
    delta: Mapping[str, torch.Tensor],
) -> None:
    expected = {
        name
        for name in model.state_dict()
        if name.startswith(H1_TRAINABLE_PREFIXES)
    }
    actual = set(delta)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"H1 model-delta contract mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(dict(delta), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected H1 delta keys: {incompatible.unexpected_keys}"
        )


class H1HardAnchorDataset(Dataset):
    """No-augmentation full images indexed by H0 pair-row position."""

    def __init__(
        self,
        dataset_path: str,
        cache: Mapping[str, object],
        pairs: Mapping[str, object],
        crop_size: int,
    ) -> None:
        self.cache = cache
        self.anchor_indices = pairs["anchor_indices"].long().flatten().cpu()
        self.anchor_image_ids = list(map(str, pairs["anchor_image_ids"]))
        if len(self.anchor_image_ids) != self.anchor_indices.numel():
            raise RuntimeError("H0 anchor image IDs do not align")
        unique_images = sorted(set(self.anchor_image_ids))
        self.image_to_index = {name: index for index, name in enumerate(unique_images)}
        self.base = P4NativeSARDataset(
            dataset_path=dataset_path,
            data_list=unique_images,
            crop_size=crop_size,
            split="cache",
            variant="a0",
        )

    def __len__(self) -> int:
        return int(self.anchor_indices.numel())

    def __getitem__(self, pair_row: int):
        pair_row = int(pair_row)
        train_index = int(self.anchor_indices[pair_row])
        image_id = self.anchor_image_ids[pair_row]
        sample = dict(self.base[self.image_to_index[image_id]])
        sample.update({
            "pair_row": pair_row,
            "train_index": train_index,
            "expected_bbox": self.cache["bboxes"][train_index].clone(),
            "expected_area": int(self.cache["areas"][train_index]),
        })
        return sample


def configure_h1_trainable(
    model: DamageFormer_S8_P4_NativeSAR_MICB,
) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(("fusion_layer_3.", "clf_2.")):
            parameter.requires_grad_(True)
    lock_frozen_batch_norm(model)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def set_h1_train_mode(model: DamageFormer_S8_P4_NativeSAR_MICB) -> None:
    model.eval()
    model.fusion_layer_3.train()
    model.clf_2.train()
    lock_frozen_batch_norm(model)


def hard_loader_for_epoch(
    dataset: Dataset,
    anchor_events: torch.Tensor,
    *,
    epoch: int,
    cap_per_event: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    rows = build_event_capped_epoch(
        anchor_events,
        cap_per_event=cap_per_event,
        seed=seed,
        epoch=epoch,
    )
    # ``rows`` is already deterministically shuffled and contains no repeats.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=rows.tolist(),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )


def checkpoint_payload(
    model: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "protocol": "CMR_H1A_train_only_occurrence_delta_checkpoint_v2",
        "base_p3_checkpoint": str(args.p3_checkpoint),
        "base_p3_checkpoint_sha256": sha256(args.p3_checkpoint),
        "model_delta": h1_model_delta(model),
        "model_delta_prefixes": list(H1_TRAINABLE_PREFIXES),
        "occurrence_projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "metrics": dict(metrics),
        "config": vars(args),
        "inference_uses_memory": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser("CMR-H1A Train-only occurrence screen")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--h0_pairs", required=True)
    parser.add_argument("--h0b_manifest", required=True)
    parser.add_argument("--h1s0_audit", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--hard_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--optimizer_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--core_learning_rate", type=float, default=1e-6)
    parser.add_argument("--projector_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--hard_weight", type=float, default=0.05)
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--negative_count", type=int, default=4)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--lovasz_weight", type=float, default=1.0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--expected_valid_miou", type=float, default=60.5019)
    parser.add_argument("--baseline_tolerance_pp", type=float, default=0.10)
    parser.add_argument("--minimum_damaged_gain_pp", type=float, default=0.30)
    parser.add_argument("--maximum_guard_drop_pp", type=float, default=0.20)
    parser.add_argument("--minimum_nonnegative_events", type=int, default=10)
    args = parser.parse_args()
    if args.optimizer_steps != 500:
        raise ValueError("H1A is a locked 500-step screen")
    if args.eval_every != 250:
        raise ValueError("H1A eval_every is locked to 250")
    if args.negative_count <= 0:
        raise ValueError("negative_count must be positive")

    replay = json.load(open(args.h1s0_audit, encoding="utf-8"))
    if replay.get("protocol") != "CMR_H1S0_frozen_p3_feature_replay_guard_v1" or not bool(replay.get("passed", False)):
        raise RuntimeError("H1A requires a passed H1S0 replay guard")
    if bool(replay.get("test_accessed", True)):
        raise RuntimeError("H1A refuses a Test-accessed replay audit")

    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    pairs = torch.load(args.h0_pairs, map_location="cpu", weights_only=False)
    manifest = torch.load(args.h0b_manifest, map_location="cpu", weights_only=False)
    if pairs.get("protocol") != "CMR_H0_train_only_occurrence_hard_pairs_v1":
        raise RuntimeError("H1A refuses an unknown H0 pair protocol")
    if pairs.get("source_split") != "train":
        raise RuntimeError("H1A requires Train-only hard pairs")
    if bool(pairs.get("validation_accessed", True)) or bool(
        pairs.get("test_accessed", True)
    ):
        raise RuntimeError("H1A refuses Validation/Test-derived hard pairs")
    if not bool(manifest.get("passed", False)) or manifest.get("protocol") != "CMR_H0B_event_capped_rotating_train_manifest_v1":
        raise RuntimeError("H1A requires a passed H0B manifest")
    if manifest.get("h0_pairs_sha256") != sha256(args.h0_pairs):
        raise RuntimeError("H0 pairs changed after H0B audit")
    required_replay_count = int(
        manifest["preview_epoch_rows"].long().numel()
    )
    if int(replay.get("requested_anchors", 0)) < required_replay_count:
        raise RuntimeError(
            "H1A requires replay of the complete first capped epoch: "
            f"required={required_replay_count}, "
            f"audited={replay.get('requested_anchors', 0)}"
        )
    if int(replay.get("component_matches", 0)) < required_replay_count:
        raise RuntimeError(
            "H1A complete replay contains unmatched components: "
            f"required={required_replay_count}, "
            f"matched={replay.get('component_matches', 0)}"
        )
    checkpoint_sha = sha256(args.p3_checkpoint)
    if checkpoint_sha != cache["metadata"].get("p3_checkpoint_sha256") or checkpoint_sha != manifest.get("p3_checkpoint_sha256"):
        raise RuntimeError("P3 checkpoint identity mismatch")
    anchor_indices = pairs["anchor_indices"].long().flatten()
    positive_indices = pairs["positive_indices"].long()
    negative_indices = pairs["negative_indices"].long()
    anchor_events = pairs["anchor_events"].long().flatten()
    pair_contract = (
        bool((labels[anchor_indices] == 1).all())
        and bool((labels[positive_indices] == 1).all())
        and bool((labels[negative_indices] == 0).all())
        and not bool(
            (events[positive_indices] == events[anchor_indices, None]).any()
        )
        and not bool(
            (events[negative_indices] == events[anchor_indices, None]).any()
        )
    )
    if not pair_contract:
        raise RuntimeError("H1A hard-pair label/event contract failed")
    if not torch.equal(anchor_events.cpu(), events[anchor_indices].cpu()):
        raise RuntimeError("H1A anchor event codes do not match the Train cache")
    if int(args.negative_count) > negative_indices.shape[1]:
        raise ValueError("negative_count exceeds fixed H0 negative candidates")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # P0 and H1-S0 are produced with the default cuDNN inference policy
    # (benchmark disabled).  ``set_random_seed(..., deterministic=False)``
    # enables cuDNN benchmarking in the shared training helper; on recent
    # GPUs that can select a different B=1 convolution algorithm after the
    # preceding B=2 natural batch.  The resulting round-off is enough to flip
    # a localization boundary pixel and invalidate an otherwise exact cached
    # component.  Keep the RNG setup, then restore the backend policy used by
    # the cache and replay guard.
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
    parameter_counts = configure_h1_trainable(student)
    projector = CMRTrainingOnlyOccurrenceProjector(feature.shape[1], 128).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in student.parameters() if p.requires_grad],
                "lr": args.core_learning_rate,
                "name": "fusion3_clf2",
            },
            {
                "params": projector.parameters(),
                "lr": args.projector_learning_rate,
                "name": "training_only_occurrence_projector",
            },
        ],
        weight_decay=args.weight_decay,
    )

    train_set = build_p4_dataset(
        args.train_dataset_path, args.train_data_list_path, args.crop_size, "train", "a0", ""
    )
    val_set = build_p4_dataset(
        args.val_dataset_path, args.val_data_list_path, args.crop_size, "val", "a0", ""
    )
    hard_set = H1HardAnchorDataset(
        args.train_dataset_path, cache, pairs, args.crop_size
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
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
    train_iterator = iter(train_loader)
    hard_epoch = 0
    hard_loader = hard_loader_for_epoch(
        hard_set,
        anchor_events,
        epoch=hard_epoch,
        cap_per_event=int(manifest["cap_per_event"]),
        seed=int(manifest["seed"]),
        batch_size=args.hard_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    hard_iterator = iter(hard_loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline = evaluate(student, val_loader, device, args.ignore_index, 0)
    print("Initial validation | " + format_metrics(baseline), flush=True)
    if abs(100.0 * baseline["valid_miou"] - args.expected_valid_miou) > args.baseline_tolerance_pp:
        raise RuntimeError("H1A Step-0 does not reproduce accepted P3")
    baseline_checks, baseline_diagnostics = h1_validation_checks(
        baseline,
        baseline,
        minimum_damaged_gain_pp=args.minimum_damaged_gain_pp,
        maximum_guard_drop_pp=args.maximum_guard_drop_pp,
        minimum_nonnegative_events=args.minimum_nonnegative_events,
    )
    initial_result = {
        "step": 0,
        "metrics": baseline,
        "checks": baseline_checks,
        "diagnostics": baseline_diagnostics,
        "passed": False,
    }
    # Step 0 is byte-for-byte the accepted P3 base.  Record a lightweight
    # manifest instead of duplicating the entire frozen checkpoint.
    safe_step0 = {
        "protocol": "CMR_H1A_safe_step0_reference_v2",
        "step": 0,
        "base_p3_checkpoint": str(args.p3_checkpoint),
        "base_p3_checkpoint_sha256": checkpoint_sha,
        "metrics": baseline,
        "inference_uses_memory": False,
        "test_accessed": False,
    }
    (output_dir / "safe_step0.json").write_text(
        json.dumps(safe_step0, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history = [initial_result]
    best_admitted = None
    ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16
    support_feature = feature.float().cpu()
    pair_generator = torch.Generator().manual_seed(args.seed + 65537)
    start_time = time.perf_counter()
    hard_location_disagreement_pixels = 0
    hard_location_replay_samples = 0

    for step in range(1, args.optimizer_steps + 1):
        try:
            natural_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            natural_batch = next(train_iterator)
        try:
            hard_batch = next(hard_iterator)
        except StopIteration:
            hard_epoch += 1
            hard_loader = hard_loader_for_epoch(
                hard_set,
                anchor_events,
                epoch=hard_epoch,
                cap_per_event=int(manifest["cap_per_event"]),
                seed=int(manifest["seed"]),
                batch_size=args.hard_batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            hard_iterator = iter(hard_loader)
            hard_batch = next(hard_iterator)

        set_h1_train_mode(student)
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        pre, sar, teacher_sar, _, damage, _ = parse_p4_batch(natural_batch, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, teacher_damage, _ = teacher(pre, teacher_sar, return_aux=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, student_damage, _ = student(pre, sar, return_aux=True)
            damage_loss, _, _ = segmentation_loss(
                student_damage,
                damage,
                ce,
                True,
                args.ce_weight,
                args.lovasz_weight,
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
            natural_loss = damage_loss + args.distill_weight * protect_loss
        natural_loss.backward()

        hard_pre = hard_batch["pre_img"].float().to(device)
        hard_post = hard_batch["teacher_post_img"].float().to(device)
        # H0/H1-S0 were cached with one FP32 image per forward.  Both dtype and
        # batch size are part of this exact replay contract: CUDA may select a
        # different convolution kernel for B=2, and a tiny logit round-off can
        # alter a boundary pixel.  Keep a logical hard batch, but replay each
        # image separately and average its gradients.  This preserves the loss
        # weight/sampling protocol without weakening the component contract.
        pair_rows = hard_batch["pair_row"].long().flatten()
        positive_columns = torch.randint(
            positive_indices.shape[1],
            (pair_rows.numel(),),
            generator=pair_generator,
        )
        positive_train_indices = positive_indices[pair_rows, positive_columns]
        negative_columns = torch.stack([
            torch.randperm(
                negative_indices.shape[1], generator=pair_generator
            )[: args.negative_count]
            for _ in range(pair_rows.numel())
        ])
        negative_train_indices = torch.gather(
            negative_indices[pair_rows], 1, negative_columns
        )
        hard_loss_values = []
        hard_microbatch_count = int(hard_pre.shape[0])
        for batch_index in range(hard_microbatch_count):
            # The cached connected component is a frozen-Teacher object.  Use
            # the exact Teacher to reconstruct it; the Student supplies only
            # the trainable damage representation.  This also prevents a
            # harmless Student/Teacher numerical tie from changing supervision
            # to a different building.
            with torch.no_grad():
                teacher_hard_loc, _ = teacher(
                    hard_pre[batch_index : batch_index + 1],
                    hard_post[batch_index : batch_index + 1],
                )
            sample_loc, _, sample_aux = student(
                hard_pre[batch_index : batch_index + 1],
                hard_post[batch_index : batch_index + 1],
                return_aux=True,
            )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=False
            ):
                teacher_building_mask = (
                    teacher_hard_loc.argmax(dim=1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    == 1
                )
                student_building_mask = (
                    sample_loc.argmax(dim=1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    == 1
                )
                hard_location_disagreement_pixels += int(
                    (teacher_building_mask != student_building_mask).sum()
                )
                hard_location_replay_samples += 1
                component = match_cached_component(
                    teacher_building_mask,
                    hard_batch["expected_bbox"][batch_index].tolist(),
                    int(hard_batch["expected_area"][batch_index]),
                )
                if component is None:
                    raise RuntimeError(
                        "H1 frozen-Teacher component replay failed for "
                        f"{hard_batch['name'][batch_index]} | "
                        f"bbox={hard_batch['expected_bbox'][batch_index].tolist()} | "
                        f"area={int(hard_batch['expected_area'][batch_index])}"
                    )
                query_key = differentiable_region_mean_key(
                    sample_aux["fused_roi"], component, batch_index=0
                )
                positive_key = support_feature[
                    positive_train_indices[batch_index]
                ].to(device)
                negative_keys = support_feature[
                    negative_train_indices[batch_index]
                ].to(device)
                projected_query = projector(query_key[None])
                projected_positive = projector(positive_key[None])
                projected_negative = projector(negative_keys)[None]
                sample_hard_loss = occurrence_info_nce(
                    projected_query,
                    projected_positive,
                    projected_negative,
                    temperature=args.temperature,
                )
            (
                args.hard_weight
                * sample_hard_loss
                / max(hard_microbatch_count, 1)
            ).backward()
            hard_loss_values.append(sample_hard_loss.detach())
        hard_loss = torch.stack(hard_loss_values).mean()
        grad_norms = {}
        for prefix in ("fusion_layer_3", "clf_2"):
            grad_norms[prefix] = sum(
                float(parameter.grad.detach().abs().sum())
                for name, parameter in student.named_parameters()
                if name.startswith(prefix + ".") and parameter.grad is not None
            )
        grad_norms["projector"] = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in projector.parameters()
            if parameter.grad is not None
        )
        if step == 1 and any(value <= 0.0 for value in grad_norms.values()):
            raise RuntimeError(f"H1 gradient contract failed: {grad_norms}")
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], 5.0
        )
        optimizer.step()

        if step % args.log_every == 0:
            print(json.dumps({
                "step": step,
                "damage_loss": float(damage_loss.detach()),
                "protect_loss": float(protect_loss.detach()),
                "hard_loss": float(hard_loss.detach()),
                "hard_epoch": hard_epoch,
                "teacher_student_location_disagreement_pixels": (
                    hard_location_disagreement_pixels
                ),
                "hard_location_replay_samples": hard_location_replay_samples,
                "grad_norms": grad_norms,
                "steps_per_second": step / max(time.perf_counter() - start_time, 1e-6),
            }), flush=True)

        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics = evaluate(student, val_loader, device, args.ignore_index, 0)
            checks, diagnostics = h1_validation_checks(
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
            print(f"H1 checks: {checks}", flush=True)
            atomic_save(
                checkpoint_payload(student, projector, optimizer, step=step, metrics=metrics, args=args),
                output_dir / "latest_resume.pth",
            )
            if result["passed"] and (
                best_admitted is None
                or diagnostics["damaged_gain_pp"] > best_admitted["diagnostics"]["damaged_gain_pp"]
            ):
                best_admitted = result
                atomic_save(
                    checkpoint_payload(student, projector, optimizer, step=step, metrics=metrics, args=args),
                    output_dir / "best_admitted.pth",
                )

    decision = (
        "ADMIT_CMR_H1A_TO_1500_STEP_EXTENSION_NO_TEST"
        if best_admitted is not None
        else "STOP_CMR_H1A_HARD_OCCURRENCE_SCREEN_DO_NOT_TEST"
    )
    summary = {
        "protocol": "CMR_H1A_train_only_occurrence_screen_summary_v1",
        "baseline": baseline,
        "history": history,
        "best_admitted": best_admitted,
        "selected_checkpoint": (
            "best_admitted.pth"
            if best_admitted
            else str(args.p3_checkpoint)
        ),
        "safe_step0_manifest": "safe_step0.json",
        "hard_epoch_completed": hard_epoch,
        "teacher_student_location_disagreement_pixels": (
            hard_location_disagreement_pixels
        ),
        "hard_location_replay_samples": hard_location_replay_samples,
        "parameter_counts": parameter_counts,
        "decision": decision,
        "inference_uses_memory": False,
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
