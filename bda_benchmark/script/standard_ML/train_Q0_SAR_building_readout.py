#!/usr/bin/env python3
"""Train the controlled Q0 building-conditioned SAR readout.

The official test split is intentionally unsupported.  Q0 is a diagnosis of
readability, not a candidate final model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    build_h6_sdcr_damageformer,
)
from bda_benchmark.model.DamageFormer_RGBSAR_Q0_Readout import (  # noqa: E402
    Q0_VARIANTS,
    Q0BuildingConditionedSARReadout,
    assert_q0_freeze_contract,
)
from bda_benchmark.script.standard_ML.q0_readout_runtime import (  # noqa: E402
    Q0MetricAccumulator,
    SourceSARAuxAccumulator,
    effective_number_weights,
    masked_three_class_ce,
    selection_key,
    stage4_cell_counts,
    supervised_mask,
)


SOURCE_PROTOCOLS = ("H6_SDCR_best_weights_v1", "H6_SDCR_resume_v1")
BEST_PROTOCOL = "Q0_SAR_building_readout_best_v1"
RESUME_PROTOCOL = "Q0_SAR_building_readout_resume_v1"


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def atomic_json(payload: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_locked_source(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol not in SOURCE_PROTOCOLS:
        raise RuntimeError(f"unsupported Q0 source protocol: {protocol!r}")
    if variant != "a2":
        raise RuntimeError(f"Q0 requires H6-A2, received variant={variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("H6-A2 checkpoint has no non-empty model state")
    model = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H6-A2 source load failed: {incompatible}")
    model.to(device).eval()
    provenance = {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "protocol": protocol,
        "variant": variant,
        "step": int(checkpoint.get("step", 0)),
        "strict_load": True,
    }
    return model, provenance


def make_dataset(args, data_list: str, split: str, max_iters: int | None = None):
    return HeterogeneousRGBSARDataset(
        dataset_path=args.dataset_path,
        data_list=data_list,
        crop_size=args.crop_size,
        max_iters=max_iters,
        split=split,
        variant="h3",
        sar_stats_path=args.sar_stats_path,
        focus_probability=args.focus_probability if split == "train" else 0.0,
        focus_class_weights=(0.0, 1.0, 4.0, 2.0),
    )


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
        generator=generator,
    )


@torch.inference_mode()
def count_train_cells(model, loader, device, minimum_purity: float, max_batches: int):
    counts = torch.zeros(3, dtype=torch.long)
    processed = 0
    for batch_index, (_, sar, _, damage, _) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        stage4 = model.source.sar_encoder(sar[:, :1])[-1]
        counts += stage4_cell_counts(stage4, damage, minimum_purity).cpu()
        processed += int(sar.shape[0])
    if torch.any(counts <= 0):
        raise RuntimeError(f"train-only Stage4 counts are incomplete: {counts.tolist()}")
    return counts, processed


@torch.inference_mode()
def evaluate(model, loader, device, mask_threshold: float, max_batches: int = 0):
    model.eval()
    accumulator = Q0MetricAccumulator()
    source_accumulator = SourceSARAuxAccumulator()
    processed = 0
    for batch_index, (rgb, sar, _, damage, names) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        outputs = model(rgb, sar)
        probability = outputs["building_probability"] if model.uses_p3_mask else None
        mask = supervised_mask(damage, probability, mask_threshold)
        accumulator.update(outputs["logits"], damage, mask, names)
        source_accumulator.update(outputs["source_sar_aux_logits"], damage)
        processed += len(names)
        if processed % 50 == 0:
            print(f"Q0 validation: {processed}", flush=True)
    metrics = accumulator.compute()
    source_metrics = source_accumulator.compute()
    metrics["source_sar_aux"] = source_metrics
    conditional = metrics["conditional"]
    metrics["delta_vs_source_sar_aux_pp"] = {
        key: 100.0 * (float(conditional[key]) - float(source_metrics[key]))
        for key in (
            "intact_recall", "damaged_recall", "destroyed_recall",
            "intact_f1", "damaged_f1", "destroyed_f1",
        )
    }
    metrics["advancement_gate"] = {
        "passed": bool(
            float(conditional["damaged_recall"]) >= 0.30
            and float(conditional["intact_f1"]) >= float(source_metrics["intact_f1"]) - 0.05
            and float(conditional["destroyed_f1"]) >= float(source_metrics["destroyed_f1"]) - 0.05
        ),
        "checks": {
            "damaged_recall_at_least_30pct": float(conditional["damaged_recall"]) >= 0.30,
            "intact_f1_drop_at_most_5pp": float(conditional["intact_f1"]) >= float(source_metrics["intact_f1"]) - 0.05,
            "destroyed_f1_drop_at_most_5pp": float(conditional["destroyed_f1"]) >= float(source_metrics["destroyed_f1"]) - 0.05,
        },
        "note": "Seed/event consistency is decided only by the multi-seed summary, not by this single run.",
    }
    metrics["processed_images"] = processed
    metrics["mask_source"] = "frozen_p3" if model.uses_p3_mask else "oracle_gt"
    return metrics


def checkpoint_payload(protocol, model, args, source, step, metrics, weights, optimizer=None):
    payload = {
        "protocol": protocol,
        "variant": model.variant,
        "step": int(step),
        "readout": model.readout.state_dict(),
        "source": dict(source),
        "metrics": metrics,
        "class_weights": [float(value) for value in weights.cpu()],
        "mask_threshold": float(args.mask_threshold),
        "minimum_purity": float(args.minimum_purity),
        "selection_policy": "macro_f1 -> event_macro_f1 -> end_to_end_damaged_recall -> conditional_damaged_recall",
        "final_model_modified": False,
        "test_accessed": False,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Q0 frozen Stage4 building-conditioned SAR readout")
    parser.add_argument("--variant", choices=Q0_VARIANTS, required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=1500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focus_probability", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--minimum_purity", type=float, default=0.75)
    parser.add_argument("--class_balance_beta", type=float, default=0.9999)
    parser.add_argument("--maximum_weight_ratio", type=float, default=5.0)
    parser.add_argument("--count_max_batches", type=int, default=0)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accumulation_steps <= 0 or args.optimizer_steps < 0:
        raise ValueError("invalid optimizer/accumulation steps")
    if not 0.0 < args.mask_threshold < 1.0:
        raise ValueError("mask threshold must lie in (0,1)")
    seed_everything(args.seed)
    device = torch.device(args.device)
    source_model, source = load_locked_source(args.source_checkpoint, device)
    model = Q0BuildingConditionedSARReadout(source_model, args.variant).to(device)
    assert_q0_freeze_contract(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_weights = torch.ones(3, dtype=torch.float32)
    class_count_report = None
    if args.variant == "q0_a1":
        count_dataset = make_dataset(args, args.train_list, split="val")
        count_loader = make_loader(
            count_dataset, args.eval_batch_size, args.num_workers, False, args.seed
        )
        counts, count_images = count_train_cells(
            model, count_loader, device, args.minimum_purity, args.count_max_batches
        )
        class_weights = effective_number_weights(
            counts.tolist(), args.class_balance_beta, args.maximum_weight_ratio
        )
        class_count_report = {
            "protocol": "Q0_train_only_stage4_cell_counts_v1",
            "train_list": str(Path(args.train_list).resolve()),
            "processed_images": count_images,
            "counts_intact_damaged_destroyed": counts.tolist(),
            "weights_intact_damaged_destroyed": class_weights.tolist(),
            "minimum_purity": args.minimum_purity,
            "test_accessed": False,
        }
        atomic_json(class_count_report, output_dir / "train_only_class_counts.json")
    class_weights = class_weights.to(device)

    train_examples = max(
        args.train_batch_size,
        args.optimizer_steps * args.train_batch_size * args.accumulation_steps,
    )
    train_dataset = make_dataset(args, args.train_list, "train", max_iters=train_examples)
    val_dataset = make_dataset(args, args.val_list, "val")
    train_loader = make_loader(
        train_dataset, args.train_batch_size, args.num_workers, True, args.seed
    )
    val_loader = make_loader(
        val_dataset, args.eval_batch_size, args.num_workers, False, args.seed
    )
    optimizer = torch.optim.AdamW(
        model.readout.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    best_metrics = None
    best_key = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("protocol") != RESUME_PROTOCOL or resume.get("variant") != args.variant:
            raise RuntimeError("Q0 resume protocol/variant mismatch")
        if resume.get("source", {}).get("sha256") != source["sha256"]:
            raise RuntimeError("Q0 resume source checkpoint fingerprint mismatch")
        model.readout.load_state_dict(resume["readout"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume.get("step", 0))
        best_metrics = resume.get("best_metrics")
        best_key = tuple(resume.get("best_key", [])) or None

    print("=" * 88)
    print(f"Q0 variant/source : {args.variant} / {source['path']}")
    print(f"Mask/loss         : {'P3' if model.uses_p3_mask else 'Oracle'} / "
          f"{'class-balanced CE' if args.variant == 'q0_a1' else 'plain CE'}")
    print(f"Trainable params  : {sum(value.numel() for value in model.parameters() if value.requires_grad)}")
    print("Final H6 output   : untouched; Q0 is diagnosis only")
    print("Test accessed     : NEVER")
    print("=" * 88, flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = start_step
    micro_step = 0
    running_loss = 0.0
    running_pixels = 0
    # Some random crops contain no supervised building pixels.  A fixed-length
    # ``for`` loop therefore stopped at 1493/1494 rather than the declared
    # 1500 optimizer steps.  Re-open the finite loader until the *optimizer*
    # budget is met exactly; skipped empty crops never consume a step.
    train_iterator = iter(train_loader)
    while optimizer_step < args.optimizer_steps:
        try:
            rgb, sar, _, damage, _ = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            rgb, sar, _, damage, _ = next(train_iterator)
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        outputs = model(rgb, sar)
        probability = outputs["building_probability"] if model.uses_p3_mask else None
        mask = supervised_mask(damage, probability, args.mask_threshold)
        loss, supervised_pixels = masked_three_class_ce(
            outputs["logits"], damage, mask, class_weights
        )
        if supervised_pixels == 0:
            continue
        (loss / args.accumulation_steps).backward()
        running_loss += float(loss.detach())
        running_pixels += supervised_pixels
        micro_step += 1
        if micro_step % args.accumulation_steps:
            continue
        torch.nn.utils.clip_grad_norm_(
            model.readout.parameters(), max_norm=5.0, error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        if optimizer_step % args.log_every == 0:
            print(json.dumps({
                "step": optimizer_step,
                "variant": args.variant,
                "loss": running_loss / args.log_every,
                "supervised_pixels": running_pixels,
            }), flush=True)
            running_loss = 0.0
            running_pixels = 0
        should_eval = optimizer_step % args.eval_every == 0 or optimizer_step == args.optimizer_steps
        if should_eval:
            metrics = evaluate(
                model, val_loader, device, args.mask_threshold, args.val_max_batches
            )
            key = selection_key(metrics)
            conditional = metrics["conditional"]
            print(
                f"Q0 validation step={optimizer_step} | macroF1={100*conditional['macro_f1']:.2f} "
                f"| recall I/D/X={100*conditional['intact_recall']:.2f}/"
                f"{100*conditional['damaged_recall']:.2f}/{100*conditional['destroyed_recall']:.2f} "
                f"| coverage={100*metrics['building_coverage']:.2f} "
                f"| event-macroF1={100*metrics['event_macro_f1']:.2f}", flush=True,
            )
            record = {
                "protocol": "Q0_SAR_building_readout_validation_v1",
                "variant": args.variant,
                "step": optimizer_step,
                "source": source,
                "metrics": metrics,
                "test_accessed": False,
            }
            atomic_json(record, output_dir / "latest_validation.json")
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = metrics
                atomic_save(
                    checkpoint_payload(
                        BEST_PROTOCOL, model, args, source, optimizer_step,
                        metrics, class_weights,
                    ),
                    output_dir / "best_readout.pth",
                )
                print("New best Q0 validation readout.", flush=True)
            resume_payload = checkpoint_payload(
                RESUME_PROTOCOL, model, args, source, optimizer_step,
                metrics, class_weights, optimizer,
            )
            resume_payload["best_metrics"] = best_metrics
            resume_payload["best_key"] = list(best_key) if best_key is not None else None
            atomic_save(resume_payload, output_dir / "latest_resume.pth")
            model.train()

    summary = {
        "protocol": "Q0_SAR_building_readout_training_summary_v1",
        "variant": args.variant,
        "completed_step": optimizer_step,
        "source": source,
        "best_metrics": best_metrics,
        "class_count_report": class_count_report,
        "trainable_parameter_names": model.trainable_parameter_names(),
        "final_model_modified": False,
        "test_accessed": False,
    }
    atomic_json(summary, output_dir / "training_summary.json")
    print(f"Q0 training complete at step {optimizer_step}; saved {output_dir}")


if __name__ == "__main__":
    main()
