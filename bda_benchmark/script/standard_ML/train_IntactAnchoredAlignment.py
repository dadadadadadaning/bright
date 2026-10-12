# -*- coding: utf-8 -*-
"""Isolated AL0/AL0-N/AL1--AL3 experiment entry."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.make_data_loader_heterogeneous_rgbsar import (
    HeterogeneousRGBSARDataset,
    RGB_MEAN,
    RGB_STD,
)
from model.IntactAnchoredAlignment import (
    IntactAnchoredMultiScaleAlignment,
    alignment_training_loss,
)
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from script.standard_ML.al_alignment_runtime import (
    AL_PROTOCOL,
    FrozenH6Stage23,
    _region_records,
    alignment_checkpoint_payload,
    evidence_audit,
    file_sha256,
    load_alignment_checkpoint,
    occurrence_evidence_audit,
    scene_relative_residual,
)
from script.standard_ML.im4cd_matching_runtime import hard_local_matching_maps
from script.standard_ML.rm_region_retrieval_runtime import event_from_name


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(args, split: str, *, repeat: bool = False) -> DataLoader:
    data_list = args.train_data_list_path if split == "train" else args.val_data_list_path
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        data_list,
        args.crop_size,
        max_iters=(args.optimizer_steps * args.train_batch_size if repeat else None),
        split=split,
        variant="h4",
        sar_stats_path=args.sar_stats_path,
        focus_probability=(args.intact_focus_probability if split == "train" else 0.0),
        focus_class_weights=(0.0, 1.0, 0.0, 0.0),
    )
    return DataLoader(
        dataset,
        batch_size=(args.train_batch_size if split == "train" else args.eval_batch_size),
        shuffle=split == "train",
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=split == "train",
    )


def _batch(batch, device: torch.device):
    rgb, sar, _loc, damage, names = batch
    return (
        rgb.float().to(device, non_blocking=True),
        sar.float().to(device, non_blocking=True),
        damage.long().to(device, non_blocking=True),
        list(names),
    )


@torch.inference_mode()
def _validate_intact_loss(model, source, loader, device) -> Dict[str, float]:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    unaligned = 0.0
    for batch in loader:
        rgb, sar, damage, _ = _batch(batch, device)
        rgb_stages, sar_stages = source(rgb, sar[:, :1])
        output = model(rgb_stages, sar_stages)
        mask = (damage == 1).float().unsqueeze(1)
        for stage in model.stages:
            residual = output["stages"][stage]["aligned_residual"]
            baseline = output["stages"][stage]["unaligned_residual"]
            stage_mask = F.interpolate(mask, residual.shape[-2:], mode="area")
            count = float(stage_mask.sum())
            if count <= 0:
                continue
            numerator += float((residual * stage_mask).sum())
            unaligned += float((baseline * stage_mask).sum())
            denominator += count
    return {
        "intact_aligned_residual": numerator / max(denominator, 1.0),
        "intact_unaligned_residual": unaligned / max(denominator, 1.0),
        "intact_feature_pixels": denominator,
    }


def train_al1(args, source, device) -> None:
    model = IntactAnchoredMultiScaleAlignment(
        args.descriptor_channels, args.radius, args.temperature, occurrence_head=False
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = _loader(args, "train", repeat=True)
    val_loader = _loader(args, "val")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    completed = 0
    iterator = iter(train_loader)
    while completed < args.optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        rgb, sar, damage, _ = _batch(batch, device)
        intact = (damage == 1).float().unsqueeze(1)
        if int(intact.sum()) == 0:
            continue
        with torch.no_grad():
            rgb_stages, sar_stages = source(rgb, sar[:, :1])
        model.train()
        result = model(rgb_stages, sar_stages)
        losses = [
            alignment_training_loss(
                result["stages"][stage],
                intact,
                smoothness_weight=args.smoothness_weight,
                magnitude_weight=args.magnitude_weight,
                variance_weight=args.variance_weight,
            )
            for stage in model.stages
        ]
        total = torch.stack([row["total"] for row in losses]).mean()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        completed += 1
        if completed % args.log_every == 0:
            print(
                f"AL1 train | step={completed} | loss={float(total):.6f} | "
                f"intact={int(sum(float(row['intact_pixels']) for row in losses))}",
                flush=True,
            )
        if completed % args.eval_every == 0 or completed == args.optimizer_steps:
            metrics = _validate_intact_loss(model, source, val_loader, device)
            payload = alignment_checkpoint_payload(
                model, step=completed, h6_source=source, metrics=metrics, optimizer=optimizer
            )
            torch.save(payload, output / "latest_resume.pth")
            value = metrics["intact_aligned_residual"]
            print(
                f"AL1 validation | step={completed} | aligned={value:.6f} | "
                f"unaligned={metrics['intact_unaligned_residual']:.6f}",
                flush=True,
            )
            if value < best:
                best = value
                torch.save(payload, output / "best_intact_alignment.pth")
    summary = {
        "protocol": AL_PROTOCOL,
        "stage": "al1",
        "completed_step": completed,
        "best_intact_aligned_residual": best,
        "selection_metric": "validation_intact_aligned_residual_only",
        "h6_checkpoint_sha256": source.checkpoint_sha256,
        "test_accessed": False,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fixed_al0(source, rgb, sar) -> Mapping[str, torch.Tensor]:
    rgb_stages, sar_stages = source(rgb, sar[:, :1])
    residuals = []
    baselines = []
    target = rgb_stages["stage2"].shape[-2:]
    for stage in ("stage2", "stage3"):
        maps = hard_local_matching_maps(
            rgb_stages[stage], sar_stages[stage], window_size=5, temperature=0.10
        )
        residuals.append(
            F.interpolate(maps["hard_residual"], target, mode="bilinear", align_corners=False)
        )
        baselines.append(
            F.interpolate(
                maps["colocated_cosine_distance"], target,
                mode="bilinear", align_corners=False,
            )
        )
    return {
        "aligned_residual": torch.stack(residuals).mean(0),
        "unaligned_residual": torch.stack(baselines).mean(0),
    }


def _load_frozen_p3(path: str, device: torch.device):
    if not path:
        raise ValueError("AL0-N requires --p3_checkpoint")
    model = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(model, path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _historical_p3_sar(sar: torch.Tensor) -> torch.Tensor:
    if sar.ndim != 4 or sar.shape[1] < 2:
        raise ValueError("AL0-N requires H4 SAR [native, raw-for-P3]")
    raw = sar[:, 1:2].float()
    mean = torch.as_tensor(RGB_MEAN, device=raw.device, dtype=raw.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(RGB_STD, device=raw.device, dtype=raw.dtype).view(1, 3, 1, 1)
    return (raw.expand(-1, 3, -1, -1) - mean) / std


@torch.inference_mode()
def audit_scene_relative_evidence(args, source, device) -> None:
    """AL0-N: target-free scene normalisation of frozen raw residuals."""

    p3 = _load_frozen_p3(args.p3_checkpoint, device)
    loader = _loader(args, "val")
    normalised_records: List[Dict[str, object]] = []
    raw_records: List[Dict[str, object]] = []
    centres: List[float] = []
    scales: List[float] = []
    supports: List[float] = []
    for index, batch in enumerate(loader, 1):
        rgb, sar, damage, names = _batch(batch, device)
        localisation, _ = p3(rgb, _historical_p3_sar(sar))
        p_building = torch.softmax(localisation.float(), dim=1)[:, 1:2]
        rgb_stages, sar_stages = source(rgb, sar[:, :1])
        target_size = rgb_stages["stage2"].shape[-2:]
        relative_stages = []
        raw_stages = []
        for stage in ("stage2", "stage3"):
            rgb_unit = F.normalize(rgb_stages[stage].float(), dim=1, eps=1e-6)
            sar_unit = F.normalize(sar_stages[stage].float(), dim=1, eps=1e-6)
            raw_residual = (1.0 - (rgb_unit * sar_unit).sum(1, keepdim=True)).clamp(0, 2)
            relative = scene_relative_residual(
                raw_residual,
                p_building,
                probability_threshold=args.building_probability_threshold,
                minimum_support=args.minimum_scene_support,
            )
            relative_stages.append(
                F.interpolate(
                    relative["normalised"], target_size,
                    mode="bilinear", align_corners=False,
                )
            )
            raw_stages.append(
                F.interpolate(
                    raw_residual, target_size,
                    mode="bilinear", align_corners=False,
                )
            )
            centres.extend(float(value) for value in relative["centre"].cpu())
            scales.extend(float(value) for value in relative["scale"].cpu())
            supports.extend(float(value) for value in relative["support_count"].cpu())
        normalised_score = torch.stack(relative_stages).mean(0)
        raw_score = torch.stack(raw_stages).mean(0)
        normalised_records.extend(
            _region_records(normalised_score, damage, names, min_area=args.min_area)
        )
        raw_records.extend(
            _region_records(raw_score, damage, names, min_area=args.min_area)
        )
        if index % 50 == 0 or index == len(loader):
            print(f"AL0-N audit: {index}/{len(loader)}", flush=True)

    audit = occurrence_evidence_audit(
        normalised_records,
        raw_records,
        minimum_event_gain=args.minimum_occurrence_event_gain,
    )
    payload = {
        "protocol": AL_PROTOCOL,
        "stage": "al0n",
        "scope": "validation-only target-free scene-relative occurrence evidence audit",
        "normalisation": {
            "definition": "per-image per-stage (residual - P3-building median) / (1.4826*MAD)",
            "building_probability_threshold": args.building_probability_threshold,
            "minimum_scene_support": args.minimum_scene_support,
            "centre_mean": float(np.mean(centres)) if centres else None,
            "scale_mean": float(np.mean(scales)) if scales else None,
            "support_mean": float(np.mean(supports)) if supports else None,
        },
        "p3_checkpoint": str(Path(args.p3_checkpoint).resolve()),
        "p3_checkpoint_sha256": file_sha256(args.p3_checkpoint),
        "h6_checkpoint": source.checkpoint_path,
        "h6_checkpoint_sha256": source.checkpoint_sha256,
        **audit,
        "decision": (
            "ADMIT_TO_RAW_OCCURRENCE_HEAD"
            if audit["passed"]
            else "STOP_SCENE_RELATIVE_OCCURRENCE_EVIDENCE_INSUFFICIENT"
        ),
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    affected = audit["affected_vs_intact"]
    damaged = audit["damaged_vs_intact_diagnostic"]
    print("=" * 96)
    print("Stage          : al0n")
    print(f"Affected AUC   : {affected['global_auroc']:.4f}")
    print(f"Affected event : {affected['event_macro_auroc']:.4f}")
    print(
        f"Consistent     : {affected['direction_consistent_events']}/"
        f"{affected['comparable_events']}"
    )
    print(f"Event gain     : {audit['affected_event_gain']:+.4f}")
    print(f"Damaged diag   : global={damaged['global_auroc']:.4f} | event={damaged['event_macro_auroc']:.4f}")
    print(f"Checks         : {audit['checks']}")
    print(f"Decision       : {payload['decision']}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


@torch.inference_mode()
def audit_evidence(args, source, device, learned: bool) -> None:
    model = None
    checkpoint = None
    if learned:
        model = IntactAnchoredMultiScaleAlignment(
            args.descriptor_channels, args.radius, args.temperature, occurrence_head=False
        ).to(device)
        checkpoint = load_alignment_checkpoint(model, args.alignment_checkpoint, source)
        model.eval()
    loader = _loader(args, "val")
    aligned_records: List[Dict[str, object]] = []
    unaligned_records: List[Dict[str, object]] = []
    for index, batch in enumerate(loader, 1):
        rgb, sar, damage, names = _batch(batch, device)
        if learned:
            rgb_stages, sar_stages = source(rgb, sar[:, :1])
            result = model(rgb_stages, sar_stages)
        else:
            result = _fixed_al0(source, rgb, sar)
        aligned_records.extend(
            _region_records(result["aligned_residual"], damage, names, min_area=args.min_area)
        )
        unaligned_records.extend(
            _region_records(result["unaligned_residual"], damage, names, min_area=args.min_area)
        )
        if index % 50 == 0 or index == len(loader):
            print(f"{'AL2' if learned else 'AL0'} audit: {index}/{len(loader)}", flush=True)
    audit = evidence_audit(aligned_records, unaligned_records)
    payload = {
        "protocol": AL_PROTOCOL,
        "stage": "al2" if learned else "al0",
        "scope": "validation-only building-region evidence audit",
        "alignment_checkpoint": (
            str(Path(args.alignment_checkpoint).resolve()) if learned else None
        ),
        "alignment_step": int(checkpoint.get("step", 0)) if checkpoint else 0,
        "h6_checkpoint": source.checkpoint_path,
        "h6_checkpoint_sha256": source.checkpoint_sha256,
        **audit,
        "decision": (
            "ADMIT_TO_AL3_OCCURRENCE"
            if learned and audit["passed"]
            else (
                "AL0_BASELINE_COMPLETE_DO_NOT_FUSE"
                if not learned
                else "STOP_BEFORE_AL3_ALIGNMENT_EVIDENCE_INSUFFICIENT"
            )
        ),
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    damaged = audit["damaged_vs_intact"]
    print("=" * 96)
    print(f"Stage          : {payload['stage']}")
    print(f"Global AUC     : {damaged['global_auroc']:.4f}")
    print(f"Event AUC      : {damaged['event_macro_auroc']:.4f}")
    print(f"Consistent     : {damaged['direction_consistent_events']}/{damaged['comparable_events']}")
    print(f"Aligned gain   : {audit['aligned_gain_global_auroc']:+.4f}")
    print(f"Checks         : {audit['checks']}")
    print(f"Decision       : {payload['decision']}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


def _occurrence_counts(logits, target, names, table):
    logits = F.interpolate(logits.float(), target.shape[-2:], mode="bilinear", align_corners=False)
    prediction = logits.argmax(1).detach().cpu()
    target = target.detach().cpu()
    for index, name in enumerate(names):
        event = event_from_name(str(name))
        confusion = table.setdefault(event, torch.zeros((2, 2), dtype=torch.long))
        valid = (target[index] >= 1) & (target[index] <= 3)
        truth = ((target[index] >= 2) & (target[index] <= 3)).long()
        for actual in range(2):
            for guessed in range(2):
                confusion[actual, guessed] += (
                    valid & (truth == actual) & (prediction[index] == guessed)
                ).sum()


def _metrics_from_counts(table) -> Dict[str, object]:
    total = sum(table.values(), torch.zeros((2, 2), dtype=torch.long))

    def metrics(confusion):
        f1 = []
        recall = []
        for cls in range(2):
            tp = float(confusion[cls, cls])
            fp = float(confusion[:, cls].sum() - confusion[cls, cls])
            fn = float(confusion[cls, :].sum() - confusion[cls, cls])
            f1.append(2 * tp / max(2 * tp + fp + fn, 1.0))
            recall.append(tp / max(tp + fn, 1.0))
        return {"macro_f1": sum(f1) / 2, "affected_f1": f1[1], "affected_recall": recall[1]}

    overall = metrics(total)
    events = {event: metrics(value) for event, value in table.items()}
    overall["event_macro_f1"] = sum(row["macro_f1"] for row in events.values()) / max(len(events), 1)
    overall["events"] = events
    return overall


@torch.inference_mode()
def _validate_occurrence(model, source, loader, device):
    model.eval()
    table = {}
    for batch in loader:
        rgb, sar, damage, names = _batch(batch, device)
        rgb_stages, sar_stages = source(rgb, sar[:, :1])
        result = model(rgb_stages, sar_stages)
        _occurrence_counts(result["occurrence_logits"], damage, names, table)
    return _metrics_from_counts(table)


def train_al3(args, source, device) -> None:
    gate = json.loads(Path(args.al2_gate_json).read_text(encoding="utf-8"))
    if gate.get("protocol") != AL_PROTOCOL or gate.get("stage") != "al2":
        raise RuntimeError("AL3 requires an AL2 gate JSON")
    if not bool(gate.get("passed", False)):
        raise RuntimeError("AL2 gate failed: AL3 occurrence training is forbidden")
    model = IntactAnchoredMultiScaleAlignment(
        args.descriptor_channels, args.radius, args.temperature, occurrence_head=True
    ).to(device)
    load_alignment_checkpoint(
        model, args.alignment_checkpoint, source, allow_occurrence_missing=True
    )
    model.freeze_alignment()
    optimizer = torch.optim.AdamW(
        model.occurrence_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = _loader(args, "train", repeat=True)
    val_loader = _loader(args, "val")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    completed = 0
    iterator = iter(train_loader)
    while completed < args.optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        rgb, sar, damage, _ = _batch(batch, device)
        with torch.no_grad():
            rgb_stages, sar_stages = source(rgb, sar[:, :1])
        model.train()
        model.aligners.eval()
        result = model(rgb_stages, sar_stages)
        logits = F.interpolate(
            result["occurrence_logits"], damage.shape[-2:], mode="bilinear", align_corners=False
        )
        valid = (damage >= 1) & (damage <= 3)
        target = ((damage >= 2) & (damage <= 3)).long()
        if int(valid.sum()) == 0:
            continue
        loss = F.cross_entropy(
            logits.permute(0, 2, 3, 1)[valid],
            target[valid],
            weight=torch.tensor([1.0, args.affected_weight], device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.occurrence_head.parameters(), args.grad_clip)
        optimizer.step()
        completed += 1
        if completed % args.log_every == 0:
            print(f"AL3 train | step={completed} | loss={float(loss):.6f}", flush=True)
        if completed % args.eval_every == 0 or completed == args.optimizer_steps:
            metrics = _validate_occurrence(model, source, val_loader, device)
            payload = {
                "protocol": AL_PROTOCOL,
                "stage": "al3",
                "step": completed,
                "model": model.state_dict(),
                "metrics": metrics,
                "alignment_checkpoint": str(Path(args.alignment_checkpoint).resolve()),
                "al2_gate_json": str(Path(args.al2_gate_json).resolve()),
                "h6_checkpoint_sha256": source.checkpoint_sha256,
                "test_accessed": False,
            }
            torch.save(payload, output / "latest_resume.pth")
            print(
                f"AL3 validation | step={completed} | macroF1={100*metrics['macro_f1']:.4f} | "
                f"AffectedRec={100*metrics['affected_recall']:.4f} | "
                f"EventF1={100*metrics['event_macro_f1']:.4f}",
                flush=True,
            )
            if metrics["event_macro_f1"] > best:
                best = metrics["event_macro_f1"]
                torch.save(payload, output / "best_event_occurrence.pth")
    summary = {
        "protocol": AL_PROTOCOL,
        "stage": "al3",
        "completed_step": completed,
        "best_event_macro_f1": best,
        "test_accessed": False,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Intact-anchored alignment AL0/AL0-N/AL1--AL3")
    parser.add_argument("--stage", choices=("al0", "al0n", "al1", "al2", "al3"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--h6_a2_checkpoint", required=True)
    parser.add_argument("--p3_checkpoint", default="")
    parser.add_argument("--alignment_checkpoint", default="")
    parser.add_argument("--al2_gate_json", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--optimizer_steps", type=int, default=1500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--descriptor_channels", type=int, default=64)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--smoothness_weight", type=float, default=0.05)
    parser.add_argument("--magnitude_weight", type=float, default=0.01)
    parser.add_argument("--variance_weight", type=float, default=0.10)
    parser.add_argument("--intact_focus_probability", type=float, default=0.75)
    parser.add_argument("--affected_weight", type=float, default=2.0)
    parser.add_argument("--min_area", type=int, default=32)
    parser.add_argument("--building_probability_threshold", type=float, default=0.50)
    parser.add_argument("--minimum_scene_support", type=int, default=16)
    parser.add_argument("--minimum_occurrence_event_gain", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    source = FrozenH6Stage23(args.h6_a2_checkpoint).to(device)
    if args.stage == "al1":
        if not args.output_dir:
            raise ValueError("AL1 requires --output_dir")
        train_al1(args, source, device)
    elif args.stage == "al0n":
        if not args.output_json or not args.p3_checkpoint:
            raise ValueError("AL0-N requires --output_json and --p3_checkpoint")
        audit_scene_relative_evidence(args, source, device)
    elif args.stage in ("al0", "al2"):
        if not args.output_json:
            raise ValueError(f"{args.stage.upper()} requires --output_json")
        if args.stage == "al2" and not args.alignment_checkpoint:
            raise ValueError("AL2 requires --alignment_checkpoint")
        audit_evidence(args, source, device, learned=args.stage == "al2")
    else:
        if not args.output_dir or not args.alignment_checkpoint or not args.al2_gate_json:
            raise ValueError("AL3 requires output, AL1 checkpoint and AL2 gate")
        train_al3(args, source, device)


if __name__ == "__main__":
    main()
