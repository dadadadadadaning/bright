#!/usr/bin/env python3
"""Controlled Q2-B0/B1 building-region residual training.

The official test split is deliberately unsupported.  Step zero is admitted
as the exact P3 checkpoint, ensuring validation selection can always reject a
harmful learned residual and keep the frozen anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
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
    read_data_list,
)
from bda_benchmark.model.DamageFormer_RGBSAR_Q0_Readout import (  # noqa: E402
    Q0BuildingConditionedSARReadout,
)
from bda_benchmark.model.DamageFormer_RGBSAR_Q2_BuildingResidual import (  # noqa: E402
    Q2_VARIANTS,
    Q2BuildingResidualFusion,
    assert_q2_freeze_contract,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.script.standard_ML.q2_building_residual_runtime import (  # noqa: E402
    advancement_gate,
    evaluate_q2,
    region_cross_entropy,
    selection_key,
)
from bda_benchmark.script.standard_ML.train_Q0_SAR_building_readout import (  # noqa: E402
    BEST_PROTOCOL as Q0_PROTOCOL,
    atomic_json,
    atomic_save,
    file_sha256,
    load_locked_source,
    seed_everything,
)


BEST_PROTOCOL = "Q2_building_residual_best_v1"
RESUME_PROTOCOL = "Q2_building_residual_resume_v1"


def checkpoint_state(checkpoint: Mapping[str, object]):
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("P3 checkpoint has no model state")
    return state


def load_experts(args, device):
    p3_checkpoint = torch.load(args.p3_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL()
    incompatible = p3.load_state_dict(checkpoint_state(p3_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    source, a2_provenance = load_locked_source(args.a2_checkpoint, device)
    q0_checkpoint = torch.load(args.q0_checkpoint, map_location="cpu", weights_only=False)
    if q0_checkpoint.get("protocol") != Q0_PROTOCOL:
        raise RuntimeError(f"unexpected Q0 protocol: {q0_checkpoint.get('protocol')!r}")
    if q0_checkpoint.get("test_accessed") is not False:
        raise RuntimeError("Q2 refuses a Q0 checkpoint that accessed test")
    if str(q0_checkpoint.get("variant", "")).lower() != "q0_b0":
        raise RuntimeError("Q2-B1 requires the practical frozen-P3-mask Q0-B0 readout")
    if not bool(q0_checkpoint.get("metrics", {}).get("advancement_gate", {}).get("passed")):
        raise RuntimeError("Q2 refuses a Q0-B0 readout that did not pass validation")
    if q0_checkpoint.get("source", {}).get("sha256") != a2_provenance["sha256"]:
        raise RuntimeError("Q0/H6-A2 fingerprint mismatch")
    q0 = Q0BuildingConditionedSARReadout(source, q0_checkpoint["variant"])
    q0.readout.load_state_dict(q0_checkpoint["readout"], strict=True)
    p3.to(device).eval()
    q0.to(device).eval()
    provenance = {
        "p3": {
            "path": str(Path(args.p3_checkpoint).resolve()),
            "sha256": file_sha256(args.p3_checkpoint),
            "step": int(p3_checkpoint.get("step", 0)),
            "strict_load": True,
        },
        "a2": a2_provenance,
        "q0": {
            "path": str(Path(args.q0_checkpoint).resolve()),
            "sha256": file_sha256(args.q0_checkpoint),
            "protocol": q0_checkpoint.get("protocol"),
            "variant": q0_checkpoint.get("variant"),
            "step": int(q0_checkpoint.get("step", 0)),
            "strict_load": True,
        },
    }
    return p3, q0, provenance


def event_balanced_names(path: str, count: int, seed: int) -> list[str]:
    names = read_data_list(path)
    groups = defaultdict(list)
    for name in names:
        groups[re.sub(r"_\d+$", "", name)].append(name)
    events = sorted(groups)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    cursors = defaultdict(int)
    result = []
    while len(result) < count:
        order = list(events)
        rng.shuffle(order)
        for event in order:
            values = groups[event]
            result.append(values[cursors[event] % len(values)])
            cursors[event] += 1
            if len(result) >= count:
                break
    return result


def make_dataset(args, split: str, examples: int | None = None):
    data_list = args.train_list if split == "train" else args.val_list
    if split == "train" and examples is not None:
        data_list = event_balanced_names(data_list, examples, args.seed)
    return HeterogeneousRGBSARDataset(
        args.dataset_path,
        data_list,
        args.crop_size,
        split=split,
        variant="h4",
        sar_stats_path=args.sar_stats_path,
        focus_probability=args.focus_probability if split == "train" else 0.0,
        focus_class_weights=(0.0, 1.0, 4.0, 2.0),
    )


def make_loader(dataset, batch_size, workers, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def residual_state(model):
    return {
        "region_head": model.region_head.state_dict(),
        "transition_raw": model.transition_raw.detach().cpu(),
    }


def load_residual_state(model, state):
    model.region_head.load_state_dict(state["region_head"], strict=True)
    with torch.no_grad():
        model.transition_raw.copy_(state["transition_raw"].to(model.transition_raw))


def checkpoint_payload(protocol, model, args, provenance, step, metrics, base, audit, optimizer=None):
    payload = {
        "protocol": protocol,
        "variant": model.variant,
        "step": int(step),
        "residual": residual_state(model),
        "sources": provenance,
        "metrics": metrics,
        "base_metrics": base,
        "audit": audit,
        "selection_policy": "event-worst -> event-mean -> Damaged IoU -> valid mIoU",
        "proposal": {
            "threshold": args.proposal_threshold,
            "minimum_area": args.minimum_area,
            "minimum_building_fraction": args.minimum_building_fraction,
            "minimum_class_purity": args.minimum_class_purity,
        },
        "maximum_transition": args.maximum_transition,
        "test_accessed": False,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def parser():
    value = argparse.ArgumentParser("Q2 exact-P3 building region residual")
    value.add_argument("--variant", choices=Q2_VARIANTS, required=True)
    value.add_argument("--p3_checkpoint", required=True)
    value.add_argument("--a2_checkpoint", required=True)
    value.add_argument("--q0_checkpoint", required=True)
    value.add_argument("--dataset_path", required=True)
    value.add_argument("--train_list", required=True)
    value.add_argument("--val_list", required=True)
    value.add_argument("--sar_stats_path", required=True)
    value.add_argument("--output_dir", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--crop_size", type=int, default=640)
    value.add_argument("--train_batch_size", type=int, default=2)
    value.add_argument("--eval_batch_size", type=int, default=1)
    value.add_argument("--accumulation_steps", type=int, default=8)
    value.add_argument("--optimizer_steps", type=int, default=1500)
    value.add_argument("--eval_every", type=int, default=500)
    value.add_argument("--log_every", type=int, default=20)
    value.add_argument("--num_workers", type=int, default=8)
    value.add_argument("--learning_rate", type=float, default=1e-3)
    value.add_argument("--weight_decay", type=float, default=1e-4)
    value.add_argument("--focus_probability", type=float, default=0.5)
    value.add_argument("--proposal_threshold", type=float, default=0.5)
    value.add_argument("--minimum_area", type=int, default=16)
    value.add_argument("--minimum_building_fraction", type=float, default=0.5)
    value.add_argument("--minimum_class_purity", type=float, default=0.75)
    value.add_argument("--maximum_transition", type=float, default=0.25)
    value.add_argument("--val_max_batches", type=int, default=0)
    value.add_argument("--seed", type=int, default=3407)
    value.add_argument("--resume", default="")
    return value


def main():
    args = parser().parse_args()
    if args.variant == "q2_b0" and args.optimizer_steps != 0:
        raise ValueError("Q2-B0 is a zero-step guard")
    seed_everything(args.seed)
    device = torch.device(args.device)
    p3, q0, provenance = load_experts(args, device)
    model = Q2BuildingResidualFusion(
        p3, q0, args.variant,
        proposal_threshold=args.proposal_threshold,
        minimum_area=args.minimum_area,
        maximum_transition=args.maximum_transition,
    ).to(device)
    assert_q2_freeze_contract(model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_dataset = make_dataset(args, "val")
    val_loader = make_loader(val_dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    initial, base, initial_audit = evaluate_q2(
        model, val_loader, device, max_batches=args.val_max_batches
    )
    exact = all(
        initial[key] == base[key]
        for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou")
    ) and initial_audit["pixels"]["changed"] == 0
    if not exact:
        raise RuntimeError("Q2 zero-transition output is not exact P3")
    print(f"Q2 exact P3 guard passed | Valid={100*base['valid_miou']:.4f} "
          f"| Dmg={100*base['damaged_iou']:.4f}", flush=True)
    initial_record = {
        "protocol": "Q2_exact_P3_initial_validation_v1",
        "variant": args.variant,
        "step": 0,
        "metrics": initial,
        "base_metrics": base,
        "audit": initial_audit,
        "sources": provenance,
        "test_accessed": False,
    }
    atomic_json(initial_record, output_dir / "initial_validation.json")
    atomic_save(
        checkpoint_payload(BEST_PROTOCOL, model, args, provenance, 0, initial, base, initial_audit),
        output_dir / "best_val_miou.pth",
    )
    if args.optimizer_steps == 0:
        print("Q2-B0 exact P3 guard complete.")
        return

    examples = max(
        args.train_batch_size,
        args.optimizer_steps * args.train_batch_size * args.accumulation_steps,
    )
    train_dataset = make_dataset(args, "train", examples)
    train_loader = make_loader(train_dataset, args.train_batch_size, args.num_workers, True, args.seed)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    step = 0
    best_key = selection_key(initial)
    best_metrics = initial
    best_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("protocol") != RESUME_PROTOCOL or resume.get("variant") != args.variant:
            raise RuntimeError("Q2 resume protocol mismatch")
        for source_name, value in provenance.items():
            if resume.get("sources", {}).get(source_name, {}).get("sha256") != value["sha256"]:
                raise RuntimeError(f"Q2 resume {source_name} fingerprint mismatch")
        load_residual_state(model, resume["residual"])
        optimizer.load_state_dict(resume["optimizer"])
        step = int(resume["step"])
        best_key = tuple(resume["best_key"])
        best_metrics = resume["best_metrics"]
        best_step = int(resume.get("best_step", 0))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    micro_step = 0
    running_loss = 0.0
    running_regions = 0
    while step < args.optimizer_steps:
        try:
            rgb, sar, _, damage, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            rgb, sar, _, damage, _ = next(iterator)
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar, return_dict=True)
        loss, region_count = region_cross_entropy(
            outputs, damage,
            minimum_building_fraction=args.minimum_building_fraction,
            minimum_class_purity=args.minimum_class_purity,
        )
        if region_count == 0:
            continue
        (loss / args.accumulation_steps).backward()
        running_loss += float(loss.detach())
        running_regions += region_count
        micro_step += 1
        if micro_step % args.accumulation_steps:
            continue
        trainable = [value for value in model.parameters() if value.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            print(json.dumps({
                "step": step,
                "variant": args.variant,
                "loss": running_loss / args.log_every,
                "supervised_regions": running_regions,
                "transition": float(
                    args.maximum_transition * torch.tanh(model.transition_raw.detach())
                ),
            }), flush=True)
            running_loss = 0.0
            running_regions = 0
        if step % args.eval_every == 0 or step == args.optimizer_steps:
            metrics, base_metrics, audit = evaluate_q2(
                model, val_loader, device, max_batches=args.val_max_batches
            )
            gate = advancement_gate(metrics, base_metrics, audit)
            print(
                f"Q2 validation step={step} | Valid={100*metrics['valid_miou']:.4f} "
                f"| Dmg={100*metrics['damaged_iou']:.4f} "
                f"| Event={100*metrics['event_mean_official_style_4class_miou']:.4f}/"
                f"{100*metrics['event_worst_official_style_4class_miou']:.4f} "
                f"| corrected/harmed={audit['regions']['corrected']}/"
                f"{audit['regions']['harmed']} | gate={gate['passed']}", flush=True,
            )
            record = {
                "protocol": "Q2_building_residual_validation_v1",
                "variant": args.variant,
                "step": step,
                "metrics": metrics,
                "base_metrics": base_metrics,
                "audit": audit,
                "advancement_gate": gate,
                "test_accessed": False,
            }
            atomic_json(record, output_dir / "latest_validation.json")
            key = selection_key(metrics)
            if key > best_key:
                best_key = key
                best_metrics = metrics
                best_step = step
                atomic_save(
                    checkpoint_payload(
                        BEST_PROTOCOL, model, args, provenance, step,
                        metrics, base_metrics, audit,
                    ), output_dir / "best_val_miou.pth",
                )
                print("New best Q2 validation checkpoint.", flush=True)
            resume_payload = checkpoint_payload(
                RESUME_PROTOCOL, model, args, provenance, step,
                metrics, base_metrics, audit, optimizer,
            )
            resume_payload["best_key"] = list(best_key)
            resume_payload["best_metrics"] = best_metrics
            resume_payload["best_step"] = best_step
            atomic_save(resume_payload, output_dir / "latest_resume.pth")
            model.train()

    summary = {
        "protocol": "Q2_building_residual_training_summary_v1",
        "variant": args.variant,
        "completed_step": step,
        "selected_step": best_step,
        "best_metrics": best_metrics,
        "sources": provenance,
        "trainable_parameter_names": model.trainable_parameter_names(),
        "selection_policy": "event-worst -> event-mean -> Damaged IoU -> valid mIoU",
        "test_accessed": False,
    }
    atomic_json(summary, output_dir / "training_summary.json")
    print(f"Q2 training complete at step {step}; saved {output_dir}")


if __name__ == "__main__":
    main()
