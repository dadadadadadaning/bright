#!/usr/bin/env python3
"""Validation-only audit for P3 + zero-gated P5-C2 residual fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BDA_ROOT = REPO_ROOT / "bda_benchmark"
for value in (str(REPO_ROOT), str(BDA_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from bda_benchmark.dataset.make_data_loader_p4_native_sar import build_p4_dataset
from bda_benchmark.experiments.p5_ablation_validation_v1.audit_model import (
    fast_hist,
    perturb_sar,
)
from bda_benchmark.experiments.p5_ablation_validation_v1.metrics import (
    aggregate_by_event,
    event_from_name,
    metrics_from_confusion,
)
from bda_benchmark.model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from bda_benchmark.model.DamageFormer_S8_P5_C2_Recovery import (
    C2RecoveryHeads,
    get_c2_recovery_spec,
)
from bda_benchmark.model.DamageFormer_S8_P5_DCMF import DamageFormer_S8_P5_DCMF


MODEL_KWARGS = dict(
    roi_alpha=0.1,
    detach_building_prior=True,
    token_dim=128,
    mil_topk=8,
    max_tokens_per_bag=256,
    mil_residual_scale=0.0,
    use_asymmetric_regions=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit P3-anchored P5 residual")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--perturbation",
        choices=(
            "none",
            "shift_x4",
            "shift_y4",
            "shift_x8",
            "shift_y8",
            "pair_shuffle",
        ),
        default="none",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def _state(checkpoint: Mapping[str, object], key: str) -> Mapping[str, torch.Tensor]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"checkpoint is missing mapping {key!r}")
    return {str(name).removeprefix("module."): tensor for name, tensor in value.items()}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    variant = str(checkpoint.get("variant", "")).lower()
    recovery_variant = str(checkpoint.get("recovery_variant", "")).lower()
    if variant != "c2" or recovery_variant != "r1":
        raise RuntimeError(
            f"expected c2/r1 checkpoint, got {variant or '<missing>'}/"
            f"{recovery_variant or '<missing>'}"
        )

    device = torch.device(args.device)
    student = DamageFormer_S8_P5_DCMF(variant="c2", **MODEL_KWARGS)
    student.load_state_dict(_state(checkpoint, "model"), strict=True)
    heads = C2RecoveryHeads()
    heads.load_state_dict(_state(checkpoint, "recovery_heads"), strict=True)
    teacher = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0", **MODEL_KWARGS)
    load_p3_checkpoint(teacher, args.base_ckpt)
    student.to(device).eval()
    heads.to(device).eval()
    teacher.to(device).eval()
    spec = get_c2_recovery_spec("r1")

    dataset = build_p4_dataset(
        args.dataset_path,
        args.data_list_path,
        1024,
        "val",
        "a2",
        args.sar_stats_json,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows = []
    total_hist = torch.zeros(4, 4, dtype=torch.int64)
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        pre = batch["pre_img"].to(device).float()
        native_sar = batch["post_img"].to(device).float()
        p3_sar = batch["teacher_post_img"].to(device).float()
        target = batch["label_dam"].to(device).long()
        # Apply the same geometric/wrong-pair perturbation to both SAR views.
        native_sar = perturb_sar(native_sar, args.perturbation)
        p3_sar = perturb_sar(p3_sar, args.perturbation)
        _, c2_damage = student(pre, native_sar)
        p3_location, p3_damage = teacher(pre, p3_sar)
        _, damage = heads.anchor_outputs(
            p3_location,
            c2_damage,
            p3_location,
            p3_damage,
            enabled=spec.p3_anchor,
        )
        prediction = damage.argmax(1)
        for sample_index, name in enumerate(batch["name"]):
            hist = fast_hist(
                prediction[sample_index],
                target[sample_index],
                ignore_index=args.ignore_index,
            ).cpu()
            total_hist += hist
            rows.append(
                {
                    "name": str(name),
                    "event": event_from_name(str(name)),
                    "confusion": hist.tolist(),
                }
            )

    payload = {
        "protocol": "P3_P5_SAFE_RESIDUAL_AUDIT_v1",
        "architecture": "p3_plus_p5_c2_zero_gated_residual",
        "variant": "c2_r1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": "val",
        "test_accessed": False,
        "perturbation": args.perturbation,
        "images": len(rows),
        "confusion": total_hist.tolist(),
        "metrics": metrics_from_confusion(total_hist.numpy()),
        "events": aggregate_by_event(rows),
        "image_rows": rows,
        "recovery_gate": float(heads.gate_value.detach().cpu().item()),
        "p3_anchor": True,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    compact = {key: value for key, value in payload.items() if key not in ("events", "image_rows")}
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
