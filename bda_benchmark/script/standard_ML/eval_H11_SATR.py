# -*- coding: utf-8 -*-
"""Strict independent validation/test evaluator for H11-SATR."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H11_SATR import (  # noqa: E402
    VALID_H11_VARIANTS,
    build_h11_satr_damageformer,
)
from bda_benchmark.script.standard_ML.h10_h3d_runtime import (  # noqa: E402
    evaluate_h10,
    metrics_line,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


VALID_PROTOCOLS = ("H11_SATR_best_weights_v1", "H11_SATR_resume_v1")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate H11-SATR")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol not in VALID_PROTOCOLS:
        raise RuntimeError(f"unexpected H11 checkpoint protocol: {protocol!r}")
    if variant not in VALID_H11_VARIANTS:
        raise RuntimeError(f"unexpected H11 checkpoint variant: {variant!r}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("H11 checkpoint has no model state dictionary")

    model = build_h11_satr_damageformer(variant, pretrained_rgb=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H11 load unexpectedly failed: {incompatible}")
    model.to(device).eval()

    dataset_variant = "h4" if variant in ("t1e", "t2e") else "h3"
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split=args.split,
        variant=dataset_variant,
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    details: Dict[str, object] = {}
    observed = evaluate_h10(
        model,
        loader,
        device,
        args.ignore_index,
        max_batches=0,
        details_sink=details,
    )
    stored = checkpoint.get("metrics", checkpoint.get("best_metrics", {}))
    deltas: Dict[str, float] = {}
    if args.split == "val" and isinstance(stored, Mapping):
        for key, value in observed.items():
            if key in stored and isinstance(stored[key], (int, float)):
                deltas[key] = float(value) - float(stored[key])

    transitions = {}
    for stage in ("stage3", "stage4"):
        key = f"sar_encoder.transfer_transition.{stage}"
        if key in state:
            transitions[stage] = float(torch.tanh(state[key]).item())
    payload = {
        "protocol": "H11_SATR_independent_evaluation_v1",
        "split": args.split,
        "test_accessed": args.split == "test",
        "processed_images": int(observed.get("processed_images", 0)),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256(args.checkpoint),
            "protocol": protocol,
            "variant": variant,
            "step": int(checkpoint.get("step", 0)),
            "selector": checkpoint.get("selector"),
            "strict_load": True,
        },
        "effective_transfer_transitions": transitions,
        "stored_metrics": dict(stored) if isinstance(stored, Mapping) else {},
        "observed_metrics": observed,
        "observed_minus_stored": deltas,
        **details,
        "data": {
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "list_path": str(Path(args.data_list_path).resolve()),
            "sar_stats_path": str(Path(args.sar_stats_path).resolve()),
            "dataset_variant": dataset_variant,
            "sarmae_input_contract": (
                "official_3ch_imagenet_photometric"
                if dataset_variant == "h4"
                else "naive_1ch_train_stats"
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(metrics_line(f"Independent {args.split}", observed), flush=True)
    print(f"Transfer transitions: {transitions}", flush=True)
    print("Strict checkpoint load: True", flush=True)
    print(f"Processed images: {payload['processed_images']}", flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
