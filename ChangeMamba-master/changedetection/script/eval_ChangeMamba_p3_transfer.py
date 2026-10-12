#!/usr/bin/env python3
"""Independent val/test evaluator for locked CM-P2/CM-P3 weights."""
import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from changedetection.checkpoints import load_model_weights
from changedetection.configs.config import get_config
from changedetection.experiments.native_sar.dataset import (
    build_native_sar_loader,
    load_sar_stats,
)
from changedetection.experiments.native_sar.trainer import evaluate_native_sar_model
from changedetection.experiments.p3_transfer.model import (
    ChangeMambaP3Transfer,
    VALID_P3_TRANSFER_VARIANTS,
)
from changedetection.script.script_utils import get_vssm_kwargs


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser("Evaluate ChangeMamba P3 transfer")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument("--variant", choices=VALID_P3_TRANSFER_VARIANTS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--roi_floor", type=float, default=0.10)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--allow_test", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Test is locked; pass --allow_test only for the final run")
    args.dataset = "BRIGHT"
    stats = load_sar_stats(args.sar_stats_path)
    config = get_config(args)
    model = ChangeMambaP3Transfer(
        output_building=2,
        output_damage=4,
        pretrained=None,
        sar_mean=stats["mean"],
        sar_std=stats["std"],
        variant=args.variant,
        roi_floor=args.roi_floor,
        **get_vssm_kwargs(config),
    )
    report = load_model_weights(model, args.checkpoint)
    if report["missing_keys"] or report["unexpected_keys"] or report["mismatched_keys"]:
        raise RuntimeError(f"strict candidate checkpoint load failed: {report}")
    names = [
        line.strip()
        for line in open(args.data_list_path, encoding="utf-8")
        if line.strip()
    ]
    loader = build_native_sar_loader(
        dataset_path=args.dataset_path,
        data_list=names,
        crop_size=args.crop_size,
        sar_stats_path=args.sar_stats_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate_native_sar_model(model, loader, device)
    metrics["building_transition"] = float(
        torch.tanh(model.building_transition.detach()).cpu()
    )
    payload = {
        "protocol": "ChangeMamba_P3_transfer_independent_eval_v1",
        "variant": args.variant,
        "split": args.split,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "processed_images": len(names),
        "metrics": metrics,
        "test_accessed": args.split == "test",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
