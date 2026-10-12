#!/usr/bin/env python3
"""Independent Validation evaluation for CM0/CM1; Test is not accepted."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from changedetection.checkpoints import load_model_weights
from changedetection.configs.config import get_config
from changedetection.datasets import build_eval_loader
from changedetection.experiments.building_prior_ablation_v1.model import VALID_VARIANTS, build_model
from changedetection.script.script_utils import get_vssm_kwargs, read_name_list
from changedetection.tasks.bright import BRIGHTTrainer


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser("Independent CM0/CM1 Validation evaluation")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument("--variant", choices=VALID_VARIANTS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dataset", default="BRIGHT")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--roi_floor", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.encoder_pretrained_path = None
    args.model_checkpoint_path = None
    args.resume_training_path = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config(args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = build_model(
        args.variant,
        pretrained=None,
        roi_floor=args.roi_floor,
        **get_vssm_kwargs(config),
    ).to(device)
    report = load_model_weights(model, args.checkpoint)
    model.eval()

    loader = build_eval_loader(
        args.dataset,
        dataset_path=args.dataset_path,
        data_list=read_name_list(args.data_list_path),
        crop_size=args.crop_size,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Reuse the official metric implementation without constructing a trainer.
    metrics = BRIGHTTrainer.evaluate_loader(
        type("EvaluationProxy", (), {"model": model, "device": device})(),
        "Validation",
        loader,
    )
    transition = getattr(model, "building_transition", None)
    metrics["building_transition"] = (
        None if transition is None else float(torch.tanh(transition.detach()).cpu())
    )
    payload = {
        "protocol": "CHANGEMAMBA_BUILDING_PRIOR_ABLATION_V1",
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "load_report": str(report),
        "validation": metrics,
        "test_accessed": False,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
