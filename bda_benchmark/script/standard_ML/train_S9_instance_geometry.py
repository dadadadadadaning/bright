#!/usr/bin/env python3
"""Unified frozen-S8 trainer for S9-Ic G1/G2/G3 and S9-Id."""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"
for import_root in (PROJECT_ROOT, BDA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (
    build_training_bag_specs,
)
from bda_benchmark.model.instance_geometry.config import geometry_preset
from bda_benchmark.model.instance_geometry.losses import reliable_geometry_loss
from bda_benchmark.model.instance_geometry.metrics import (
    CenterMetricAccumulator,
    evaluate_center_mechanism,
)
from bda_benchmark.model.instance_geometry.s8_backbone import FrozenS8Backbone
from bda_benchmark.model.instance_geometry.targets import (
    build_center_offset_targets,
    build_unique_gaussian_targets,
)
from bda_benchmark.script.standard_ML.s9ic_shared import (
    EXPECTED_FEATURE_CHANNELS,
    EXPECTED_FEATURE_KEY,
    GAUSSIAN_SIGMA_RULE,
    TARGET_DEFINITION,
    build_geometry_head,
    checkpoint_metadata,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_names(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise ValueError(f"data list is empty: {path}")
    return names


def _find_key(value: Any, wanted: str) -> Optional[Any]:
    if isinstance(value, Mapping):
        if wanted in value:
            return value[wanted]
        for nested in value.values():
            result = _find_key(nested, wanted)
            if result is not None:
                return result
    elif isinstance(value, list):
        for nested in value:
            result = _find_key(nested, wanted)
            if result is not None:
                return result
    return None


def _load_protocol_json(path: Optional[str]) -> Mapping[str, Any]:
    if not path:
        return {}
    protocol_path = Path(path).expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol JSON not found: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("protocol JSON root must be an object")
    return payload


def _resolve_data_argument(
    explicit: Optional[str],
    protocol: Mapping[str, Any],
    key: str,
) -> str:
    value = explicit or _find_key(protocol, key)
    if not value:
        raise ValueError(
            f"{key} is required. Pass --{key} or include it in --s9ib_json."
        )
    return str(value)


def _amp_context(device: torch.device, mode: str):
    key = mode.lower()
    if key == "none":
        return nullcontext()
    dtype = torch.bfloat16 if key == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_loader(
    dataset_path: str,
    list_path: str,
    *,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    max_samples: Optional[int] = None,
) -> DataLoader:
    # Keep CLI/config inspection usable even in a lightweight environment;
    # imageio and other dataset dependencies are needed only for real loading.
    from bda_benchmark.dataset.make_data_loader import (
        MultimodalDamageAssessmentDatset,
    )

    dataset = MultimodalDamageAssessmentDatset(
        dataset_path,
        _read_names(list_path),
        crop_size=crop_size if train else 1024,
        max_iters=max_samples,
        type="train" if train else "test",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _targets_for_variant(
    variant: str,
    labels_loc: torch.Tensor,
    labels_damage: torch.Tensor,
    output_size: Tuple[int, int],
):
    builder = (
        build_center_offset_targets
        if variant.lower() in {"id", "s9-id", "center-offset"}
        else build_unique_gaussian_targets
    )
    return builder(
        labels_loc,
        labels_damage,
        build_training_bag_specs,
        output_size=output_size,
    )


@torch.no_grad()
def evaluate_center_loader(
    backbone: FrozenS8Backbone,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    accumulator = CenterMetricAccumulator()
    head.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pre, post, labels_loc, labels_damage, _ = batch
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        frozen = backbone(pre, post)
        predictions = head(frozen.feature)
        center_probability = torch.sigmoid(predictions["center_logits"])
        specs = build_training_bag_specs(
            labels_loc.long(),
            labels_damage.long(),
            min_area=32,
        )
        for item_index in range(pre.shape[0]):
            masks = [
                spec.mask
                for spec in specs
                if int(spec.batch_index) == item_index
            ]
            report = evaluate_center_mechanism(
                center_probability[item_index, 0],
                masks,
            )
            accumulator.update(report)
    head.train()
    return accumulator.compute()


def save_checkpoint(
    path: Path,
    *,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    variant: str,
    decoder_config: Mapping[str, Any],
    head_config: Mapping[str, Any],
    center_metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "metadata": dict(checkpoint_metadata(variant, step)),
        "decoder_config": dict(decoder_config),
        "head_config": dict(head_config),
        "center_metrics": dict(center_metrics or {}),
    }
    torch.save(payload, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train S9-Ic G1/G2/G3 or S9-Id over a strictly frozen S8"
    )
    parser.add_argument("--variant", default="g1", choices=("g1", "g2", "g3", "id"))
    parser.add_argument("--checkpoint", "--s8_checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--s9ib_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_dataset_path")
    parser.add_argument("--train_data_list_path")
    parser.add_argument("--val_dataset_path")
    parser.add_argument("--val_data_list_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=3000)
    parser.add_argument("--max_iters", type=int, default=48000)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--center_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=1.0)
    parser.add_argument("--offset_weight", type=float, default=1.0)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--allow_unsafe_checkpoint", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.accumulation_steps < 1 or args.optimizer_steps < 1:
        raise ValueError("accumulation_steps and optimizer_steps must be positive")
    _set_seed(args.seed)
    protocol = _load_protocol_json(args.s9ib_json)
    train_dataset_path = _resolve_data_argument(
        args.train_dataset_path, protocol, "train_dataset_path"
    )
    train_data_list_path = _resolve_data_argument(
        args.train_data_list_path, protocol, "train_data_list_path"
    )

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone = FrozenS8Backbone.from_checkpoint(
        args.checkpoint,
        feature_key=EXPECTED_FEATURE_KEY,
        expected_channels=EXPECTED_FEATURE_CHANNELS,
        allow_unsafe=args.allow_unsafe_checkpoint,
    ).to(device)
    head = build_geometry_head(
        args.variant,
        hidden_channels=args.hidden_channels,
    ).to(device)
    backbone.assert_frozen()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler_enabled = args.amp == "fp16" and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    train_loader = _make_loader(
        train_dataset_path,
        train_data_list_path,
        crop_size=args.crop_size,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        train=True,
        max_samples=args.max_iters,
    )

    val_loader = None
    val_dataset_path = args.val_dataset_path or _find_key(protocol, "val_dataset_path")
    val_list_path = args.val_data_list_path or _find_key(
        protocol, "val_data_list_path"
    )
    if val_dataset_path and val_list_path:
        val_loader = _make_loader(
            str(val_dataset_path),
            str(val_list_path),
            crop_size=args.crop_size,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            train=False,
        )

    decoder = geometry_preset(args.variant)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "arguments": vars(args),
                "decoder": decoder.to_dict(),
                "feature_key": EXPECTED_FEATURE_KEY,
                "feature_channels": EXPECTED_FEATURE_CHANNELS,
                "target_definition": TARGET_DEFINITION,
                "gaussian_sigma_rule": GAUSSIAN_SIGMA_RULE,
                "s8_strict_loaded": True,
                "s8_frozen": True,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    head.train()
    for batch in train_loader:
        pre, post, labels_loc, labels_damage, _ = batch
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc = labels_loc.to(device, non_blocking=True).long()
        labels_damage = labels_damage.to(device, non_blocking=True).long()

        frozen = backbone(pre, post)
        with _amp_context(device, args.amp):
            predictions = head(frozen.feature)
            targets = _targets_for_variant(
                args.variant,
                labels_loc,
                labels_damage,
                frozen.feature.shape[-2:],
            )
            losses = reliable_geometry_loss(
                predictions["center_logits"],
                predictions["boundary_logits"],
                targets,
                predictions.get("offset_logits"),
                center_weight=args.center_weight,
                boundary_weight=args.boundary_weight,
                offset_weight=args.offset_weight,
            )
            scaled_loss = losses["loss"] / args.accumulation_steps
        if not bool(torch.isfinite(losses["loss"]).all().item()):
            values = {
                key: float(value.detach().float().item())
                for key, value in losses.items()
            }
            raise FloatingPointError(
                "non-finite geometry loss detected before backward: "
                f"{values}"
            )
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        if micro_step % args.accumulation_steps:
            continue
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        if optimizer_step % args.log_every == 0:
            values = {
                key: float(value.detach().float().item())
                for key, value in losses.items()
            }
            print(
                json.dumps(
                    {"step": optimizer_step, **values},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        should_save = (
            optimizer_step % args.eval_every == 0
            or optimizer_step >= args.optimizer_steps
        )
        if should_save:
            center_metrics = (
                evaluate_center_loader(
                    backbone,
                    head,
                    val_loader,
                    device,
                    max_batches=args.eval_max_batches,
                )
                if val_loader is not None
                else {}
            )
            checkpoint_path = output_dir / f"checkpoint_step_{optimizer_step}.pth"
            save_checkpoint(
                checkpoint_path,
                head=head,
                optimizer=optimizer,
                step=optimizer_step,
                variant=args.variant,
                decoder_config=decoder.to_dict(),
                head_config={
                    "in_channels": EXPECTED_FEATURE_CHANNELS,
                    "hidden_channels": args.hidden_channels,
                },
                center_metrics=center_metrics,
            )
            with (
                output_dir / f"center_metrics_step_{optimizer_step}.json"
            ).open("w", encoding="utf-8") as handle:
                json.dump(center_metrics, handle, indent=2, ensure_ascii=False)
            print(f"saved: {checkpoint_path}", flush=True)
        if optimizer_step >= args.optimizer_steps:
            break
    if optimizer_step < args.optimizer_steps:
        raise RuntimeError(
            f"training data exhausted at optimizer step {optimizer_step}; "
            f"expected {args.optimizer_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
