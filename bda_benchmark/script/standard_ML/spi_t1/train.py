#!/usr/bin/env python3
"""Paired fixed-step T0/T1 continuation from the same locked P3 checkpoint.

This trainer intentionally never reads validation instances or the hidden test
split.  T0 and T1 share every operation except lambda_cb: T0 uses zero and T1
adds the class-component-balanced dense NLL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[3]
for path in (str(BDA_ROOT.parent), str(BDA_ROOT), str(BDA_ROOT / "script/standard_ML")):
    if path not in sys.path:
        sys.path.insert(0, path)

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    BuildingBagSpec,
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (  # noqa: E402
    checkpoint_state,
)
from bda_benchmark.script.standard_ML.spi_t1.core import (  # noqa: E402
    VARIANTS,
    SemanticComponent,
    class_component_partition,
    component_balanced_nll,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (  # noqa: E402
    build_dataset,
    parse_batch,
    seed_worker,
    segmentation_loss,
    set_random_seed,
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def select_bags(
    components: Sequence[SemanticComponent],
    *,
    minimum_area: int,
    maximum_per_class: int,
) -> list[BuildingBagSpec]:
    bags = [
        BuildingBagSpec(
            batch_index=component.batch_index,
            mask=component.mask,
            label=component.class_label,
            area=component.area,
        )
        for component in components
        if component.area >= minimum_area
    ]
    if maximum_per_class <= 0:
        return bags
    selected: list[BuildingBagSpec] = []
    for class_label in (1, 2, 3):
        class_bags = [bag for bag in bags if bag.label == class_label]
        if len(class_bags) > maximum_per_class:
            class_bags = random.sample(class_bags, maximum_per_class)
        selected.extend(class_bags)
    return selected


def p3_bag_losses(
    bag_result: Mapping[str, object] | None,
    reference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    zero = reference.float().sum() * 0.0
    result = {
        "mil": zero,
        "exist": zero,
        "severity": zero,
        "intact_guard": zero,
    }
    if not bag_result or int(bag_result.get("bag_count", 0)) == 0:
        return result
    labels = bag_result["labels"]
    if not isinstance(labels, torch.Tensor):
        raise TypeError("bag labels must be a tensor")
    labels = labels.long()
    log_probs3 = bag_result["log_probs3"]
    exist_logits = bag_result["exist_logits"]
    severity_logits = bag_result["severity_logits"]
    if not all(isinstance(value, torch.Tensor) for value in (log_probs3, exist_logits, severity_logits)):
        raise TypeError("bag outputs must be tensors")
    result["mil"] = F.nll_loss(log_probs3.float(), labels - 1)
    exist_target = (labels > 1).long()
    result["exist"] = F.cross_entropy(exist_logits.float(), exist_target)
    affected = labels > 1
    if bool(affected.any().item()):
        result["severity"] = F.cross_entropy(
            severity_logits[affected].float(), labels[affected] - 2
        )
    guard = bag_result.get("intact_guard_logits")
    if isinstance(guard, torch.Tensor) and guard.numel() > 0:
        result["intact_guard"] = F.binary_cross_entropy_with_logits(
            guard.float(), torch.zeros_like(guard, dtype=torch.float32)
        )
    return result


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("SPI paired T0/T1 fixed-step training")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_base_sha256", default="")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--optimizer_steps", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--minimum_component_area", type=int, default=32)
    parser.add_argument("--maximum_bags_per_class", type=int, default=16)
    parser.add_argument("--lambda_mil", type=float, default=0.25)
    parser.add_argument("--lambda_exist", type=float, default=0.25)
    parser.add_argument("--lambda_severity", type=float, default=0.25)
    parser.add_argument("--lambda_intact_guard", type=float, default=0.05)
    parser.add_argument("--lambda_cb", type=float, default=0.10)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args(argv)

    if args.optimizer_steps < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("optimizer_steps, batch_size, and accumulation_steps must be positive")
    if args.minimum_component_area < 1:
        raise ValueError("minimum_component_area must be positive")
    effective_lambda_cb = 0.0 if args.variant == VARIANTS[0] else float(args.lambda_cb)
    if args.variant == VARIANTS[1] and effective_lambda_cb <= 0.0:
        raise ValueError("T1 requires lambda_cb > 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    generator = set_random_seed(args.seed, 0, True)
    base_sha256 = file_sha256(args.base_checkpoint)
    if args.expected_base_sha256 and base_sha256 != args.expected_base_sha256:
        raise RuntimeError(
            f"base checkpoint SHA256 mismatch: expected={args.expected_base_sha256}, observed={base_sha256}"
        )

    model = DamageFormer_S8_MICB_DSMIL().to(device)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict base checkpoint load failed: {incompatible}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and args.amp_dtype == "fp16"
    )
    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    if not len(loader):
        raise RuntimeError("training loader is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["effective_lambda_cb"] = effective_lambda_cb
    config["base_checkpoint_sha256"] = base_sha256
    config["test_accessed"] = False
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=" * 96, flush=True)
    print(f"SPI T0/T1 paired training | variant={args.variant}", flush=True)
    print(f"base={args.base_checkpoint}", flush=True)
    print(f"seed={args.seed} steps={args.optimizer_steps} EBS={args.batch_size * args.accumulation_steps}", flush=True)
    print(f"lambda_cb={effective_lambda_cb} device={device} validation/test accessed=NEVER", flush=True)
    print("=" * 96, flush=True)

    location_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    damage_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    started = time.perf_counter()
    model.train()
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, location, damage, _ = parse_batch(batch, device)
            component_ids, components = class_component_partition(
                damage,
                minimum_area=args.minimum_component_area,
                ignore_index=args.ignore_index,
            )
            bags = select_bags(
                components,
                minimum_area=args.minimum_component_area,
                maximum_per_class=args.maximum_bags_per_class,
            )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output_loc, output_damage, auxiliary = model(
                    pre, post, bag_specs=bags, return_aux=True
                )
            loss_loc, _, _ = segmentation_loss(
                output_loc, location, location_ce, True, 1.0, 1.0, args.ignore_index
            )
            loss_damage, _, _ = segmentation_loss(
                output_damage, damage, damage_ce, True, 1.0, 1.0, args.ignore_index
            )
            bag_losses = p3_bag_losses(auxiliary["bag_result"], output_damage)
            loss_bag = (
                args.lambda_mil * bag_losses["mil"]
                + args.lambda_exist * bag_losses["exist"]
                + args.lambda_severity * bag_losses["severity"]
                + args.lambda_intact_guard * bag_losses["intact_guard"]
            )
            loss_cb, component_stats = component_balanced_nll(
                output_damage,
                damage,
                minimum_area=args.minimum_component_area,
                ignore_index=args.ignore_index,
                component_ids=component_ids,
            )
            total_loss = loss_loc + loss_damage + loss_bag + effective_lambda_cb * loss_cb
            if not bool(torch.isfinite(total_loss.detach()).item()):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            scaled = total_loss / float(args.accumulation_steps)
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                continue
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - started
                row = {
                    "step": step,
                    "variant": args.variant,
                    "total_loss": float(total_loss.detach().item()),
                    "location_loss": float(loss_loc.detach().item()),
                    "damage_loss": float(loss_damage.detach().item()),
                    "bag_loss": float(loss_bag.detach().item()),
                    "base_objective": float((loss_loc + loss_damage + loss_bag).detach().item()),
                    "component_balanced_loss": float(loss_cb.detach().item()),
                    "weighted_component_loss": float((effective_lambda_cb * loss_cb).detach().item()),
                    "weighted_component_to_base_ratio": float(
                        (effective_lambda_cb * loss_cb.detach() / (loss_loc + loss_damage + loss_bag).detach().clamp_min(1e-12)).item()
                    ),
                    "effective_lambda_cb": effective_lambda_cb,
                    **component_stats,
                    "elapsed_seconds": elapsed,
                }
                with (output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    f"{args.variant} step={step}/{args.optimizer_steps} "
                    f"loss={row['total_loss']:.4f} cb={row['component_balanced_loss']:.4f} "
                    f"components={row['component_count']} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if step % args.save_every == 0 or step == args.optimizer_steps:
                payload = {
                    "model": model.state_dict(),
                    "step": step,
                    "variant": args.variant,
                    "protocol": "SPI_T0_T1_FIXED_STEP_PAIRED_v1",
                    "base_checkpoint_sha256": base_sha256,
                    "seed": args.seed,
                    "effective_lambda_cb": effective_lambda_cb,
                    "args": config,
                    "test_accessed": False,
                    "resume_supported": False,
                }
                _atomic_save(payload, output_dir / "last.pth")
            if step >= args.optimizer_steps:
                break

    print(f"Saved: {output_dir / 'last.pth'}", flush=True)
    print("Validation accessed: False | Test accessed: False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
