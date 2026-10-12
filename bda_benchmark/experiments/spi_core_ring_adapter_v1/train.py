from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import checkpoint_state, file_sha256
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset,
    parse_batch,
    seed_worker,
    segmentation_loss,
    set_random_seed,
)

from . import VARIANTS
from .model import DamageReadoutControl, ParameterMatchedAdapter, adapter_inputs, trainable_parameter_count
from .regions import build_region_maps


PROTOCOL = "SPI_CORE_RING_ADAPTER_TRAIN_ONLY_STANDARD_ML_v1"


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_trainable(variant: str, p3: DamageFormer_S8_MICB_DSMIL, hidden: int):
    if variant == "m1_readout_finetune":
        return DamageReadoutControl(p3.clf_2)
    return ParameterMatchedAdapter(256, hidden, 4)


def forward_variant(variant, p3, trainable, pre, post, region_args):
    with torch.no_grad():
        _, base_output, auxiliary = p3(pre, post, return_aux=True)
        feature = auxiliary["fused_roi"].detach()
        base_low = p3.clf_2(feature)
    if variant == "m1_readout_finetune":
        new_low = trainable(feature)
        region_count = 0
    else:
        # M2/M3/M4 deliberately share the exact same LocCC support. Their only
        # difference is local, region-mean, or core/ring context construction.
        regions = build_region_maps(auxiliary["p_building"], **region_args)
        context, contrast, support = adapter_inputs(variant, feature, regions)
        new_low = base_low + trainable(feature, context, contrast, support)
        region_count = 0 if regions is None else regions.region_count
    output = F.interpolate(new_low, size=pre.shape[-2:], mode="bilinear", align_corners=False)
    return output, base_output.detach(), region_count


def safe_anchor_kl(output, base_output, target, confidence_threshold: float):
    with torch.no_grad():
        base_prob = F.softmax(base_output.float(), dim=1)
        confidence, base_prediction = base_prob.max(dim=1)
        safe = (target != 255) & (base_prediction == target) & (confidence >= confidence_threshold)
    log_prob = F.log_softmax(output.float(), dim=1)
    per_pixel = F.kl_div(log_prob, base_prob, reduction="none").sum(dim=1)
    safe_float = safe.to(per_pixel.dtype)
    count = safe_float.sum()
    return (per_pixel * safe_float).sum() / count.clamp_min(1.0), count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-feature-area", type=int, default=2)
    parser.add_argument("--core-radius", type=int, default=1)
    parser.add_argument("--ring-radius", type=int, default=2)
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--anchor-confidence", type=float, default=0.80)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="slower bitwise-oriented cuDNN mode")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # The default favours 4090 throughput. Scientific reproducibility is assessed
    # with locked seeds/checkpoint provenance and later multi-seed experiments.
    generator = set_random_seed(args.seed, 0, args.deterministic)
    base_sha = file_sha256(args.base_checkpoint)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval()
    p3.requires_grad_(False)

    trainable = make_trainable(args.variant, p3, args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pth"
    start_step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("variant") != args.variant or saved.get("base_checkpoint_sha256") != base_sha:
            raise RuntimeError("existing checkpoint provenance does not match this run")
        start_step = int(saved["optimizer_step"])
        trainable.load_state_dict(saved["trainable_state"], strict=True)
        if start_step < args.optimizer_steps and "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])
        if start_step >= args.optimizer_steps:
            print(f"{args.variant}: already complete at step={start_step}", flush=True)
            return 0

    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker, generator=generator,
    )
    if not len(loader):
        raise RuntimeError("empty training loader")
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)
    amp_enabled = device.type == "cuda"
    region_args = dict(
        threshold=args.building_threshold,
        minimum_area=args.minimum_feature_area,
        core_radius=args.core_radius,
        ring_radius=args.ring_radius,
    )
    config = vars(args).copy()
    config.update(protocol=PROTOCOL, base_checkpoint_sha256=base_sha, official_instances_accessed=False, validation_accessed=False, test_accessed=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("=" * 100, flush=True)
    print(f"Core/Ring adapter | {args.variant} | frozen P3 | trainable={trainable_parameter_count(trainable):,}", flush=True)
    print(f"steps={args.optimizer_steps} EBS={args.batch_size * args.accumulation_steps} BF16={amp_enabled}", flush=True)
    print("validation/official instances/test accessed: NEVER / NEVER / NEVER", flush=True)
    print("=" * 100, flush=True)

    trainable.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step, micro_step = start_step, 0
    started = time.perf_counter()
    data_tick = time.perf_counter()
    while optimizer_step < args.optimizer_steps:
        for batch in loader:
            data_seconds = time.perf_counter() - data_tick
            pre, post, _, damage, _ = parse_batch(batch, device)
            if args.profile_timing and device.type == "cuda": torch.cuda.synchronize()
            forward_tick = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output, base_output, region_count = forward_variant(
                    args.variant, p3, trainable, pre, post, region_args
                )
            semantic_loss, ce, lovasz = segmentation_loss(output, damage, ce_fn, True, 1.0, 1.0, 255)
            anchor, safe_pixels = safe_anchor_kl(output, base_output, damage, args.anchor_confidence)
            total_loss = semantic_loss + args.lambda_anchor * anchor
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at step {optimizer_step}")
            (total_loss / args.accumulation_steps).backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                data_tick = time.perf_counter()
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            optimizer_step += 1
            if args.profile_timing and device.type == "cuda": torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_tick

            if optimizer_step == 1 or optimizer_step % args.log_every == 0:
                row = dict(
                    optimizer_step=optimizer_step, variant=args.variant,
                    loss=float(total_loss.detach()), semantic_loss=float(semantic_loss.detach()),
                    ce=float(ce.detach()), lovasz=float(lovasz.detach()), anchor=float(anchor.detach()),
                    safe_pixels=int(safe_pixels.detach()), regions=region_count,
                    gradient_norm_before_clip=float(gradient_norm), data_seconds=data_seconds,
                    forward_backward_seconds=forward_seconds,
                    elapsed_seconds=time.perf_counter() - started,
                )
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if optimizer_step % args.save_every == 0 or optimizer_step == args.optimizer_steps:
                atomic_save(dict(
                    protocol=PROTOCOL, variant=args.variant, optimizer_step=optimizer_step,
                    seed=args.seed, base_checkpoint_sha256=base_sha,
                    trainable_state=trainable.state_dict(), optimizer_state=optimizer.state_dict(),
                    args=config, validation_accessed=False, official_instances_accessed=False,
                    test_accessed=False,
                ), checkpoint_path)
            if optimizer_step >= args.optimizer_steps:
                break
            data_tick = time.perf_counter()
    print(f"Saved: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
