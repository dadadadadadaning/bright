from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import build_region_maps
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import checkpoint_state, file_sha256
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset,
    parse_batch,
    seed_worker,
    segmentation_loss,
    set_random_seed,
)

from . import PROTOCOL, TRAINABLE_VARIANTS
from .core import (
    DistillationConfig,
    StructuralResidualStudent,
    build_structural_batch,
    outside_anchor_kl,
    weighted_forward_kl,
)


def _make_modules(hidden: int) -> nn.ModuleDict:
    # K2/K3 start from exactly the same parameters, so their only causal
    # difference is the S2 distillation term.
    local = StructuralResidualStudent(hidden=hidden, block="local")
    k2 = copy.deepcopy(local)
    k3 = copy.deepcopy(local)
    k4 = StructuralResidualStudent(hidden=hidden, block="gbc")
    k5 = StructuralResidualStudent(hidden=hidden, block="gbc", use_variance=True)
    common = {key: value for key, value in k4.state_dict().items() if key in k5.state_dict()}
    k5.load_state_dict(common, strict=False)
    return nn.ModuleDict({
        "k2_semantic_local_control": k2,
        "k3_s2_distilled_local": k3,
        "k4_s2_distilled_gbc": k4,
        "k5_s2_distilled_gbc_variance": k5,
    })


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Paired no-Oracle S2 structural distillation")
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
    parser.add_argument("--lambda-s2", type=float, default=0.50)
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-feature-area", type=int, default=2)
    parser.add_argument("--minimum-purity", type=float, default=0.70)
    parser.add_argument("--minimum-margin", type=float, default=0.10)
    parser.add_argument("--blend-strength", type=float, default=0.75)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    if args.lambda_s2 < 0 or args.lambda_anchor < 0:
        raise ValueError("loss weights must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    generator = set_random_seed(args.seed, 0, args.deterministic)
    base_sha = file_sha256(args.base_checkpoint)
    saved_base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(saved_base), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)

    modules = _make_modules(args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(modules.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "joint_last.pth"
    optimizer_step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("base_checkpoint_sha256") != base_sha or tuple(saved.get("variants", ())) != TRAINABLE_VARIANTS:
            raise RuntimeError("existing checkpoint provenance mismatch")
        optimizer_step = int(saved["optimizer_step"])
        modules.load_state_dict(saved["trainable_states"], strict=True)
        if optimizer_step < args.optimizer_steps:
            optimizer.load_state_dict(saved["optimizer_state"])
        else:
            print(f"S2 distillation already complete at step={optimizer_step}", flush=True)
            return 0

    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if len(loader) == 0:
        raise RuntimeError("empty training loader")
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)
    config = DistillationConfig(args.minimum_purity, args.minimum_margin, args.blend_strength)
    run_config = vars(args).copy()
    run_config.update({
        "protocol": PROTOCOL,
        "base_checkpoint_sha256": base_sha,
        "official_instances_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "oracle_used": False,
    })
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print("=" * 110)
    print("S2 structural distillation | paired K2-K5 | one frozen-P3 forward/batch | no Oracle")
    print(f"steps={optimizer_step}->{args.optimizer_steps} EBS={args.batch_size * args.accumulation_steps} device={device}")
    print("validation/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 110, flush=True)

    modules.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.perf_counter()
    data_tick = time.perf_counter()
    while optimizer_step < args.optimizer_steps:
        for batch in loader:
            data_seconds = time.perf_counter() - data_tick
            pre, post, _, damage, _ = parse_batch(batch, device)
            if args.profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            phase = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    _, _, auxiliary = p3(pre, post, return_aux=True)
                    feature = auxiliary["fused_roi"].detach()
                    base_low = p3.clf_2(feature).detach()
                    regions = build_region_maps(
                        auxiliary["p_building"],
                        threshold=args.building_threshold,
                        minimum_area=args.minimum_feature_area,
                        core_radius=0,
                        ring_radius=0,
                    )
                    structural = build_structural_batch(base_low, feature, regions, config)
                    base_probability = F.softmax(base_low.float(), dim=1).to(feature.dtype)

                low_outputs = {
                    name: base_low + module(feature, base_probability, structural)
                    for name, module in modules.items()
                }
                full_outputs = {
                    name: F.interpolate(low, size=damage.shape[-2:], mode="bilinear", align_corners=False)
                    for name, low in low_outputs.items()
                }

            total_loss = feature.sum() * 0.0
            rows = {}
            for name in TRAINABLE_VARIANTS:
                semantic, ce, lovasz = segmentation_loss(
                    full_outputs[name], damage, ce_fn, True, 1.0, 1.0, 255
                )
                anchor = outside_anchor_kl(low_outputs[name], base_low, structural.accepted_support)
                distill = weighted_forward_kl(
                    low_outputs[name], structural.teacher_probability, structural.distillation_weight
                )
                distill_weight = 0.0 if name == "k2_semantic_local_control" else args.lambda_s2
                loss = semantic + args.lambda_anchor * anchor + distill_weight * distill
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss for {name} at step={optimizer_step}")
                total_loss = total_loss + loss
                rows[name] = {
                    "loss": loss.detach(), "semantic": semantic.detach(), "ce": ce.detach(),
                    "lovasz": lovasz.detach(), "s2_distill": distill.detach(), "anchor": anchor.detach(),
                }
            (total_loss / args.accumulation_steps).backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                data_tick = time.perf_counter()
                continue
            norms = {
                name: torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
                for name, module in modules.items()
            }
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            optimizer_step += 1
            if args.profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - phase

            if optimizer_step == 1 or optimizer_step % args.log_every == 0:
                record = {
                    "optimizer_step": optimizer_step,
                    "losses": {
                        name: {key: float(value) for key, value in row.items()}
                        for name, row in rows.items()
                    },
                    "gradient_norm_before_clip": {name: float(value) for name, value in norms.items()},
                    "regions": structural.total_regions,
                    "accepted_regions": structural.accepted_regions,
                    "data_seconds": data_seconds,
                    "forward_backward_seconds": step_seconds,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if optimizer_step % args.save_every == 0 or optimizer_step == args.optimizer_steps:
                _atomic_save({
                    "protocol": PROTOCOL,
                    "variants": TRAINABLE_VARIANTS,
                    "optimizer_step": optimizer_step,
                    "seed": args.seed,
                    "base_checkpoint_sha256": base_sha,
                    "trainable_states": modules.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": run_config,
                    "validation_accessed": False,
                    "official_instances_accessed": False,
                    "test_accessed": False,
                    "oracle_used": False,
                }, checkpoint_path)
            if optimizer_step >= args.optimizer_steps:
                break
            data_tick = time.perf_counter()
    print(f"Saved: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
