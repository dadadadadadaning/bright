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
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, parse_batch, seed_worker, segmentation_loss, set_random_seed

from . import VARIANTS
from .model import adapter_inputs
from .regions import build_region_maps
from .train import PROTOCOL, make_trainable, safe_anchor_kl


JOINT_PROTOCOL = PROTOCOL + "_PAIRED_ONE_P3_FORWARD"


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser("Paired M1-M4 training with one frozen-P3 forward per batch")
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
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--anchor-confidence", type=float, default=0.80)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-feature-area", type=int, default=2)
    parser.add_argument("--core-radius", type=int, default=1)
    parser.add_argument("--ring-radius", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    generator = set_random_seed(args.seed, 0, args.deterministic)
    base_sha = file_sha256(args.base_checkpoint)
    source = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(source), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)
    modules = nn.ModuleDict({variant: make_trainable(variant, p3, args.hidden_channels) for variant in VARIANTS}).to(device)
    optimizer = torch.optim.AdamW(modules.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "joint_last.pth"
    step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("base_checkpoint_sha256") != base_sha or tuple(saved.get("variants", ())) != VARIANTS:
            raise RuntimeError("joint checkpoint provenance mismatch")
        step = int(saved["optimizer_step"])
        modules.load_state_dict(saved["trainable_states"], strict=True)
        if step < args.optimizer_steps:
            optimizer.load_state_dict(saved["optimizer_state"])
        if step >= args.optimizer_steps:
            print(f"paired M1-M4 already complete at step={step}", flush=True)
            return 0

    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker, generator=generator)
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)
    region_args = dict(threshold=args.building_threshold, minimum_area=args.minimum_feature_area,
                       core_radius=args.core_radius, ring_radius=args.ring_radius)
    config = vars(args).copy()
    config.update(protocol=JOINT_PROTOCOL, base_checkpoint_sha256=base_sha,
                  validation_accessed=False, official_instances_accessed=False, test_accessed=False)
    (output_dir / "joint_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("=" * 104)
    print("Paired M1-M4 | one frozen-P3 forward/batch | independent parameters | semantic TIFF train only")
    print(f"step={step}->{args.optimizer_steps} EBS={args.batch_size*args.accumulation_steps} device={device}")
    print("validation/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 104, flush=True)

    modules.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.perf_counter()
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, _, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    _, base_output, auxiliary = p3(pre, post, return_aux=True)
                    feature = auxiliary["fused_roi"].detach()
                    base_low = p3.clf_2(feature)
                regions = build_region_maps(auxiliary["p_building"], **region_args)
                outputs = {
                    VARIANTS[0]: F.interpolate(modules[VARIANTS[0]](feature), size=pre.shape[-2:], mode="bilinear", align_corners=False)
                }
                for variant in VARIANTS[1:]:
                    context, contrast, support = adapter_inputs(variant, feature, regions)
                    outputs[variant] = F.interpolate(
                        base_low + modules[variant](feature, context, contrast, support),
                        size=pre.shape[-2:], mode="bilinear", align_corners=False,
                    )
            losses, rows = {}, {}
            total = feature.sum() * 0.0
            for variant, output in outputs.items():
                semantic, ce, lovasz = segmentation_loss(output, damage, ce_fn, True, 1.0, 1.0, 255)
                anchor, safe_pixels = safe_anchor_kl(output, base_output, damage, args.anchor_confidence)
                loss = semantic + args.lambda_anchor * anchor
                losses[variant] = loss
                total = total + loss
                rows[variant] = {"loss": loss.detach(), "semantic": semantic.detach(),
                                 "ce": ce.detach(), "lovasz": lovasz.detach(),
                                 "anchor": anchor.detach(), "safe_pixels": safe_pixels.detach()}
            (total / args.accumulation_steps).backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                continue
            # Clip each disjoint branch independently so paired execution is
            # numerically equivalent to four separate training jobs.
            grad_norm_tensors = {
                variant: torch.nn.utils.clip_grad_norm_(modules[variant].parameters(), 5.0)
                for variant in VARIANTS
            }
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            step += 1
            if step == 1 or step % args.log_every == 0:
                grad_norms = {name: float(value) for name, value in grad_norm_tensors.items()}
                serializable_rows = {
                    variant: {
                        key: (int(value) if key == "safe_pixels" else float(value))
                        for key, value in values.items()
                    }
                    for variant, values in rows.items()
                }
                row = {"optimizer_step": step, "variants": serializable_rows, "regions": regions.region_count,
                       "gradient_norm_before_clip": grad_norms, "elapsed_seconds": time.perf_counter()-started}
                with (output_dir / "joint_train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save({
                    "protocol": JOINT_PROTOCOL, "variants": VARIANTS, "optimizer_step": step,
                    "seed": args.seed, "base_checkpoint_sha256": base_sha,
                    "trainable_states": modules.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "args": config, "validation_accessed": False,
                    "official_instances_accessed": False, "test_accessed": False,
                }, checkpoint_path)
            if step >= args.optimizer_steps:
                break
    print(f"Saved: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
