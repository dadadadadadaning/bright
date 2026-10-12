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

from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, parse_batch, seed_worker, segmentation_loss, set_random_seed

from .core import RiskGate, apply_gate, balanced_selector_bce, risk_features, selector_targets
from .runtime import frozen_candidate_forward, load_frozen_models, upsample_logits


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser("Train only the D4 risk gate")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--m4-checkpoint", required=True)
    parser.add_argument("--m4-steps", type=int, default=500)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--lambda-selector", type=float, default=1.0)
    parser.add_argument("--lambda-safe", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    generator = set_random_seed(args.seed, 0, False)
    p3, m4, manifest = load_frozen_models(args.base_checkpoint, args.m4_checkpoint, device, args.m4_steps)
    gate = RiskGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gate_last.pth"
    step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("p3_sha256") != manifest["p3_sha256"] or saved.get("m4_checkpoint") != manifest["m4_checkpoint"]:
            raise RuntimeError("existing gate checkpoint provenance mismatch")
        step = int(saved["optimizer_step"])
        gate.load_state_dict(saved["gate_state"], strict=True)
        if step < args.optimizer_steps:
            optimizer.load_state_dict(saved["optimizer_state"])
        if step >= args.optimizer_steps:
            print(f"D4 gate already complete at step={step}")
            return 0

    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker, generator=generator)
    ce_fn = nn.CrossEntropyLoss(ignore_index=255)
    region_args = dict(threshold=0.5, minimum_area=2, core_radius=1, ring_radius=2)
    config = vars(args).copy()
    config.update(protocol="SPI_D4_RISK_GATE_TRAIN_ONLY_v1", **manifest,
                  validation_accessed=False, official_instances_accessed=False, test_accessed=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("=" * 100)
    print(f"D4 risk gate only | trainable={sum(p.numel() for p in gate.parameters()):,} | P3/M4 frozen")
    print(f"steps={args.optimizer_steps} EBS={args.batch_size*args.accumulation_steps} device={device}")
    print("validation/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 100, flush=True)
    gate.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.perf_counter()
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, _, target, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, base_output, base_low, candidate_low, feature, regions = frozen_candidate_forward(p3, m4, pre, post, region_args)
                inputs, support, _ = risk_features(base_low, candidate_low, feature, regions)
                gate_values = gate(inputs, support)
                gated_low = apply_gate(base_low, candidate_low, gate_values)
                gated_output = upsample_logits(gated_low, pre.shape[-2:])
            semantic, ce, lovasz = segmentation_loss(gated_output, target, ce_fn, True, 1.0, 1.0, 255)
            target_low = F.interpolate(target[:, None].float(), size=base_low.shape[-2:], mode="nearest")[:, 0].long()
            selector_target, decisive, positive, negative = selector_targets(base_low, candidate_low, target_low)
            selector = balanced_selector_bce(gate_values.float(), selector_target, decisive)
            with torch.no_grad():
                base_probability = F.softmax(base_low.float(), dim=1)
                confidence, prediction = base_probability.max(1)
                safe = (prediction == target_low) & (confidence >= 0.8) & (target_low != 255)
            safe_float = safe[:, None].float()
            safe_loss = (gate_values.float() * safe_float).sum() / safe_float.sum().clamp_min(1.0)
            total = semantic + args.lambda_selector * selector + args.lambda_safe * safe_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite D4 loss at step={step}")
            (total / args.accumulation_steps).backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            step += 1
            if step == 1 or step % args.log_every == 0:
                row = {"optimizer_step": step, "loss": float(total.detach()), "semantic": float(semantic.detach()),
                       "selector": float(selector.detach()), "safe": float(safe_loss.detach()),
                       "gate_mean": float(gate_values.detach().mean()), "positive": int(positive.sum()),
                       "negative": int(negative.sum()), "gradient_norm_before_clip": float(grad_norm),
                       "elapsed_seconds": time.perf_counter() - started}
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save({"protocol": "SPI_D4_RISK_GATE_TRAIN_ONLY_v1", "optimizer_step": step,
                             "seed": args.seed, "p3_sha256": manifest["p3_sha256"],
                             "m4_checkpoint": manifest["m4_checkpoint"], "gate_state": gate.state_dict(),
                             "optimizer_state": optimizer.state_dict(), "args": config,
                             "validation_accessed": False, "official_instances_accessed": False,
                             "test_accessed": False}, checkpoint_path)
            if step >= args.optimizer_steps:
                break
    print(f"Saved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

