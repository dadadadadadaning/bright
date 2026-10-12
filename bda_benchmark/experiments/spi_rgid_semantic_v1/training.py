from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .fast_targets import build_rgid_targets_fast


def train_rgid_head_fast(*, head, train_loader, config, output_dir,
                         max_optimizer_steps, accumulation_steps,
                         learning_rate, weight_decay, checkpoint_steps,
                         device, criterion, checkpoint_extra=None,
                         log_every=20, gradient_clip_norm=5.0):
    device = torch.device(device)
    head = head.to(device).train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"
    checkpoints = set(map(int, checkpoint_steps)) | {int(max_optimizer_steps)}
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = micro_step = 0
    sums = {"loss": 0.0, "center_loss": 0.0, "boundary_loss": 0.0,
            "valid_instances": 0, "target_seconds": 0.0}
    step_started = time.perf_counter()
    while optimizer_step < max_optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        fused = batch["fused_roi"].to(device).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            predictions = head(fused)
        target_started = time.perf_counter()
        targets, target_stats = build_rgid_targets_fast(
            batch["instance_ids"], predictions["center_logits"].shape[-2:], config, device
        )
        sums["target_seconds"] += time.perf_counter() - target_started
        losses = criterion(predictions, targets)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("non-finite RGID loss")
        (losses["loss"] / accumulation_steps).backward()
        for key in ("loss", "center_loss", "boundary_loss"):
            sums[key] += float(losses[key].detach().cpu())
        sums["valid_instances"] += sum(row.valid_instances for row in target_stats)
        micro_step += 1
        if micro_step < accumulation_steps:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        elapsed = time.perf_counter() - step_started
        row = {
            "optimizer_step": optimizer_step,
            "loss": sums["loss"] / accumulation_steps,
            "center_loss": sums["center_loss"] / accumulation_steps,
            "boundary_loss": sums["boundary_loss"] / accumulation_steps,
            "valid_instances": sums["valid_instances"],
            "gradient_norm_before_clip": float(torch.as_tensor(gradient_norm).cpu()),
            "step_seconds": elapsed,
            "target_seconds": sums["target_seconds"],
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if log_every > 0 and optimizer_step % log_every == 0:
            print(json.dumps(row), flush=True)
        if optimizer_step in checkpoints:
            payload = {
                "optimizer_step": optimizer_step,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                **(checkpoint_extra or {}),
            }
            path = output / f"rgid_step_{optimizer_step}.pth"
            temporary = path.with_suffix(".pth.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        micro_step = 0
        sums = {"loss": 0.0, "center_loss": 0.0, "boundary_loss": 0.0,
                "valid_instances": 0, "target_seconds": 0.0}
        step_started = time.perf_counter()

