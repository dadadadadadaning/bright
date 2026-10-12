from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .config import RGIDConfig
from .targets_losses import RGIDLoss, build_rgid_targets


def _save_checkpoint(
    path,
    head,
    optimizer,
    optimizer_step,
    config,
    checkpoint_extra=None,
) -> None:
    payload = {
        "optimizer_step": int(optimizer_step),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    if checkpoint_extra:
        overlap = sorted(set(payload).intersection(checkpoint_extra))
        if overlap:
            raise KeyError(f"checkpoint_extra cannot replace core fields: {overlap}")
        payload.update(dict(checkpoint_extra))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_rgid_head(
    *,
    head,
    train_loader,
    config: RGIDConfig,
    output_dir,
    max_optimizer_steps=3000,
    accumulation_steps=4,
    learning_rate=1e-3,
    weight_decay=1e-4,
    checkpoint_steps=(1000, 2000, 3000),
    device="cuda",
    use_bf16=True,
    log_every=20,
    gradient_clip_norm=5.0,
    log_jsonl_path=None,
    checkpoint_extra=None,
):
    if accumulation_steps <= 0 or max_optimizer_steps <= 0:
        raise ValueError("optimizer and accumulation steps must be positive")
    device = torch.device(device)
    head = head.to(device)
    head.train()
    criterion = RGIDLoss(config)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(log_jsonl_path)
        if log_jsonl_path is not None
        else output / "train_log.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_set = {int(step) for step in checkpoint_steps}
    checkpoint_set.add(int(max_optimizer_steps))
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    history = []
    iterator = iter(train_loader)
    while optimizer_step < max_optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        if "fused_roi" not in batch or "instance_ids" not in batch:
            raise KeyError("batch requires fused_roi and instance_ids")
        fused_roi = batch["fused_roi"].to(device).float()
        instance_ids = batch["instance_ids"]
        amp_enabled = bool(use_bf16 and device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            predictions = head(fused_roi)
        targets, target_stats = build_rgid_targets(
            instance_ids,
            predictions["center_logits"].shape[-2:],
            config,
            device=device,
        )
        losses = criterion(predictions, targets)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError(
                f"non-finite RGID loss at optimizer step {optimizer_step}"
            )
        (losses["loss"] / accumulation_steps).backward()
        micro_step += 1
        if micro_step < accumulation_steps:
            continue
        if gradient_clip_norm is not None and float(gradient_clip_norm) > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), float(gradient_clip_norm)
            )
        else:
            gradient_norm = torch.zeros((), device=device)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        optimizer_step += 1
        row = {
            "optimizer_step": optimizer_step,
            "loss": float(losses["loss"].detach().cpu()),
            "center_loss": float(losses["center_loss"].detach().cpu()),
            "boundary_loss": float(losses["boundary_loss"].detach().cpu()),
            "valid_instances": int(
                sum(stats.valid_instances for stats in target_stats)
            ),
            "gradient_norm": float(
                torch.as_tensor(gradient_norm).detach().float().cpu()
            ),
        }
        history.append(row)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if log_every > 0 and optimizer_step % log_every == 0:
            print(json.dumps(row), flush=True)
        if optimizer_step in checkpoint_set:
            _save_checkpoint(
                output / f"rgid_step_{optimizer_step}.pth",
                head,
                optimizer,
                optimizer_step,
                config,
                checkpoint_extra,
            )
    return {
        "optimizer_steps": optimizer_step,
        "history": history,
        "last_loss": history[-1],
    }
