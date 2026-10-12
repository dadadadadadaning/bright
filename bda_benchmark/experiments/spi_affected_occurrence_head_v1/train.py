from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_AffectedOccurrence_V1 import DamageFormerAffectedOccurrenceV1, VARIANTS
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, parse_batch, seed_worker, segmentation_loss, set_random_seed
from bda_benchmark.experiments.spi_model_centric_v1.train import anchor_kl
from . import PROTOCOL
from .spatial import boundary_and_small_masks


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def balanced_occurrence_loss(logits, damage, location, *, boundary_weight: float, small_weight: float, small_area: float):
    logits = F.interpolate(logits.float(), damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    valid = (damage != 255) & (location > 0)
    positive, negative = valid & (damage >= 2), valid & (damage <= 1)
    boundary, small = boundary_and_small_masks(location, damage, small_area)
    weight = torch.ones_like(logits)
    weight = weight + float(boundary_weight) * (positive & boundary)
    weight = weight + float(small_weight) * (positive & small)
    bce = F.binary_cross_entropy_with_logits(logits, positive.float(), reduction="none") * weight
    positive_loss = bce[positive].mean() if bool(positive.any()) else bce.sum() * 0.0
    negative_loss = bce[negative].mean() if bool(negative.any()) else bce.sum() * 0.0
    return .5 * (positive_loss + negative_loss), {
        "positive_pixels": int(positive.sum()), "negative_pixels": int(negative.sum()),
        "boundary_positive_pixels": int((positive & boundary).sum()), "small_positive_pixels": int((positive & small).sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train frozen-P3 Affected Occurrence Head")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--small-area", type=float, required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--crop-size", type=int, default=640); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8); parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-occurrence", type=float, default=.5); parser.add_argument("--lambda-anchor", type=float, default=.05)
    parser.add_argument("--boundary-weight", type=float, default=1.0); parser.add_argument("--small-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20); parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)
    variant = VARIANTS[args.variant]
    device = torch.device(args.device); generator = set_random_seed(args.seed, 0, False)
    model = DamageFormerAffectedOccurrenceV1(args.variant); model.load_parent_checkpoint(args.base_checkpoint)
    model.freeze_parent(); model.to(device); model.set_plugin_training()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker, generator=generator)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": PROTOCOL, "variant": args.variant, "args": vars(args), "frozen_p3": True,
                "small_damage_target_used_at_inference": False, "meta_calibration_accessed": False,
                "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ce_fn = nn.CrossEntropyLoss(ignore_index=255); optimizer.zero_grad(set_to_none=True)
    step = micro = 0; started = time.perf_counter()
    print(f"AO-H {args.variant} | frozen P3 | trainable={sum(p.numel() for p in parameters):,}", flush=True)
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, location, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, prediction, aux = model(pre, post, return_aux=True)
                semantic, ce, lovasz = segmentation_loss(prediction, damage, ce_fn, True, 1., 1., 255)
                occurrence, counts = balanced_occurrence_loss(aux["occurrence_logits"], damage, location,
                    boundary_weight=args.boundary_weight if variant.boundary_weight else 0.,
                    small_weight=args.small_weight if variant.small_damage_weight else 0., small_area=args.small_area)
                parent = F.interpolate(aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False)
                anchor, safe = anchor_kl(prediction, parent.detach(), damage, .85)
                loss = semantic + args.lambda_occurrence * occurrence + args.lambda_anchor * anchor
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / args.accumulation_steps).backward(); micro += 1
            if micro < args.accumulation_steps: continue
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step(); model.project_safe_parameters()
            optimizer.zero_grad(set_to_none=True); micro = 0; step += 1
            if step == 1 or step % args.log_every == 0:
                row = {"step": step, "variant": args.variant, "loss": float(loss.detach()), "semantic": float(semantic.detach()),
                       "occurrence": float(occurrence.detach()), "anchor": float(anchor.detach()), "ce": float(ce.detach()),
                       "lovasz": float(lovasz.detach()), "gradient_norm": float(gradient), "affected_gain": float(model.occurrence_residual.gain.detach()),
                       "delta_mean": float(aux["affected_delta"].detach().float().mean()), "sar_quality_mean": float(aux["sar_quality"].detach().float().mean()),
                       "elapsed_seconds": time.perf_counter()-started, **counts, "safe_pixels": safe}
                with (output / "train_log.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save({**manifest, "optimizer_step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, output / "last.pth")
            if step >= args.optimizer_steps: break
    print(f"Saved: {output / 'last.pth'}", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
