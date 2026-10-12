from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_SuperpixelRelationOccurrence_V1 import DamageFormerSuperpixelRelationOccurrenceV1, VARIANTS
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, parse_batch, seed_worker, set_random_seed
from . import PROTOCOL
from .core import candidate_losses, candidate_token_targets, reliable_region_consistency


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(value, temporary); os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train frozen P3 superpixel relation occurrence adapter")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--crop-size", type=int, default=640); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8); parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--segments", type=int, default=192); parser.add_argument("--compactness", type=float, default=8.)
    parser.add_argument("--shift-cap", type=float, default=4.5); parser.add_argument("--lambda-margin", type=float, default=.25)
    parser.add_argument("--lambda-consistency", type=float, default=.02); parser.add_argument("--log-every", type=int, default=20); parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--min-building-fraction", type=float, default=.25)
    parser.add_argument("--max-consecutive-skips", type=int, default=500)
    args = parser.parse_args(argv); device = torch.device(args.device); generator = set_random_seed(args.seed, 0, False)
    model = DamageFormerSuperpixelRelationOccurrenceV1(args.variant, segments=args.segments, compactness=args.compactness)
    model.load_parent_checkpoint(args.base_checkpoint); model.freeze_parent(); model.to(device); model.set_plugin_training()
    parameters = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    # Prefer damaged-centred crops when supported by the standard BRIGHT
    # dataset.  This changes crop selection only, never labels or split files.
    if hasattr(dataset, "use_damage_aware_sampler"):
        dataset.use_damage_aware_sampler = True
        dataset.damage_crop_prob = 0.50
        dataset.intact_crop_prob = 0.25
        dataset.random_crop_prob = 0.25
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0, pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": PROTOCOL, "variant": args.variant, "args": vars(args), "frozen_p3": True,
                "superpixels_from": "augmented_pre_disaster_rgb_only", "semantic_superpixel_labels": False,
                "token_grid_supervision": True, "meta_calibration_accessed": False, "meta_confirm_accessed": False,
                "val450_accessed": False, "test_accessed": False}
    (out / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    step = micro = skipped = consecutive_skips = consumed = cumulative_pos = cumulative_neg = 0; started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
    print(f"SPR protocol={PROTOCOL} | {args.variant} | trainable={sum(p.numel() for p in parameters):,} | candidate-token supervision", flush=True)
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, location, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, _, aux = model(pre, post, return_aux=True)
                positive, negative = candidate_token_targets(
                    aux["parent_logits_feature"], aux["p_building"], damage, location,
                    min_building_fraction=args.min_building_fraction,
                )
                # A zero-positive batch cannot teach a gate-miss head; skip it rather than silently train only negatives.
                if not bool(positive.any()):
                    skipped += 1; consecutive_skips += 1
                    if consecutive_skips >= args.max_consecutive_skips:
                        raise RuntimeError(
                            "too many consecutive batches without a P3 affected miss; "
                            "inspect crop sampling and min_building_fraction"
                        )
                    continue
                consecutive_skips = 0
                occurrence, margin, counts = candidate_losses(aux["occurrence_logits"], aux["affected_delta"], aux["parent_logits_feature"], positive, negative, args.shift_cap)
                consistency = aux["relation_feature"].square().mean() * 0.0
                if VARIANTS[args.variant].consistency:
                    # Only inner predicted-building tokens are regularised; no GT region label enters this term.
                    consistency = reliable_region_consistency(aux["occurrence_logits"], aux["region_ids"], aux["safe_gate"])
                loss = occurrence + args.lambda_margin * margin + args.lambda_consistency * consistency
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            (loss / args.accumulation_steps).backward(); micro += 1; consumed += 1
            cumulative_pos += counts["positive_tokens"]; cumulative_neg += counts["negative_tokens"]
            if micro < args.accumulation_steps: continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step(); model.project_safe_parameters(); optimizer.zero_grad(set_to_none=True)
            micro = 0; step += 1
            if step == 1 or step % args.log_every == 0:
                row = {"step": step, "variant": args.variant, "loss": float(loss.detach()), "occurrence_loss": float(occurrence.detach()), "margin_loss": float(margin.detach()), "consistency_loss": float(consistency.detach()), "gradient_norm": float(norm), "affected_gain": float(model.occurrence_residual.gain.detach()), "delta_mean": float(aux["affected_delta"].detach().float().mean()), "consumed_batches": consumed, "skipped_batches": skipped, "cumulative_positive_tokens": cumulative_pos, "cumulative_negative_tokens": cumulative_neg, "positive_batch_fraction": consumed / max(consumed + skipped, 1), "elapsed_seconds": time.perf_counter() - started, **counts}
                with (out / "train_log.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save({**manifest, "optimizer_step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, out / "last.pth")
            if step >= args.optimizer_steps: break
    print(f"Saved: {out / 'last.pth'}", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
