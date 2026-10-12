from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_MarginAwareAO_V21 import DamageFormerMarginAwareAOV21
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, parse_batch, seed_worker, set_random_seed


PROTOCOL = "SPI_MARGIN_AWARE_AO_TRAIN_ONLY_v21"


def candidate_balanced_loss(score_logits, candidate, damage):
    score = F.interpolate(score_logits, damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0].float()
    mask = F.interpolate(candidate.float(), damage.shape[-2:], mode="nearest")[:, 0].bool() & (damage != 255)
    positive = mask & (damage >= 2)
    negative = mask & (damage <= 1)
    raw = F.binary_cross_entropy_with_logits(score, positive.float(), reduction="none")
    positive_loss = (raw * positive).sum() / positive.sum().clamp_min(1)
    negative_loss = (raw * negative).sum() / negative.sum().clamp_min(1)
    loss = 0.5 * (positive_loss + negative_loss)
    return loss, int(positive.sum()), int(negative.sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train candidate-only margin-aware AO V2.1")
    parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--crop-size", type=int, default=640); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8); parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5); parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)
    device = torch.device(args.device); generator = set_random_seed(args.seed, 0, False)
    model = DamageFormerMarginAwareAOV21(threshold=args.threshold)
    model.load_parent_checkpoint(args.base_checkpoint); model.freeze_parent(); model.to(device); model.set_plugin_training()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    dataset = build_dataset(args.dataset_path, args.train_list, args.crop_size, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
                        worker_init_fn=seed_worker, generator=generator)
    step = micro = 0; optimizer.zero_grad(set_to_none=True)
    while step < args.optimizer_steps:
        for batch in loader:
            pre, post, _, damage, _ = parse_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, _, aux = model(pre, post, return_aux=True)
                loss, positive, negative = candidate_balanced_loss(aux["score_logits"], aux["candidate_mask"], damage)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite candidate loss; positive={positive}, negative={negative}")
            (loss / args.accumulation_steps).backward(); micro += 1
            if micro < args.accumulation_steps: continue
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); micro = 0; step += 1
            if step == 1 or step % args.log_every == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach()), "positive": positive,
                                  "negative": negative, "gradient_norm": float(gradient)}), flush=True)
            if step >= args.optimizer_steps: break
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": PROTOCOL, "model_state": model.state_dict(), "args": vars(args),
                "meta_calibration_accessed": False, "meta_confirm_accessed": False,
                "val450_accessed": False, "test_accessed": False}, path)
    print(f"Saved: {path}", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
