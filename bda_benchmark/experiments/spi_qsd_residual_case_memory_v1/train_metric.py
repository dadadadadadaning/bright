from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_independent_building_memory_v1.audit import load_independent
from bda_benchmark.experiments.spi_independent_building_memory_v1.core import group_matrix

from . import PROTOCOL
from .core import (DAMAGE_GROUPS, QUALITY_GROUPS, STRUCTURE_GROUPS,
                   FactorizedCaseMetric, build_pair_pools,
                   sample_metric_pairs)


def main(argv=None):
    parser = argparse.ArgumentParser("Train QSD cross-event residual-direction Memory metric")
    parser.add_argument("--train-cards", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--minimum-purity", type=float, default=.60)
    parser.add_argument("--progress-every", type=int, default=100); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = load_independent(args.train_cards)
    # Build each train matrix once.  prepare_groups(train, train) would retain a
    # second 88k-row copy and can needlessly exhaust host RAM.
    structure, structure_stats = group_matrix(train, STRUCTURE_GROUPS)
    damage, damage_stats = group_matrix(train, DAMAGE_GROUPS)
    _, quality_stats = group_matrix(train, QUALITY_GROUPS)
    model = FactorizedCaseMetric(structure.shape[1], damage.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    pools = build_pair_pools(train, args.minimum_purity)
    for step in range(1, args.steps + 1):
        left, right, label = sample_metric_pairs(train, args.batch_size, generator, pools)
        logits = model.pair_logits(structure[left].to(device), damage[left].to(device),
                                   structure[right].to(device), damage[right].to(device))
        loss = F.binary_cross_entropy_with_logits(logits, label.to(device))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % args.progress_every == 0:
            with torch.no_grad():
                prediction = (logits >= 0).float(); accuracy = float((prediction == label.to(device)).float().mean())
            print(json.dumps({"step": step, "loss": float(loss.detach()), "pair_accuracy": accuracy}), flush=True)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": PROTOCOL, "model_state_dict": model.cpu().state_dict(),
                "structure_dim": structure.shape[1], "damage_dim": damage.shape[1], "hidden": args.hidden,
                "statistics": {"structure": structure_stats, "damage": damage_stats,
                               "quality": quality_stats}, "configuration": vars(args),
                "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}, path)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
