from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_independent_building_memory_v1.audit import load_independent
from bda_benchmark.experiments.spi_independent_building_memory_v1.core import group_matrix
from bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.core import DAMAGE_GROUPS, QUALITY_GROUPS, STRUCTURE_GROUPS
from . import PROTOCOL
from .core import FactorizedCaseMetric, build_hierarchical_pools, sample_hierarchical_pairs


def main(argv=None):
    p = argparse.ArgumentParser("Train error-focused hierarchical residual Memory metric")
    p.add_argument("--train-cards", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--minimum-purity", type=float, default=.60)
    p.add_argument("--progress-every", type=int, default=100); p.add_argument("--seed", type=int, default=7341)
    a = p.parse_args(argv); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cache = load_independent(a.train_cards)
    structure, s_stats = group_matrix(cache, STRUCTURE_GROUPS); damage, d_stats = group_matrix(cache, DAMAGE_GROUPS)
    _, q_stats = group_matrix(cache, QUALITY_GROUPS)
    state = build_hierarchical_pools(cache, a.minimum_purity)
    model = FactorizedCaseMetric(structure.shape[1], damage.shape[1], a.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.steps + 1):
        left, right, label, kinds = sample_hierarchical_pairs(cache, a.batch_size, generator, state)
        logits = model.pair_logits(structure[left].to(device), damage[left].to(device),
                                   structure[right].to(device), damage[right].to(device))
        loss = F.binary_cross_entropy_with_logits(logits, label.to(device))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % a.progress_every == 0:
            accuracy = float(((logits >= 0).cpu() == label.bool()).float().mean())
            print(json.dumps({"step": step, "loss": float(loss.detach()), "pair_accuracy": accuracy,
                              "composition": Counter(kinds)}), flush=True)
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": PROTOCOL, "model_state_dict": model.cpu().state_dict(),
                "structure_dim": structure.shape[1], "damage_dim": damage.shape[1], "hidden": a.hidden,
                "statistics": {"structure": s_stats, "damage": d_stats, "quality": q_stats},
                "configuration": vars(a), "meta_confirm_accessed": False,
                "val450_accessed": False, "test_accessed": False}, path)
    print(f"Saved: {path}"); return 0


if __name__ == "__main__": raise SystemExit(main())

