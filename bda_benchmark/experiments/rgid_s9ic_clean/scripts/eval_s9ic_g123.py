#!/usr/bin/env python3
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from rgid.evaluation import evaluate_g123, load_head_checkpoint, save_evaluation


def build_validation_loader(args) -> DataLoader:
    """Return batches with fused_roi, parent_mask and instance_ids."""

    raise NotImplementedError(
        "Connect the frozen S8 validation feature/pseudo-instance loader."
    )


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate one RGID checkpoint as G1/G2/G3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boundary-gate-threshold", type=float, default=None)
    args = parser.parse_args()
    head, config, _ = load_head_checkpoint(args.checkpoint, args.device)
    results = evaluate_g123(
        head=head,
        dataloader=build_validation_loader(args),
        config=config,
        device=args.device,
        boundary_gate_threshold=args.boundary_gate_threshold,
    )
    save_evaluation(results, args.output_json)


if __name__ == "__main__":
    main()
