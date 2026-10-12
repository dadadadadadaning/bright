#!/usr/bin/env python3
"""Strict independent validation for Q2-B0/B1; test is not exposed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_Q2_BuildingResidual import (  # noqa: E402
    Q2BuildingResidualFusion,
    assert_q2_freeze_contract,
)
from bda_benchmark.script.standard_ML.q2_building_residual_runtime import (  # noqa: E402
    advancement_gate,
    evaluate_q2,
)
from bda_benchmark.script.standard_ML.train_Q0_SAR_building_readout import (  # noqa: E402
    atomic_json,
    file_sha256,
    seed_everything,
)
from bda_benchmark.script.standard_ML.train_Q2_building_residual import (  # noqa: E402
    BEST_PROTOCOL,
    load_experts,
    load_residual_state,
    make_dataset,
    make_loader,
)


def main():
    parser = argparse.ArgumentParser("Independent Q2 validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--a2_checkpoint", required=True)
    parser.add_argument("--q0_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--focus_probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != BEST_PROTOCOL:
        raise RuntimeError(f"unexpected Q2 checkpoint protocol: {checkpoint.get('protocol')!r}")
    if checkpoint.get("test_accessed") is not False:
        raise RuntimeError("Q2 independent validation refuses test-accessed weights")
    device = torch.device(args.device)
    p3, q0, provenance = load_experts(args, device)
    for source, value in provenance.items():
        expected = checkpoint.get("sources", {}).get(source, {}).get("sha256")
        if expected != value["sha256"]:
            raise RuntimeError(f"Q2 {source} fingerprint mismatch")
    proposal = checkpoint["proposal"]
    model = Q2BuildingResidualFusion(
        p3, q0, checkpoint["variant"],
        proposal_threshold=proposal["threshold"],
        minimum_area=proposal["minimum_area"],
        maximum_transition=checkpoint["maximum_transition"],
    ).to(device)
    load_residual_state(model, checkpoint["residual"])
    assert_q2_freeze_contract(model)
    dataset = make_dataset(args, "val")
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    metrics, base, audit = evaluate_q2(model, loader, device)
    gate = advancement_gate(metrics, base, audit)
    payload = {
        "protocol": "Q2_building_residual_independent_validation_v1",
        "split": "val",
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": file_sha256(args.checkpoint),
            "variant": checkpoint["variant"],
            "step": int(checkpoint.get("step", 0)),
            "strict_load": True,
        },
        "metrics": metrics,
        "base_metrics": base,
        "audit": audit,
        "advancement_gate": gate,
        "test_accessed": False,
    }
    atomic_json(payload, args.output_json)
    print(
        f"Q2 independent val | Valid={100*metrics['valid_miou']:.4f} "
        f"| Dmg={100*metrics['damaged_iou']:.4f} "
        f"| Event={100*metrics['event_mean_official_style_4class_miou']:.4f}/"
        f"{100*metrics['event_worst_official_style_4class_miou']:.4f}"
    )
    print(f"Correction/harm regions: {audit['regions'].get('corrected', 0)}/"
          f"{audit['regions'].get('harmed', 0)}")
    print(f"Advancement gate: {'PASSED' if gate['passed'] else 'FAILED'}")
    print(f"Saved: {args.output_json}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
