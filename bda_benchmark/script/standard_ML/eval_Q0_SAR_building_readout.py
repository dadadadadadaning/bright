#!/usr/bin/env python3
"""Strict independent validation for Q0; the test split is forbidden."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_Q0_Readout import (  # noqa: E402
    Q0BuildingConditionedSARReadout,
    assert_q0_freeze_contract,
)
from bda_benchmark.script.standard_ML.train_Q0_SAR_building_readout import (  # noqa: E402
    BEST_PROTOCOL,
    atomic_json,
    evaluate,
    file_sha256,
    load_locked_source,
    make_dataset,
    make_loader,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser("Strict Q0 independent validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mask_threshold", type=float, default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != BEST_PROTOCOL:
        raise RuntimeError(f"unexpected Q0 protocol: {checkpoint.get('protocol')!r}")
    if checkpoint.get("test_accessed") is not False:
        raise RuntimeError("Q0 checkpoint does not declare test_accessed=False")
    if file_sha256(args.source_checkpoint) != checkpoint.get("source", {}).get("sha256"):
        raise RuntimeError("independent validation source fingerprint mismatch")
    device = torch.device(args.device)
    source_model, source = load_locked_source(args.source_checkpoint, device)
    model = Q0BuildingConditionedSARReadout(source_model, checkpoint["variant"]).to(device)
    model.readout.load_state_dict(checkpoint["readout"], strict=True)
    assert_q0_freeze_contract(model)
    threshold = (
        float(checkpoint["mask_threshold"])
        if args.mask_threshold is None else float(args.mask_threshold)
    )
    dataset = make_dataset(args, args.data_list, "val")
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    metrics = evaluate(model, loader, device, threshold, args.max_batches)
    payload = {
        "protocol": "Q0_SAR_building_readout_independent_validation_v1",
        "split": "val",
        "variant": checkpoint["variant"],
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "step": int(checkpoint.get("step", 0)),
            "strict_readout_load": True,
        },
        "source": source,
        "metrics": metrics,
        "final_model_modified": False,
        "test_accessed": False,
    }
    atomic_json(payload, args.output_json)
    conditional = metrics["conditional"]
    print("=" * 88)
    print(f"Q0 {checkpoint['variant']} independent validation")
    print(f"Macro F1       : {100*conditional['macro_f1']:.4f}")
    print(f"Recall I/D/X   : {100*conditional['intact_recall']:.4f} / "
          f"{100*conditional['damaged_recall']:.4f} / {100*conditional['destroyed_recall']:.4f}")
    print(f"Building cover : {100*metrics['building_coverage']:.4f}")
    print(f"Event macro F1 : {100*metrics['event_macro_f1']:.4f}")
    print(f"Old sar_aux D-R: {100*metrics['source_sar_aux']['damaged_recall']:.4f}")
    print(f"Q0 gate passed : {metrics['advancement_gate']['passed']}")
    print(f"Saved          : {args.output_json}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
