#!/usr/bin/env python3
"""Validation-only Q1-R0 audit of sparse building damage evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_Q0_Readout import (  # noqa: E402
    Q0BuildingConditionedSARReadout,
    assert_q0_freeze_contract,
)
from bda_benchmark.script.standard_ML.q1_r0_runtime import (  # noqa: E402
    RegionProtocolAccumulator,
    decide_r0,
)
from bda_benchmark.script.standard_ML.train_Q0_SAR_building_readout import (  # noqa: E402
    BEST_PROTOCOL,
    atomic_json,
    file_sha256,
    load_locked_source,
    make_dataset,
    make_loader,
    seed_everything,
)


@torch.inference_mode()
def p3_building_probability(model, rgb: torch.Tensor, output_size) -> torch.Tensor:
    stages = model.source.rgb_encoder(rgb)
    _, logits = model.source.p3_spatial_anchor(stages)
    probability = logits.float().softmax(1)[:, 1:2]
    return F.interpolate(
        probability, output_size, mode="bilinear", align_corners=False
    )[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser("Q1-R0 parameter-free building aggregation audit")
    parser.add_argument("--q0_checkpoint", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--support", choices=("oracle", "p3"), default="p3")
    parser.add_argument("--p3_threshold", type=float, default=0.5)
    parser.add_argument("--minimum_pixels", type=int, default=16)
    parser.add_argument("--minimum_building_fraction", type=float, default=0.50)
    parser.add_argument("--minimum_class_purity", type=float, default=0.75)
    parser.add_argument("--lse_beta", type=float, default=10.0)
    parser.add_argument("--area_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if not 0 < args.p3_threshold < 1:
        raise ValueError("P3 threshold must lie in (0,1)")
    if not 0 < args.minimum_class_purity <= 1:
        raise ValueError("class purity must lie in (0,1]")
    seed_everything(args.seed)
    checkpoint = torch.load(args.q0_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != BEST_PROTOCOL:
        raise RuntimeError(f"unexpected Q0 checkpoint: {checkpoint.get('protocol')!r}")
    if checkpoint.get("test_accessed") is not False:
        raise RuntimeError("Q1-R0 refuses a Q0 checkpoint without test_accessed=False")
    source_hash = file_sha256(args.source_checkpoint)
    if checkpoint.get("source", {}).get("sha256") != source_hash:
        raise RuntimeError("Q1-R0 H6-A2 source fingerprint mismatch")

    device = torch.device(args.device)
    source, provenance = load_locked_source(args.source_checkpoint, device)
    model = Q0BuildingConditionedSARReadout(source, checkpoint["variant"]).to(device)
    model.readout.load_state_dict(checkpoint["readout"], strict=True)
    assert_q0_freeze_contract(model)
    model.eval()
    dataset_args = argparse.Namespace(
        dataset_path=args.dataset_path,
        crop_size=args.crop_size,
        sar_stats_path=args.sar_stats_path,
        focus_probability=0.0,
    )
    dataset = make_dataset(dataset_args, args.data_list, "val")
    loader = make_loader(dataset, 1, args.num_workers, False, args.seed)
    accumulator = RegionProtocolAccumulator()
    processed = 0
    for rgb, sar, _, damage, names in loader:
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        outputs = model(rgb, sar)
        probabilities = outputs["logits"].float().softmax(1)
        if args.support == "oracle":
            support = (damage >= 1) & (damage <= 3)
        else:
            support = p3_building_probability(model, rgb, damage.shape[-2:]) >= args.p3_threshold
        accumulator.add_image(
            probabilities[0], damage[0], support[0], names[0],
            minimum_pixels=args.minimum_pixels,
            minimum_building_fraction=args.minimum_building_fraction,
            minimum_class_purity=args.minimum_class_purity,
            lse_beta=args.lse_beta,
            area_threshold=args.area_threshold,
        )
        processed += 1
        if processed % 50 == 0:
            print(f"Q1-R0 region audit: {processed}", flush=True)
        if args.max_images and processed >= args.max_images:
            break
    results = accumulator.compute()
    decision = decide_r0(results)
    payload = {
        "protocol": "Q1_R0_building_region_aggregation_audit_v1",
        "q0_checkpoint": {
            "path": str(Path(args.q0_checkpoint).resolve()),
            "variant": checkpoint["variant"],
            "step": int(checkpoint.get("step", 0)),
            "strict_readout_load": True,
        },
        "source": provenance,
        "split": "val",
        "support": args.support,
        "parameters": {
            "p3_threshold": args.p3_threshold,
            "minimum_pixels": args.minimum_pixels,
            "minimum_building_fraction": args.minimum_building_fraction,
            "minimum_class_purity": args.minimum_class_purity,
            "lse_beta": args.lse_beta,
            "area_threshold": args.area_threshold,
        },
        "processed_images": processed,
        "results": results,
        "advancement_gate": decision,
        "training": "NONE",
        "test_accessed": False,
    }
    atomic_json(payload, args.output_json)
    print("=" * 104)
    print(f"Q1-R0 | support={args.support} | regions={results['eligible_regions']} "
          f"| building coverage={100*results['building_pixel_coverage']:.2f}%")
    print(f"{'Method':<10} {'MacroF1':>10} {'DmgRec':>10} {'DesRec':>10} {'EventF1':>10}")
    print("-" * 58)
    for method, value in results["methods"].items():
        metric = value["region"]
        print(f"{method:<10} {100*metric['macro_f1']:>10.3f} "
              f"{100*metric['damaged_recall']:>10.3f} "
              f"{100*metric['destroyed_recall']:>10.3f} "
              f"{100*value['event_macro_f1']:>10.3f}")
    print(f"Decision : {decision['decision']}")
    print(f"Best     : {decision['best_local_method']} | "
          f"dDmg={decision['best_minus_mean_pp']['damaged_recall']:+.3f} pp")
    print(f"Saved    : {args.output_json}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
