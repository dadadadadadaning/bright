#!/usr/bin/env python3
"""Re-evaluate one saved ablation checkpoint on validation only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
BDA = ROOT / "bda_benchmark"
for value in (str(ROOT), str(BDA), str(HERE), str(BDA / "script/standard_ML")):
    if value not in sys.path:
        sys.path.insert(0, value)

from core import extract_state, get_variant, load_parent, seed_worker, sha256_file, write_json  # noqa: E402
from dataset.make_data_loader import MultimodalDamageAssessmentDatset  # noqa: E402
from model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL  # noqa: E402
from train import evaluate, format_metrics, read_list  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("a0", "a1", "a2", "a3"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint_kind", choices=("ablation", "parent_damageformer"),
        default="ablation",
        help="Use parent_damageformer to migrate/evaluate the frozen D0 checkpoint as A0 step 0.",
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow_test_access", action="store_true",
        help="Explicit one-shot authorization for a frozen test evaluation.",
    )
    args = parser.parse_args()
    is_test = "test" in Path(args.val_list).name.lower()
    if is_test and not args.allow_test_access:
        raise RuntimeError(
            "test access is locked; use the D0/A0 test_locked workflow after freezing both checkpoints"
        )
    spec = get_variant(args.variant)
    model = DamageFormer_S8_MICB_DSMIL(
        roi_alpha=spec.roi_alpha, detach_building_prior=True,
        token_dim=128, mil_topk=8, max_tokens_per_bag=256,
        mil_residual_scale=0.0,
    )
    parent_report = None
    if args.checkpoint_kind == "parent_damageformer":
        if args.variant != "a0":
            raise RuntimeError("a parent DamageFormer checkpoint may only be evaluated as a0")
        parent_report = load_parent(model, args.checkpoint)
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("variant") not in (None, args.variant):
            raise RuntimeError("checkpoint variant mismatch")
        model.load_state_dict(extract_state(payload), strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), 640, type="val"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    metrics = evaluate(model, loader, device, args.ignore_index)
    result = {
        "protocol": "D0_A0_TEST878_LOCKED_V1" if is_test else "P3_DAMAGEFORMER_ABLATION_VAL_V1",
        "variant": args.variant,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_kind": args.checkpoint_kind, "parent_report": parent_report,
        "metrics": metrics,
        "processed_images": len(dataset), "test_accessed": is_test,
    }
    write_json(Path(args.output_json), result)
    print(format_metrics(metrics))
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
