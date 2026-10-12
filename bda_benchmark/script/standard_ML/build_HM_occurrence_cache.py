#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build exact-replay Stage-2 Occurrence-key caches for O-K0.

The accepted P3 building components are taken from a current-runtime CMR-P0
cache.  Every component must replay exactly.  The script loads a fixed-step
S2O checkpoint, extracts label-free building keys, and refuses Test by design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import P4NativeSARDataset
from model.DamageFormer_P3_S2Occurrence import DamageFormerP3Stage2Occurrence
from model.DamageFormer_S8_P4_NativeSAR_MICB import load_p3_checkpoint
from model.hmemory.occurrence_key import (
    extract_occurrence_key_variants,
    occurrence_key_variant_names,
)
from script.standard_ML.p3_hard_occurrence_runtime import match_cached_component


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_temperatures(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("temperatures must be a non-empty list of positive values")
    return values


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Build hierarchical-memory Occurrence keys")
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--p0_cache", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--s2o_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--expected_s2o_step", type=int, default=500)
    parser.add_argument("--temperatures", default="0.25,0.5,1.0")
    args = parser.parse_args()

    p0 = torch.load(args.p0_cache, map_location="cpu", weights_only=False)
    metadata = p0.get("metadata", {})
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("O-K0 requires a current-runtime CMR-P0 cache")
    expected_p0_role = "train" if args.role == "train" else "query"
    if metadata.get("role") != expected_p0_role:
        raise RuntimeError(f"P0 role mismatch: {metadata.get('role')} != {expected_p0_role}")
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("O-K0 refuses a Test-derived P0 cache")

    p3_sha = sha256(args.p3_checkpoint)
    if p3_sha != metadata.get("p3_checkpoint_sha256"):
        raise RuntimeError("P3 checkpoint SHA differs from the P0 cache")
    checkpoint = torch.load(args.s2o_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != "P3_S2O_frozen_stage2_occurrence_delta_v1":
        raise RuntimeError("O-K0 requires a fixed P3-S2O checkpoint")
    if checkpoint.get("base_p3_checkpoint_sha256") != p3_sha:
        raise RuntimeError("S2O and P0 use different accepted P3 checkpoints")
    if int(checkpoint.get("step", -1)) != int(args.expected_s2o_step):
        raise RuntimeError(
            f"S2O step {checkpoint.get('step')} != locked step {args.expected_s2o_step}"
        )

    temperatures = parse_temperatures(args.temperatures)
    variant_names = occurrence_key_variant_names(temperatures)
    device = torch.device(args.device)
    config = checkpoint.get("config", {})
    model = DamageFormerP3Stage2Occurrence(
        maximum_delta=float(config.get("maximum_delta", 2.0)),
    ).to(device)
    load_p3_checkpoint(model.p3, args.p3_checkpoint)
    model.occurrence_encoder.load_state_dict(checkpoint["occurrence_encoder"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows_by_image: dict[str, list[int]] = defaultdict(list)
    for row, image_id in enumerate(p0["image_ids"]):
        rows_by_image[str(image_id)].append(int(row))
    dataset = P4NativeSARDataset(
        dataset_path=args.dataset_path,
        data_list=sorted(rows_by_image),
        crop_size=args.crop_size,
        split="cache",
        variant="a0",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    extracted: dict[str, list[torch.Tensor | None]] = {
        name: [None] * len(p0["image_ids"]) for name in variant_names
    }
    matched = 0
    for image_index, batch in enumerate(loader, 1):
        name = str(batch["name"][0])
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        output_loc, _, auxiliary = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        for row in rows_by_image[name]:
            component = match_cached_component(
                building_mask,
                p0["bboxes"][row].tolist(),
                int(p0["areas"][row]),
            )
            if component is None:
                raise RuntimeError(
                    f"O-K0 exact component replay failed for {name} row={row}; "
                    "rebuild the P0 cache instead of using approximate boxes"
                )
            keys = extract_occurrence_key_variants(
                auxiliary,
                torch.from_numpy(component.astype("float32")),
                temperatures=temperatures,
            )
            for variant, key in keys.items():
                # O-K0 is a representation audit; retain FP32 so close AUROC
                # ranks cannot change because of cache quantisation.
                extracted[variant][row] = key.float()
            matched += 1
        if image_index % 50 == 0 or image_index == len(loader):
            print(
                f"HM O-K0 {args.role}: {image_index}/{len(loader)} | rows={matched}",
                flush=True,
            )

    if matched != len(p0["image_ids"]):
        raise RuntimeError("O-K0 did not replay every cached building")
    if any(any(value is None for value in values) for values in extracted.values()):
        raise RuntimeError("O-K0 key extraction left empty rows")
    key_variants = {
        name: torch.stack([value for value in values if value is not None])
        for name, values in extracted.items()
    }
    payload = {
        "protocol": "HM_OK0_exact_replay_s2o_occurrence_keys_v1",
        "role": args.role,
        "key_variants": key_variants,
        "labels": p0["labels"].long(),
        "base_predictions": p0["base_predictions"].long(),
        "base_probabilities": p0["base_probabilities"].float(),
        "events": list(map(str, p0["events"])),
        "image_ids": list(map(str, p0["image_ids"])),
        "bboxes": p0["bboxes"].long(),
        "areas": p0["areas"].long(),
        "source_rows": torch.arange(len(p0["image_ids"]), dtype=torch.long),
        "metadata": {
            "p0_cache": str(Path(args.p0_cache).resolve()),
            "p0_cache_sha256": sha256(args.p0_cache),
            "p3_checkpoint_sha256": p3_sha,
            "s2o_checkpoint": str(Path(args.s2o_checkpoint).resolve()),
            "s2o_checkpoint_sha256": sha256(args.s2o_checkpoint),
            "s2o_step": int(checkpoint["step"]),
            "temperatures": list(temperatures),
            "key_variants": list(variant_names),
            "component_match_rate": 1.0,
            "diagnostic_gt_eligibility_inherited_from_p0": bool(
                metadata.get("diagnostic_gt_eligibility", False)
            ),
            "inference_uses_memory": False,
            "validation_selects_key": False,
            "test_accessed": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(payload["metadata"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 104)
    print(f"Role/regions   : {args.role}/{matched}")
    print(f"Key variants   : {', '.join(variant_names)}")
    print("Exact replay   : 100.000%")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
