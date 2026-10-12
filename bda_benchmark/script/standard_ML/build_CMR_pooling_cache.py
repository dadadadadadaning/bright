# -*- coding: utf-8 -*-
"""Build train/validation caches for the isolated CMR pooling audit.

This is a diagnostic-only cache builder.  It freezes the accepted P3, reuses
its trained MICB modules, and stores several building pooling variants.  It
never updates P3 and refuses a test split by contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_p4_dataset,
    read_data_list,
)
from model.DamageFormer_S7_SIRM_M2B import connected_components_4
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from model.cmr.pooling import (
    CMR_KEY_VARIANTS,
    build_cmr_feature_maps,
    extract_cmr_pooling_keys,
)
from script.standard_ML.build_RM0_region_cache import _pool_component
from script.standard_ML.rm_region_retrieval_runtime import (
    class_event_balanced_indices,
    class_event_count_table,
    event_from_name,
    validate_disjoint_ids,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_fractions(target: torch.Tensor, component: np.ndarray) -> torch.Tensor:
    mask = torch.from_numpy(component.astype(np.bool_, copy=False))
    values = target.cpu()[mask]
    building = (values >= 1) & (values <= 3)
    counts = torch.bincount(values[building].long() - 1, minlength=3).float()
    return counts / counts.sum().clamp_min(1)


@torch.inference_mode()
def extract(args: argparse.Namespace) -> Dict[str, object]:
    ids = read_data_list(args.data_list_path)
    train_ids = read_data_list(args.train_data_list_path)
    val_ids = read_data_list(args.val_data_list_path)
    if args.role == "train":
        if set(ids) != set(train_ids):
            raise RuntimeError("CMR train cache source must exactly equal train_set.txt")
        validate_disjoint_ids(ids, val_ids)
    else:
        if set(ids) != set(val_ids):
            raise RuntimeError("CMR query cache source must exactly equal val_set.txt")
        validate_disjoint_ids(train_ids, ids)

    device = torch.device(args.device)
    model = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(model, args.p3_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = build_p4_dataset(
        args.dataset_path, args.data_list_path, args.crop_size, "cache", "a0", ""
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    records: List[Dict[str, object]] = []
    rejected_quality = 0
    rejected_resolution = 0
    for index, batch in enumerate(loader, 1):
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        target = batch["label_dam"][0].long()
        name = str(batch["name"][0])
        output_loc, output_damage, aux = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        probability = torch.softmax(output_damage[0].float(), dim=0).cpu()
        feature_maps = build_cmr_feature_maps(
            model.micb_dsmil,
            aux["rgb_feature"],
            aux["sar_feature"],
            aux["fused_roi"],
        )
        raw_sar = (
            post[0, 0].float().cpu() * float(IMAGENET_STD[0])
        ) + float(IMAGENET_MEAN[0])
        scene_stats = torch.tensor(
            [float(raw_sar.mean()), float(raw_sar.std(unbiased=False))],
            dtype=torch.float32,
        )
        for component in connected_components_4(building_mask):
            area = int(component.sum())
            if area < args.min_area or area > args.max_area:
                continue
            pooled = _pool_component(
                aux["fused_roi"][0].float(),
                probability,
                raw_sar,
                scene_stats,
                target,
                component,
            )
            if pooled is None:
                continue
            # Controlled P0 comparison: preserve RM3E's historical diagnostic
            # eligibility while recording it explicitly.  This cache is never
            # permitted for test inference; label-free query eligibility is V1.
            if (
                float(pooled["coverage"]) < args.min_building_coverage
                or float(pooled["purity"]) < args.min_class_purity
            ):
                rejected_quality += 1
                continue
            component_tensor = torch.from_numpy(
                component.astype(np.float32, copy=False)
            )
            try:
                keys = extract_cmr_pooling_keys(
                    model.micb_dsmil, feature_maps, component_tensor
                )
            except RuntimeError:
                rejected_resolution += 1
                continue
            pooled.update({
                "key_variants": keys,
                "class_fractions": _class_fractions(target, component),
                "image_id": name,
                "event": event_from_name(name),
            })
            records.append(pooled)
        if index % 50 == 0 or index == len(loader):
            print(
                f"CMR {args.role} cache: {index}/{len(loader)} | regions={len(records)}",
                flush=True,
            )

    if not records:
        raise RuntimeError("no eligible CMR building regions were extracted")
    key_variants = {
        name: torch.stack([item["key_variants"][name] for item in records])
        for name in CMR_KEY_VARIANTS
    }
    payload: Dict[str, object] = {
        "key_variants": key_variants,
        # Compatibility alias: this must exactly be the historical mean key.
        "keys": key_variants["mean"],
        "labels": torch.tensor([item["label"] for item in records], dtype=torch.long),
        "base_predictions": torch.tensor(
            [item["base_prediction"] for item in records], dtype=torch.long
        ),
        "base_probabilities": torch.stack(
            [item["base_probability"] for item in records]
        ),
        "class_fractions": torch.stack(
            [item["class_fractions"] for item in records]
        ),
        "raw_stats": torch.stack([item["raw_stats"] for item in records]),
        "scene_stats": torch.stack([item["scene_stats"] for item in records]),
        "areas": torch.tensor([item["area"] for item in records], dtype=torch.long),
        "coverage": torch.tensor([item["coverage"] for item in records]),
        "purity": torch.tensor([item["purity"] for item in records]),
        "bboxes": torch.tensor([item["bbox"] for item in records], dtype=torch.long),
        "image_ids": [str(item["image_id"]) for item in records],
        "events": [str(item["event"]) for item in records],
    }
    counts = torch.bincount(payload["labels"], minlength=3)
    if args.role == "train":
        memory_indices = class_event_balanced_indices(
            payload["labels"],
            payload["events"],
            args.capacity_per_class,
            args.capacity_per_event_class,
            args.seed,
        )
        payload["memory_indices"] = memory_indices
        memory_counts = torch.bincount(
            payload["labels"][memory_indices], minlength=3
        ).tolist()
    else:
        memory_counts = None
    payload["metadata"] = {
        "protocol": "CMR_P0_frozen_pooling_audit_v1",
        "role": args.role,
        "source_split": args.role,
        "source_list": str(Path(args.data_list_path).resolve()),
        "source_list_sha256": sha256(args.data_list_path),
        "p3_checkpoint": str(Path(args.p3_checkpoint).resolve()),
        "p3_checkpoint_sha256": sha256(args.p3_checkpoint),
        "key_variants": list(CMR_KEY_VARIANTS),
        "region_source": "frozen_p3_connected_components",
        "feature_source": "frozen_p3_fused_roi_and_micb",
        "diagnostic_gt_eligibility": True,
        "test_accessed": False,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "min_building_coverage": args.min_building_coverage,
        "min_class_purity": args.min_class_purity,
        "class_counts": counts.tolist(),
        "class_event_counts": class_event_count_table(
            payload["labels"], payload["events"]
        ),
        "memory_class_counts": memory_counts,
        "capacity_per_class": args.capacity_per_class,
        "capacity_per_event_class": args.capacity_per_event_class,
        "rejected_quality": int(rejected_quality),
        "rejected_feature_resolution": int(rejected_resolution),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("Build CMR frozen-P3 pooling cache")
    parser.add_argument("--role", choices=("train", "query"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_area", type=int, default=32)
    parser.add_argument("--max_area", type=int, default=300000)
    parser.add_argument("--min_building_coverage", type=float, default=0.50)
    parser.add_argument("--min_class_purity", type=float, default=0.75)
    parser.add_argument("--capacity_per_class", type=int, default=2000)
    parser.add_argument("--capacity_per_event_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    payload = extract(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(payload["metadata"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 88)
    print(f"Role           : {args.role}")
    print(f"Regions        : {payload['labels'].numel()}")
    print(f"Class counts   : {payload['metadata']['class_counts']}")
    print(f"Key variants   : {', '.join(CMR_KEY_VARIANTS)}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
