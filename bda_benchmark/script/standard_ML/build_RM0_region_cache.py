# -*- coding: utf-8 -*-
"""Build frozen-P3 region caches for memory, calibration, and val queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
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
from script.standard_ML.rm_region_retrieval_runtime import (
    balanced_indices,
    class_event_balanced_indices,
    class_event_count_table,
    event_from_name,
    raw_stat_normalizer,
    scene_relative_raw_stats,
    validate_disjoint_ids,
)
from script.standard_ML.im4cd_matching_runtime import (
    COLOCATED_STAT_NAMES,
    MATCHING_STAT_NAMES,
    hard_local_matching_maps,
    pool_region_matching_statistics,
    vector_normalizer,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_component(
    feature: torch.Tensor,
    damage_probability: torch.Tensor,
    raw_sar: torch.Tensor,
    scene_stats: torch.Tensor,
    target: torch.Tensor,
    component: np.ndarray,
    matching_maps: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, object] | None:
    full_mask = torch.from_numpy(component.astype(np.float32, copy=False)).to(feature.device)
    weights = F.interpolate(
        full_mask[None, None], size=feature.shape[-2:], mode="area"
    )[0, 0]
    denominator = weights.sum()
    if float(denominator) <= 1e-6:
        return None
    visual = (feature * weights[None]).sum(dim=(1, 2)) / denominator
    visual = F.normalize(visual.float(), dim=0).cpu()

    full_mask_cpu = full_mask.bool().cpu()
    region_target = target.cpu()[full_mask_cpu]
    building = (region_target >= 1) & (region_target <= 3)
    coverage = float(building.float().mean()) if region_target.numel() else 0.0
    if int(building.sum()) == 0:
        return None
    counts = torch.bincount(region_target[building].long() - 1, minlength=3)
    label = int(counts.argmax())
    purity = float(counts.max() / counts.sum().clamp_min(1))

    probability = damage_probability[:, full_mask_cpu].mean(dim=1)
    conditional = probability[1:4]
    conditional = conditional / conditional.sum().clamp_min(1e-8)
    raw_values = raw_sar.cpu()[full_mask_cpu]
    ys, xs = np.nonzero(component)
    result = {
        "key": visual,
        "label": label,
        "base_prediction": int(conditional.argmax()),
        "base_probability": conditional.float().cpu(),
        "raw_stats": torch.tensor(
            [float(raw_values.mean()), float(raw_values.std(unbiased=False))],
            dtype=torch.float32,
        ),
        "scene_stats": scene_stats.float().cpu(),
        "coverage": coverage,
        "purity": purity,
        "area": int(component.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
    }
    if matching_maps is not None:
        result.update(pool_region_matching_statistics(matching_maps, weights))
    return result


@torch.inference_mode()
def extract(args: argparse.Namespace) -> Dict[str, object]:
    ids = read_data_list(args.data_list_path)
    train_ids = read_data_list(args.train_data_list_path)
    val_ids = read_data_list(args.val_data_list_path)
    if args.role in ("memory", "calibration"):
        if set(ids) != set(train_ids):
            raise RuntimeError(
                f"RM {args.role} source must exactly equal train_set.txt"
            )
        validate_disjoint_ids(ids, val_ids)
    else:
        if set(ids) != set(val_ids):
            raise RuntimeError("RM query source must exactly equal val_set.txt")
        validate_disjoint_ids(train_ids, ids)

    device = torch.device(args.device)
    model = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(model, args.p3_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = build_p4_dataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        "cache",
        "a0",
        "",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    records: List[Dict[str, object]] = []
    for index, batch in enumerate(loader, 1):
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        target = batch["label_dam"][0].long()
        name = str(batch["name"][0])
        output_loc, output_damage, aux = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        probability = torch.softmax(output_damage[0].float(), dim=0).cpu()
        feature = aux["fused_roi"][0].float()
        matching_maps = None
        if args.matching_stats:
            matching_maps = hard_local_matching_maps(
                aux["rgb_feature"].float(),
                aux["sar_feature"].float(),
                window_size=args.matching_window,
                temperature=args.matching_temperature,
            )
        # teacher channel zero is (raw - ImageNet_mean[0]) / ImageNet_std[0].
        raw_sar = (post[0, 0].float().cpu() * float(IMAGENET_STD[0])) + float(
            IMAGENET_MEAN[0]
        )
        scene_stats = torch.tensor(
            [float(raw_sar.mean()), float(raw_sar.std(unbiased=False))],
            dtype=torch.float32,
        )
        for component in connected_components_4(building_mask):
            area = int(component.sum())
            if area < args.min_area or area > args.max_area:
                continue
            pooled = _pool_component(
                feature,
                probability,
                raw_sar,
                scene_stats,
                target,
                component,
                matching_maps=matching_maps,
            )
            if pooled is None:
                continue
            if (
                float(pooled["coverage"]) < args.min_building_coverage
                or float(pooled["purity"]) < args.min_class_purity
            ):
                continue
            pooled["image_id"] = name
            pooled["event"] = event_from_name(name)
            records.append(pooled)
        if index % 50 == 0 or index == len(loader):
            print(f"RM {args.role} cache: {index}/{len(loader)} | regions={len(records)}", flush=True)

    if not records:
        raise RuntimeError("no eligible P3 building regions were extracted")
    tensors = {
        "keys": torch.stack([item["key"] for item in records]),
        "labels": torch.tensor([item["label"] for item in records], dtype=torch.long),
        "base_predictions": torch.tensor(
            [item["base_prediction"] for item in records], dtype=torch.long
        ),
        "base_probabilities": torch.stack(
            [item["base_probability"] for item in records]
        ),
        "raw_stats": torch.stack([item["raw_stats"] for item in records]),
        "scene_stats": torch.stack([item["scene_stats"] for item in records]),
        "areas": torch.tensor([item["area"] for item in records], dtype=torch.long),
        "coverage": torch.tensor([item["coverage"] for item in records]),
        "purity": torch.tensor([item["purity"] for item in records]),
        "bboxes": torch.tensor([item["bbox"] for item in records], dtype=torch.long),
    }
    if args.matching_stats:
        tensors["colocated_stats"] = torch.stack(
            [item["colocated_stats"] for item in records]
        )
        tensors["matching_stats"] = torch.stack(
            [item["matching_stats"] for item in records]
        )
    strings = {
        "image_ids": [str(item["image_id"]) for item in records],
        "events": [str(item["event"]) for item in records],
    }
    counts_before = torch.bincount(tensors["labels"], minlength=3)
    class_event_before = class_event_count_table(tensors["labels"], strings["events"])
    if args.role == "memory":
        if args.balance_mode == "class_event":
            selected = class_event_balanced_indices(
                tensors["labels"],
                strings["events"],
                args.capacity_per_class,
                args.capacity_per_event_class,
                args.seed,
            )
        else:
            selected = balanced_indices(
                tensors["labels"], args.capacity_per_class, args.seed
            )
        tensors = {key: value[selected] for key, value in tensors.items()}
        selected_list = selected.tolist()
        strings = {
            key: [values[i] for i in selected_list] for key, values in strings.items()
        }
    tensors["relative_raw_stats"] = scene_relative_raw_stats(
        tensors["raw_stats"], tensors["scene_stats"]
    )
    normalizer = raw_stat_normalizer(tensors["raw_stats"])
    relative_normalizer = raw_stat_normalizer(tensors["relative_raw_stats"])
    colocated_normalizer = (
        vector_normalizer(tensors["colocated_stats"])
        if args.matching_stats
        else None
    )
    matching_normalizer = (
        vector_normalizer(tensors["matching_stats"])
        if args.matching_stats
        else None
    )
    class_event_after = class_event_count_table(tensors["labels"], strings["events"])
    metadata = {
        "protocol": "RM_region_cache_v3" if args.matching_stats else "RM_region_cache_v2",
        "role": args.role,
        "source_split": (
            "train" if args.role in ("memory", "calibration") else "val"
        ),
        "source_list": str(Path(args.data_list_path).resolve()),
        "source_list_sha256": sha256(args.data_list_path),
        "p3_checkpoint": str(Path(args.p3_checkpoint).resolve()),
        "p3_checkpoint_sha256": sha256(args.p3_checkpoint),
        "region_source": "frozen_p3_connected_components",
        "feature_source": "frozen_p3_fused_roi",
        "min_area": args.min_area,
        "max_area": args.max_area,
        "min_building_coverage": args.min_building_coverage,
        "min_class_purity": args.min_class_purity,
        "class_counts_before_balance": counts_before.tolist(),
        "class_counts": torch.bincount(tensors["labels"], minlength=3).tolist(),
        "balance_mode": args.balance_mode if args.role == "memory" else "none",
        "capacity_per_class": args.capacity_per_class,
        "capacity_per_event_class": args.capacity_per_event_class,
        "class_event_counts_before_balance": class_event_before,
        "class_event_counts": class_event_after,
        "raw_stat_normalizer": normalizer,
        "relative_raw_stat_normalizer": relative_normalizer,
        "relative_raw_stat_definition": "[(region_mean-scene_mean)/scene_std, region_std/scene_std]",
        "matching_stats_enabled": bool(args.matching_stats),
        "matching_contract": (
            "hard local cosine response peak; dominant translation residual; no warping"
            if args.matching_stats
            else "disabled"
        ),
        "matching_window": args.matching_window if args.matching_stats else None,
        "matching_temperature": args.matching_temperature if args.matching_stats else None,
        "colocated_stat_names": list(COLOCATED_STAT_NAMES) if args.matching_stats else [],
        "matching_stat_names": list(MATCHING_STAT_NAMES) if args.matching_stats else [],
        "colocated_stat_normalizer": colocated_normalizer,
        "matching_stat_normalizer": matching_normalizer,
        "test_accessed": False,
    }
    return {**tensors, **strings, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser("Build RM0 frozen-P3 region cache")
    parser.add_argument(
        "--role", choices=("memory", "calibration", "query"), required=True
    )
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
    parser.add_argument("--balance_mode", choices=("class", "class_event"), default="class")
    parser.add_argument("--capacity_per_event_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--matching_stats", action="store_true")
    parser.add_argument("--matching_window", type=int, default=5)
    parser.add_argument("--matching_temperature", type=float, default=0.10)
    args = parser.parse_args()
    payload = extract(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    manifest = output.with_suffix(".json")
    manifest.write_text(
        json.dumps(payload["metadata"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 88)
    print(f"Role           : {args.role}")
    print(f"Regions        : {payload['keys'].shape[0]}")
    print(f"Class counts   : {payload['metadata']['class_counts']}")
    print(f"Balance mode   : {payload['metadata']['balance_mode']}")
    print(f"P3 source      : {args.p3_checkpoint}")
    print(f"Saved          : {output}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
