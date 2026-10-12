#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
for value in (str(ROOT), str(ROOT / "bda_benchmark")):
    if value not in sys.path:
        sys.path.insert(0, value)

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    event_name, multimodal_descriptor, parse_batch, read_list,
    reliable_token_targets, sha256_file,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL


def main() -> None:
    parser = argparse.ArgumentParser("Build reliable prototypes from the complete training set")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--purity", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if "test" in Path(args.train_list).name.lower() or "val" in Path(args.train_list).name.lower():
        raise RuntimeError("Memory construction accepts the training-list file only")
    torch.manual_seed(args.seed)
    names = read_list(args.train_list)
    destination = Path(args.output)
    expected_train_sha = sha256_file(args.train_list)
    expected_parent_sha = sha256_file(args.parent_ckpt)
    if destination.is_file():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        metadata = existing.get("metadata", {})
        existing_parent = metadata.get("parent", {})
        compatible = (
            metadata.get("train_list_sha256") == expected_train_sha
            and existing_parent.get("sha256") == expected_parent_sha
            and float(metadata.get("purity_threshold", -1.0)) == float(args.purity)
            and "projection_mean" in existing
            and "projection_components" in existing
            and int(metadata.get("projection_dimension", -1)) == 64
        )
        if not compatible:
            raise RuntimeError(
                "existing Memory was built from a different train list, parent, or purity; "
                "use a new MEMORY_PATH instead of overwriting it"
            )
        print(f"Verified and preserved compatible train-only Memory: {destination}")
        return
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormer_S8_MICB_DSMIL(
        roi_alpha=1.0, detach_building_prior=True, token_dim=128,
        mil_topk=8, max_tokens_per_bag=256, mil_residual_scale=0.0,
    ).to(device).eval()
    parent_report = load_parent(model, args.parent_ckpt)
    dataset = MultimodalDamageAssessmentDatset(
        args.dataset_path, names, crop_size=640, max_iters=None, type="val",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    features, labels, events, images, purities, token_counts, confidences = [], [], [], [], [], [], []
    class_counts: Counter[int] = Counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            pre, post, location, damage, sample_names = parse_batch(batch, device)
            _, _, aux = model(pre, post, return_aux=True)
            descriptor = multimodal_descriptor(aux["rgb_feature"], aux["sar_feature"])
            parent_probability = torch.softmax(model.clf_2(aux["fused_roi"]).float(), dim=1)
            target, reliable, purity = reliable_token_targets(damage, location, aux["p_building"], args.purity)
            for sample_index, sample_name in enumerate(sample_names):
                for label in (1, 2, 3):
                    mask = reliable[sample_index] & (target[sample_index] == label)
                    if not bool(mask.any()):
                        continue
                    token_feature = descriptor[sample_index].permute(1, 2, 0)[mask].float()
                    prototype = token_feature.mean(dim=0)
                    features.append(prototype.cpu())
                    labels.append(label)
                    events.append(event_name(str(sample_name)))
                    images.append(str(sample_name))
                    purities.append(float(purity[sample_index][mask].mean().item()))
                    token_counts.append(int(mask.sum().item()))
                    confidences.append(float(parent_probability[sample_index, label][mask].mean().item()))
                    class_counts[label] += 1
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"Memory build: {batch_index}/{len(loader)} batches, entries={len(features)}", flush=True)
    if not features:
        raise RuntimeError("no reliable training prototypes were found")
    raw_features = torch.stack(features).float()
    projection_mean = raw_features.mean(dim=0)
    centered = raw_features - projection_mean
    projection_dim = min(64, centered.shape[0] - 1, centered.shape[1])
    if projection_dim < 8:
        raise RuntimeError(f"too few Memory entries for a stable projection: {centered.shape[0]}")
    _, _, projection_components = torch.pca_lowrank(centered, q=projection_dim, center=False, niter=4)
    projected = F.normalize(centered @ projection_components, dim=1)
    payload = {
        "protocol": "SPI_BUILDING_PROTOTYPE_MEMORY_V1",
        "features": projected,
        "projection_mean": projection_mean,
        "projection_components": projection_components,
        "labels": torch.tensor(labels, dtype=torch.long),
        "events": events,
        "images": images,
        "purities": torch.tensor(purities, dtype=torch.float32),
        "token_counts": torch.tensor(token_counts, dtype=torch.long),
        "confidences": torch.tensor(confidences, dtype=torch.float32),
        "metadata": {
            "source": "complete train_set only",
            "train_list": str(Path(args.train_list).resolve()),
            "train_list_sha256": sha256_file(args.train_list),
            "train_image_count": len(names),
            "purity_threshold": args.purity,
            "entry_count": len(features),
            "raw_descriptor": "L2(RGB) + L2(SAR) + L2(abs(RGB-SAR))",
            "raw_dimension": int(raw_features.shape[1]),
            "projection": "train-only PCA",
            "projection_dimension": projection_dim,
            "seed": args.seed,
            "class_entry_counts": {str(key): value for key, value in sorted(class_counts.items())},
            "event_count": len(set(events)),
            "parent": parent_report,
            "validation_accessed": False,
            "test_accessed": False,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(f"Saved train-only Memory: {destination}")


if __name__ == "__main__":
    main()
