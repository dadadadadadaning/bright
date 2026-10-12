#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent, sha256_file
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import event_name, parse_batch, read_list
from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.core import region_geometry
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    _p3_state, make_region_ids, split_regions_by_building,
)
from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import (
    confusion, force_occurrence_predictions, valid_miou,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist

from . import PROTOCOL
from .b2_features import forward_b2_features, load_b2_model
from .core import sample_region_evidence


def resize_map(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False).to(value.dtype)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Cache P3 regions and frozen B2 pixel evidence")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--p3_ckpt", required=True)
    parser.add_argument("--changemamba_root", required=True)
    parser.add_argument("--changemamba_ckpt", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region_mode", choices=("slico", "slic", "grid"), default="slico")
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--minimum_area", type=int, default=4)
    parser.add_argument("--minimum_building", type=float, default=0.50)
    parser.add_argument("--max_region_tokens", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--expected_images", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--expected_p3_miou", type=float, default=-1.0)
    parser.add_argument("--p3_tolerance_pp", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.data_list).name.lower():
        raise RuntimeError("Test access is forbidden")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = read_list(args.data_list)
    if args.max_images > 0:
        names = names[: args.max_images]
    if args.expected_images and len(names) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(names)}")

    p3 = DamageFormer_S8_MICB_DSMIL().to(device).eval()
    p3_report = load_parent(p3, args.p3_ckpt)
    for parameter in p3.parameters():
        parameter.requires_grad_(False)
    b2, b2_report = load_b2_model(args, device)
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=1024, type="test")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    token_rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    token_masks: list[torch.Tensor] = []
    p3_states: list[torch.Tensor] = []
    geometries: list[torch.Tensor] = []
    soft_labels: list[torch.Tensor] = []
    labels: list[int] = []
    events: list[str] = []
    images: list[str] = []
    region_ids: list[int] = []
    p3_region_labels: list[int] = []
    p3_uncertainties: list[float] = []
    delta_occ0: list[torch.Tensor] = []
    delta_occ1: list[torch.Tensor] = []
    base_hist = torch.zeros(4, 4, dtype=torch.float64)
    loc_hist = torch.zeros(2, 2, dtype=torch.float64)
    event_base = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    gamma_raw = gamma_effective = None

    processed = 0
    for batch in loader:
        pre, post, location, damage, sample_names = parse_batch(batch, device)
        loc_logits, p3_logits, aux = p3(pre, post, return_aux=True)
        p_building = aux["p_building"].detach()
        del aux
        b2_feature = forward_b2_features(b2, pre, post)
        gamma_raw = float(b2_feature.pop("diff_gamma_raw")[0])
        gamma_effective = float(b2_feature.pop("diff_gamma_effective")[0])
        feature_size = b2_feature["base_change"].shape[-2:]
        b2_feature = {
            name: resize_map(value, feature_size) for name, value in b2_feature.items()
        }
        base_ids = make_region_ids(
            pre, feature_size, mode=args.region_mode,
            n_segments=args.segments, compactness=args.compactness,
        )
        ids = split_regions_by_building(base_ids, p_building)
        p3_logits_full = F.interpolate(p3_logits.float(), damage.shape[-2:], mode="bilinear", align_corners=False)
        loc_logits_full = F.interpolate(loc_logits.float(), location.shape[-2:], mode="bilinear", align_corners=False)
        p3_prediction = p3_logits_full.argmax(1)
        loc_prediction = loc_logits_full.argmax(1)
        building_small = resize_map(p_building, feature_size)[:, 0]
        probability_small = resize_map(F.softmax(p3_logits.float(), dim=1), feature_size)
        target_small = F.interpolate(damage[:, None].float(), size=feature_size, mode="nearest")[:, 0].long()

        for batch_index, name_value in enumerate(sample_names):
            name = str(name_value)
            event = event_name(name)
            local_hist = fast_hist(p3_prediction[batch_index], damage[batch_index], 4, 255).cpu().double()
            base_hist += local_hist
            event_base[event] += local_hist
            loc_hist += fast_hist(loc_prediction[batch_index], location[batch_index], 2, 255).cpu().double()
            ids_full = F.interpolate(ids[batch_index:batch_index + 1, None].float(), size=damage.shape[-2:], mode="nearest")[0, 0].long()
            building_full = F.interpolate(p_building[batch_index:batch_index + 1].float(), size=damage.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            feature_maps = {name: value[batch_index] for name, value in b2_feature.items()}
            for region_id in torch.unique(ids[batch_index]).tolist():
                region_mask = ids[batch_index] == int(region_id)
                area = int(region_mask.sum())
                if area < args.minimum_area:
                    continue
                if float(building_small[batch_index][region_mask].mean()) < args.minimum_building:
                    continue
                valid = region_mask & (target_small[batch_index] != 255)
                foreground = valid & (target_small[batch_index] >= 1) & (target_small[batch_index] <= 3)
                if not bool(foreground.any()):
                    continue
                counts = torch.stack([
                    (target_small[batch_index][foreground] == value).sum() for value in (1, 2, 3)
                ]).float()
                soft = counts / counts.sum().clamp_min(1.0)
                label = int(soft.argmax()) + 1
                evidence, evidence_mask = sample_region_evidence(
                    feature_maps, region_mask, args.max_region_tokens
                )
                p3_state, p3_label, uncertainty = _p3_state(
                    probability_small[batch_index], region_mask
                )
                geometry = region_geometry(region_mask, building_small[batch_index])
                full_mask = (
                    (ids_full == int(region_id))
                    & (building_full >= args.minimum_building)
                    & (damage[batch_index] != 255)
                )
                if not bool(full_mask.any()):
                    continue
                logits = p3_logits_full[batch_index, :, full_mask].transpose(0, 1)
                target = damage[batch_index, full_mask]
                base = logits.argmax(1)
                for feature_name in ("rgb_structure", "base_change", "explicit_diff"):
                    token_rows[feature_name].append(evidence[feature_name].half().cpu())
                token_masks.append(evidence_mask.cpu())
                p3_states.append(p3_state.cpu())
                geometries.append(geometry.cpu())
                soft_labels.append(soft.cpu())
                labels.append(label)
                events.append(event)
                images.append(name)
                region_ids.append(int(region_id))
                p3_region_labels.append(int(p3_label))
                p3_uncertainties.append(float(uncertainty))
                delta_occ0.append(confusion(force_occurrence_predictions(logits, False), target) - confusion(base, target))
                delta_occ1.append(confusion(force_occurrence_predictions(logits, True), target) - confusion(base, target))
        processed += len(sample_names)
        if processed % 25 < len(sample_names) or processed == len(names):
            print(f"records {processed}/{len(names)} regions={len(labels)}", flush=True)

    p3_miou = valid_miou(base_hist)
    if args.expected_p3_miou >= 0:
        error_pp = 100.0 * (p3_miou - args.expected_p3_miou)
        if abs(error_pp) > args.p3_tolerance_pp:
            raise RuntimeError(
                f"P3 reproduction failed: expected={args.expected_p3_miou:.8f}, "
                f"observed={p3_miou:.8f}, error={error_pp:+.4f}pp"
            )
    payload = {
        "protocol": PROTOCOL + "_RECORDS",
        "split": args.split,
        "allowed_for_training": args.split == "train",
        "tokens": {name: torch.stack(rows) for name, rows in token_rows.items()},
        "token_mask": torch.stack(token_masks),
        "p3_state": torch.stack(p3_states),
        "geometry": torch.stack(geometries),
        "soft_labels": torch.stack(soft_labels),
        "labels": torch.tensor(labels, dtype=torch.long),
        "events": events,
        "images": images,
        "region_ids": torch.tensor(region_ids, dtype=torch.long),
        "p3_region_labels": torch.tensor(p3_region_labels, dtype=torch.long),
        "p3_uncertainties": torch.tensor(p3_uncertainties),
        "delta_occurrence_intact": torch.stack(delta_occ0),
        "delta_occurrence_affected": torch.stack(delta_occ1),
        "base_hist": base_hist,
        "loc_hist": loc_hist,
        "event_base_hist": dict(event_base),
        "metadata": {
            "data_list": str(Path(args.data_list).resolve()),
            "data_list_sha256": sha256_file(args.data_list),
            "image_count": len(names),
            "region_mode": args.region_mode,
            "segments": args.segments,
            "max_region_tokens": args.max_region_tokens,
            "p3_checkpoint": str(Path(args.p3_ckpt).resolve()),
            "b2_checkpoint": str(Path(args.changemamba_ckpt).resolve()),
            "p3_miou": p3_miou,
            "p3_load": p3_report,
            "b2_load": b2_report,
            "diff_gamma_raw": gamma_raw,
            "diff_gamma_effective": gamma_effective,
            "token_selection": "half top enhanced-change anomaly; half spatially representative",
            "same_regions_for_all_variants": True,
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} regions={len(labels)} P3_mIoU={p3_miou:.8f}")


if __name__ == "__main__":
    main()

