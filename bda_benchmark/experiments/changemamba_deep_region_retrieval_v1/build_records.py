#!/usr/bin/env python3
"""Build pure-ChangeMamba region records from Train or Val only.

Unlike the earlier SRAC-mechanism cache, this file never loads P3.  Region
splitting, baseline logits and region state all come from the same frozen B2
ChangeMamba checkpoint.  GT is used only after region formation to attach
training/audit labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import event_name, parse_batch, read_list
from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.core import region_geometry
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    _p3_state as region_state, make_region_ids, split_regions_by_building,
)
from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import (
    confusion, force_occurrence_predictions, valid_miou,
)
from bda_benchmark.experiments.spi_srac_mechanism_ablation_v1 import PROTOCOL as CACHE_PROTOCOL
from bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.b2_features import (
    forward_b2_features, load_b2_model,
)
from bda_benchmark.experiments.spi_srac_mechanism_ablation_v1.core import sample_region_evidence
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist

from . import PROTOCOL


def resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False).to(value.dtype)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Pure ChangeMamba region record builder")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--changemamba_root", required=True)
    parser.add_argument("--changemamba_ckpt", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region_mode", choices=("grid", "slico", "slic"), required=True)
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--minimum_area", type=int, default=4)
    parser.add_argument("--minimum_building", type=float, default=0.50)
    parser.add_argument("--max_region_tokens", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--expected_images", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.data_list).name.lower():
        raise RuntimeError("Test access is forbidden by this protocol")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = read_list(args.data_list)
    if args.max_images:
        names = names[: args.max_images]
    if args.expected_images and len(names) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(names)}")
    model, load_report = load_b2_model(args, device)
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=1024, type="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    token_rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    token_masks: list[torch.Tensor] = []
    base_states: list[torch.Tensor] = []
    geometries: list[torch.Tensor] = []
    soft_labels: list[torch.Tensor] = []
    labels: list[int] = []
    events: list[str] = []
    images: list[str] = []
    region_ids: list[int] = []
    base_region_labels: list[int] = []
    base_uncertainties: list[float] = []
    delta_occ0: list[torch.Tensor] = []
    delta_occ1: list[torch.Tensor] = []
    damage_hist = torch.zeros(4, 4, dtype=torch.float64)
    loc_hist = torch.zeros(2, 2, dtype=torch.float64)
    event_hist = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    processed = 0
    gamma_raw = gamma_effective = None

    for batch in loader:
        pre, post, location, damage, sample_names = parse_batch(batch, device)
        feat = forward_b2_features(model, pre, post)
        gamma_raw = float(feat.pop("diff_gamma_raw")[0])
        gamma_effective = float(feat.pop("diff_gamma_effective")[0])
        feature_size = feat["base_change"].shape[-2:]
        feat = {key: resize(value, feature_size) for key, value in feat.items()}
        building_logits_small = model.aux_clf(feat["rgb_structure"])
        damage_logits_small = model.main_clf(feat["enhanced_change"])
        building_probability = building_logits_small.float().softmax(1)[:, 1:2]
        base_ids = make_region_ids(pre, feature_size, mode=args.region_mode,
                                   n_segments=args.segments, compactness=args.compactness)
        ids = split_regions_by_building(base_ids, building_probability)
        target_small = F.interpolate(damage[:, None].float(), size=feature_size, mode="nearest")[:, 0].long()
        probability_small = damage_logits_small.float().softmax(1)
        damage_logits_full = resize(damage_logits_small, damage.shape[-2:])
        building_logits_full = resize(building_logits_small, location.shape[-2:])
        damage_prediction = damage_logits_full.argmax(1)
        location_prediction = building_logits_full.argmax(1)

        for sample_index, sample_name in enumerate(sample_names):
            image_name = str(sample_name)
            event = event_name(image_name)
            local_hist = fast_hist(damage_prediction[sample_index], damage[sample_index], 4, 255).cpu().double()
            damage_hist += local_hist
            event_hist[event] += local_hist
            loc_hist += fast_hist(location_prediction[sample_index], location[sample_index], 2, 255).cpu().double()
            ids_full = F.interpolate(ids[sample_index:sample_index + 1, None].float(),
                                     size=damage.shape[-2:], mode="nearest")[0, 0].long()
            building_full = resize(building_probability[sample_index:sample_index + 1], damage.shape[-2:])[0, 0]
            feature_maps = {key: value[sample_index] for key, value in feat.items()}
            for rid in torch.unique(ids[sample_index]).tolist():
                region_mask = ids[sample_index] == int(rid)
                if int(region_mask.sum()) < args.minimum_area:
                    continue
                if float(building_probability[sample_index, 0][region_mask].mean()) < args.minimum_building:
                    continue
                valid = region_mask & (target_small[sample_index] != 255)
                foreground = valid & (target_small[sample_index] >= 1) & (target_small[sample_index] <= 3)
                if not bool(foreground.any()):
                    continue
                counts = torch.stack([(target_small[sample_index][foreground] == cls).sum()
                                      for cls in (1, 2, 3)]).float()
                soft = counts / counts.sum().clamp_min(1.0)
                label = int(soft.argmax()) + 1
                evidence, evidence_mask = sample_region_evidence(feature_maps, region_mask, args.max_region_tokens)
                state, base_label, uncertainty = region_state(probability_small[sample_index], region_mask)
                geometry = region_geometry(region_mask, building_probability[sample_index, 0])
                full_mask = ((ids_full == int(rid)) & (building_full >= args.minimum_building)
                             & (damage[sample_index] != 255))
                if not bool(full_mask.any()):
                    continue
                logits = damage_logits_full[sample_index, :, full_mask].transpose(0, 1)
                target = damage[sample_index, full_mask]
                base = logits.argmax(1)
                for key in ("rgb_structure", "base_change", "explicit_diff"):
                    token_rows[key].append(evidence[key].half().cpu())
                token_masks.append(evidence_mask.cpu())
                base_states.append(state.cpu())
                geometries.append(geometry.cpu())
                soft_labels.append(soft.cpu())
                labels.append(label)
                events.append(event)
                images.append(image_name)
                region_ids.append(int(rid))
                base_region_labels.append(int(base_label))
                base_uncertainties.append(float(uncertainty))
                delta_occ0.append(confusion(force_occurrence_predictions(logits, False), target) - confusion(base, target))
                delta_occ1.append(confusion(force_occurrence_predictions(logits, True), target) - confusion(base, target))
        processed += len(sample_names)
        if processed % 25 < len(sample_names) or processed == len(names):
            print(f"records {processed}/{len(names)} regions={len(labels)}", flush=True)

    if not labels:
        raise RuntimeError("no valid ChangeMamba regions were produced")
    base_miou = valid_miou(damage_hist)
    payload = {
        # Compatibility with the already tested deep encoder/RAG modules.
        "protocol": CACHE_PROTOCOL + "_RECORDS",
        "owner_protocol": PROTOCOL,
        "split": args.split,
        "allowed_for_training": args.split == "train",
        "tokens": {key: torch.stack(rows) for key, rows in token_rows.items()},
        "token_mask": torch.stack(token_masks),
        "p3_state": torch.stack(base_states),
        "geometry": torch.stack(geometries),
        "soft_labels": torch.stack(soft_labels),
        "labels": torch.tensor(labels, dtype=torch.long),
        "events": events, "images": images,
        "region_ids": torch.tensor(region_ids, dtype=torch.long),
        # Compatibility aliases; metadata explicitly records ChangeMamba origin.
        "p3_region_labels": torch.tensor(base_region_labels, dtype=torch.long),
        "p3_uncertainties": torch.tensor(base_uncertainties),
        "delta_occurrence_intact": torch.stack(delta_occ0),
        "delta_occurrence_affected": torch.stack(delta_occ1),
        "base_hist": damage_hist, "loc_hist": loc_hist,
        "event_base_hist": dict(event_hist),
        "metadata": {
            "data_list": str(Path(args.data_list).resolve()),
            "data_list_sha256": sha256_file(args.data_list),
            "image_count": len(names), "region_mode": args.region_mode,
            "segments": args.segments, "max_region_tokens": args.max_region_tokens,
            "base_model": "ChangeMambaMMBDA_B2_DiffFusion",
            "base_checkpoint": str(Path(args.changemamba_ckpt).resolve()),
            "base_valid_miou": base_miou, "base_load": load_report,
            "building_prior_source": "ChangeMamba decoder_building + aux_clf",
            "damage_state_source": "ChangeMamba enhanced_change + main_clf",
            "gt_role": "labels/audit only; never region construction or retrieval key",
            "diff_gamma_raw": gamma_raw, "diff_gamma_effective": gamma_effective,
            "same_regions_for_all_feature_pooling_variants": True,
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} regions={len(labels)} ChangeMamba_valid_mIoU={base_miou:.8f}")


if __name__ == "__main__":
    main()

