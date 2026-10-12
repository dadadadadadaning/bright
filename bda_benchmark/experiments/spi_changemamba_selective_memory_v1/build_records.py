#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent, sha256_file
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    event_name, parse_batch, read_list,
)
from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import (
    make_region_ids, recover_sar_display, split_regions_by_building,
)
from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import (
    confusion, force_occurrence_predictions, valid_miou,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist

from . import PROTOCOL
from .core import extract_region_groups
from .model import forward_features


def load_changemamba(args, device: torch.device):
    root = Path(args.changemamba_root).resolve()
    if not (root / "changedetection").is_dir():
        raise RuntimeError(f"invalid ChangeMamba root: {root}")
    sys.path.insert(0, str(root))
    from changedetection.checkpoints import load_model_weights
    from changedetection.configs.config import get_config
    from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
    from changedetection.script.script_utils import get_vssm_kwargs

    args.opts = None
    config = get_config(args)
    model = ChangeMambaMMBDA(
        output_building=2, output_damage=4, pretrained=None,
        **get_vssm_kwargs(config),
    ).to(device).eval()
    report = load_model_weights(model, args.changemamba_ckpt)
    if report["missing_keys"] or report["unexpected_keys"] or report["mismatched_keys"]:
        raise RuntimeError(f"strict ChangeMamba checkpoint load failed: {report}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, report


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Build frozen P3 + ChangeMamba region records")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--p3_ckpt", required=True)
    parser.add_argument("--changemamba_root", required=True)
    parser.add_argument("--changemamba_ckpt", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--sar_stats_json", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region_mode", choices=("slico", "slic", "grid"), default="slico")
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--minimum_area", type=int, default=4)
    parser.add_argument("--minimum_building", type=float, default=0.50)
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
    changemamba, cm_report = load_changemamba(args, device)
    dataset = MultimodalDamageAssessmentDatset(
        args.dataset_path, names, crop_size=1024, type="test"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    stats = None
    if args.sar_stats_json:
        stats = json.loads(Path(args.sar_stats_json).read_text(encoding="utf-8"))

    group_rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    soft_labels: list[torch.Tensor] = []
    labels: list[int] = []
    purities: list[float] = []
    areas: list[int] = []
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

    processed = 0
    for batch in loader:
        pre, post, location, damage, sample_names = parse_batch(batch, device)
        loc_logits, p3_logits, aux = p3(pre, post, return_aux=True)
        # Retain only the small P3 tensors needed downstream before running
        # ChangeMamba.  This avoids keeping the complete P3 auxiliary feature
        # pyramid alive beside the ChangeMamba activations on a 24-GiB GPU.
        p_building = aux["p_building"].detach()
        del aux
        cm = forward_features(changemamba, pre, post)
        p3_logits_full = F.interpolate(
            p3_logits.float(), damage.shape[-2:], mode="bilinear", align_corners=False
        )
        loc_logits_full = F.interpolate(
            loc_logits.float(), location.shape[-2:], mode="bilinear", align_corners=False
        )
        p3_prediction = p3_logits_full.argmax(1)
        loc_prediction = loc_logits_full.argmax(1)
        raw_sar = recover_sar_display(post, stats)
        feature_size = cm["change_local"].shape[-2:]
        base_ids = make_region_ids(
            pre, feature_size, mode=args.region_mode,
            n_segments=args.segments, compactness=args.compactness,
        )
        ids = split_regions_by_building(base_ids, p_building)
        for b, name_value in enumerate(sample_names):
            name = str(name_value)
            event = event_name(name)
            local_base = fast_hist(p3_prediction[b], damage[b], 4, 255).cpu().double()
            base_hist += local_base
            event_base[event] += local_base
            loc_hist += fast_hist(loc_prediction[b], location[b], 2, 255).cpu().double()
            rows = extract_region_groups(
                rgb_structure=cm["rgb_structure"][b],
                change_local=cm["change_local"][b],
                change_semantic=cm["change_semantic"][b],
                raw_sar=raw_sar[b, 0], ids=ids[b],
                p_building=p_building[b, 0],
                p3_probability=F.softmax(p3_logits[b].float(), dim=0),
                target=damage[b], minimum_area=args.minimum_area,
                minimum_building=args.minimum_building,
            )
            ids_full = F.interpolate(
                ids[b:b + 1, None].float(), size=damage.shape[-2:], mode="nearest"
            )[0, 0].long()
            building_full = F.interpolate(
                p_building[b:b + 1].float(), size=damage.shape[-2:],
                mode="bilinear", align_corners=False,
            )[0, 0]
            for row in rows:
                mask = (
                    (ids_full == row["region_id"])
                    & (building_full >= args.minimum_building)
                    & (damage[b] != 255)
                )
                if not bool(mask.any()):
                    continue
                logits = p3_logits_full[b, :, mask].transpose(0, 1)
                target = damage[b, mask]
                base = logits.argmax(1)
                for group_name, value in row["groups"].items():
                    group_rows[group_name].append(value.cpu())
                soft_labels.append(row["soft_label"].cpu())
                labels.append(row["label"])
                purities.append(row["purity"])
                areas.append(row["area"])
                events.append(event)
                images.append(name)
                region_ids.append(row["region_id"])
                p3_region_labels.append(row["p3_region_label"])
                p3_uncertainties.append(row["p3_uncertainty"])
                delta_occ0.append(
                    confusion(force_occurrence_predictions(logits, False), target)
                    - confusion(base, target)
                )
                delta_occ1.append(
                    confusion(force_occurrence_predictions(logits, True), target)
                    - confusion(base, target)
                )
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
        "allowed_for_verifier_training": args.split == "train",
        "groups": {name: torch.stack(values) for name, values in group_rows.items()},
        "soft_labels": torch.stack(soft_labels),
        "labels": torch.tensor(labels, dtype=torch.long),
        "purities": torch.tensor(purities),
        "areas": torch.tensor(areas, dtype=torch.long),
        "events": events, "images": images,
        "region_ids": torch.tensor(region_ids, dtype=torch.long),
        "p3_region_labels": torch.tensor(p3_region_labels, dtype=torch.long),
        "p3_uncertainties": torch.tensor(p3_uncertainties),
        "delta_occurrence_intact": torch.stack(delta_occ0),
        "delta_occurrence_affected": torch.stack(delta_occ1),
        "base_hist": base_hist, "loc_hist": loc_hist,
        "event_base_hist": dict(event_base),
        "metadata": {
            "data_list": str(Path(args.data_list).resolve()),
            "data_list_sha256": sha256_file(args.data_list),
            "image_count": len(names), "region_mode": args.region_mode,
            "segments": args.segments, "compactness": args.compactness,
            "p3_checkpoint": str(Path(args.p3_ckpt).resolve()),
            "changemamba_checkpoint": str(Path(args.changemamba_ckpt).resolve()),
            "p3_miou": p3_miou, "p3_load": p3_report,
            "changemamba_load": cm_report,
            "same_regions_for_all_variants": True,
            "gt_role": "record label and metric-aligned utility only",
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
