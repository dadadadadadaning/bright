# -*- coding: utf-8 -*-
"""Freeze P3-predicted component identities for isolated CMR-V1.

This is the only V1 stage allowed to replay connected components.  A component
must pass the existing strict identity checks before its compact predicted mask
is persisted.  Unresolved rows are quarantined and reported; they are never
silently substituted with a nearby building or a GT region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import P4NativeSARDataset
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from model.DamageFormer_S8_MICB_DSMIL import connected_components_4
from model.cmr.pooling import build_cmr_feature_maps
from model.cmr_v1.region_cache import REGION_CACHE_PROTOCOL, pack_component
from model.cmr_v1.replay import build_component_index, replay_cached_component


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(p0_path: str, candidate_path: str) -> tuple[dict, dict]:
    p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if p0.get("metadata", {}).get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("region cache requires a frozen CMR-P0 cache")
    if candidate.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("region cache requires frozen Raw-Mean Top-K candidates")
    if bool(p0["metadata"].get("test_accessed", True)) or bool(candidate.get("test_accessed", True)):
        raise RuntimeError("region cache refuses Test-derived inputs")
    if int(candidate["query_labels"].numel()) != int(p0["labels"].numel()):
        raise RuntimeError("P0 and candidate rows do not align")
    return p0, candidate


def selected_rows(role: str, p0: dict, candidate: dict) -> torch.Tensor:
    if role == "support":
        memory_labels = candidate["memory_labels"].long()
        # Freeze all three classes now so later D-vs-X work does not require a
        # second component-replay protocol.
        memory_local = torch.nonzero(
            (memory_labels >= 0) & (memory_labels <= 2), as_tuple=False
        ).flatten()
        return candidate["memory_indices_in_train"].long()[memory_local]
    labels = candidate["candidate_labels"].long()
    # Stage-1 proposal is label-free: an Intact P3 prediction with at least one
    # affected (Damaged or Destroyed) counterexample in the frozen Top-K.
    has_affected = ((labels == 1) | (labels == 2)).any(dim=1)
    proposal = (p0["base_predictions"].long() == 0) & has_affected
    return torch.nonzero(proposal, as_tuple=False).flatten()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("CMR-V1 frozen predicted-region cache")
    parser.add_argument("--role", choices=("support", "train_query", "validation_query"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--p0_cache", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_bbox_iou", type=float, default=0.90)
    parser.add_argument("--min_area_ratio", type=float, default=0.90)
    parser.add_argument("--min_key_cosine", type=float, default=0.995)
    parser.add_argument("--min_key_margin", type=float, default=0.001)
    parser.add_argument("--warn_verified_fraction", type=float, default=0.02)
    parser.add_argument("--max_verified_fraction", type=float, default=0.05)
    parser.add_argument("--max_unresolved_fraction", type=float, default=0.005)
    args = parser.parse_args()

    p0, candidate = load_inputs(args.p0_cache, args.candidate_cache)
    expected_p0_role = "train" if args.role in {"support", "train_query"} else "query"
    expected_candidate_role = "train" if args.role in {"support", "train_query"} else "validation"
    if p0["metadata"].get("role") != expected_p0_role:
        raise RuntimeError(f"{args.role} requires P0 role {expected_p0_role}")
    if candidate.get("role") != expected_candidate_role:
        raise RuntimeError(f"{args.role} requires candidate role {expected_candidate_role}")
    checkpoint_hash = sha256(args.p3_checkpoint)
    if checkpoint_hash != p0["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("region-cache P3 checkpoint hash differs from P0")

    wanted = selected_rows(args.role, p0, candidate).long()
    if wanted.numel() == 0 or int(torch.unique(wanted).numel()) != int(wanted.numel()):
        raise RuntimeError("region cache selected zero or duplicate P0 rows")
    rows_by_image: dict[str, list[int]] = defaultdict(list)
    for source_row in wanted.tolist():
        rows_by_image[str(p0["image_ids"][source_row])].append(int(source_row))

    dataset = P4NativeSARDataset(
        dataset_path=args.dataset_path,
        data_list=sorted(rows_by_image),
        crop_size=args.crop_size,
        split="cache",
        variant="a0",
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    device = torch.device(args.device)
    model = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(model, args.p3_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    mode_counts = {"exact_bbox_area": 0, "verified_geometry_plus_rawmean": 0}
    for image_index, batch in enumerate(loader, 1):
        name = str(batch["name"][0])
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        output_loc, _, aux = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        # One connected-component pass per image.  The previous implementation
        # repeated this CPU operation once per selected building, which made
        # dense scenes appear GPU-idle for long periods.
        components = connected_components_4(building_mask)
        component_index = build_component_index(components)
        maps = build_cmr_feature_maps(
            model.micb_dsmil, aux["rgb_feature"], aux["sar_feature"], aux["fused_roi"]
        )
        fused = maps["fused_feature"][0].float()

        def current_mean_key(component):
            mask = torch.from_numpy(component.astype("float32", copy=False)).to(fused.device)
            weights = F.interpolate(mask[None, None], size=fused.shape[-2:], mode="area")[0, 0].clamp(0, 1)
            mean = (fused * weights[None]).sum(dim=(1, 2)) / weights.sum().clamp_min(1e-6)
            return F.normalize(mean, dim=0, eps=1e-8).detach().cpu()

        for source_row in rows_by_image[name]:
            match = replay_cached_component(
                building_mask,
                p0["bboxes"][source_row].tolist(),
                int(p0["areas"][source_row]),
                p0["key_variants"]["mean"][source_row].float(),
                current_mean_key,
                precomputed_components=components,
                exact_index=component_index,
                min_bbox_iou=args.min_bbox_iou,
                min_area_ratio=args.min_area_ratio,
                min_key_cosine=args.min_key_cosine,
                min_key_margin=args.min_key_margin,
            )
            if match is None:
                unresolved.append({
                    "source_row": int(source_row), "image_id": name,
                    "cached_bbox": p0["bboxes"][source_row].tolist(),
                    "cached_area": int(p0["areas"][source_row]),
                    "reason": "no_unique_identity_safe_component",
                })
                continue
            packed, bbox, image_shape = pack_component(match.component)
            mode_counts[match.mode] += 1
            records.append({
                "source_row": int(source_row), "image_id": name,
                "packed": packed, "bbox": bbox, "image_shape": image_shape,
                "area": int(match.component.sum()), "match": match,
            })
        if image_index % 50 == 0 or image_index == len(loader):
            print(
                f"CMR-V1 region {args.role}: {image_index}/{len(loader)} | "
                f"frozen={len(records)} unresolved={len(unresolved)}", flush=True,
            )

    selected_count = int(wanted.numel())
    unresolved_fraction = float(len(unresolved)) / float(selected_count)
    verified_fraction = (
        float(mode_counts["verified_geometry_plus_rawmean"]) / float(len(records))
        if records else 0.0
    )
    verified_matches = [
        item["match"] for item in records
        if item["match"].mode == "verified_geometry_plus_rawmean"
    ]

    def metric_summary(attribute: str) -> dict[str, float | None]:
        values = torch.tensor(
            [float(getattr(match, attribute)) for match in verified_matches],
            dtype=torch.float32,
        )
        if values.numel() == 0:
            return {"minimum": None, "p05": None, "median": None}
        return {
            "minimum": float(values.min()),
            "p05": float(torch.quantile(values, 0.05)),
            "median": float(values.median()),
        }

    safety_diagnostics = {
        "protocol": "CMR_V1_ISOLATED_region_identity_safety_diagnostic_v1",
        "role": args.role,
        "selected_source_rows": selected_count,
        "frozen_rows": len(records),
        "unresolved_rows": len(unresolved),
        "unresolved_fraction": unresolved_fraction,
        "mode_counts": mode_counts,
        "verified_fraction": verified_fraction,
        "warning_fraction": float(args.warn_verified_fraction),
        "hard_stop_fraction": float(args.max_verified_fraction),
        "verified_quality": {
            "bbox_iou": metric_summary("bbox_iou"),
            "area_ratio": metric_summary("area_ratio"),
            "key_cosine": metric_summary("key_cosine"),
            "key_margin": metric_summary("key_margin"),
        },
        "identity_thresholds_unchanged": {
            "min_bbox_iou": float(args.min_bbox_iou),
            "min_area_ratio": float(args.min_area_ratio),
            "min_key_cosine": float(args.min_key_cosine),
            "min_key_margin": float(args.min_key_margin),
        },
        "unresolved_details": unresolved,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_output = output.with_suffix(".identity_diagnostics.json")
    diagnostic_output.write_text(
        json.dumps(safety_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not records:
        raise RuntimeError("region-cache build froze no regions")
    if verified_fraction > float(args.warn_verified_fraction):
        print(
            f"WARNING: verified replay fraction {verified_fraction:.4%} exceeds "
            f"the diagnostic warning line {args.warn_verified_fraction:.4%}; "
            f"per-instance identity thresholds remain unchanged. "
            f"See {diagnostic_output}",
            flush=True,
        )
    if verified_fraction > float(args.max_verified_fraction):
        raise RuntimeError(
            f"verified replay fraction {verified_fraction:.4%} exceeds "
            f"hard safety limit {args.max_verified_fraction:.4%}; "
            f"inspect {diagnostic_output} before changing any threshold"
        )
    if unresolved_fraction > float(args.max_unresolved_fraction):
        raise RuntimeError(
            f"unresolved fraction {unresolved_fraction:.4%} exceeds "
            f"safety limit {args.max_unresolved_fraction:.4%}; no cache was published"
        )

    records.sort(key=lambda item: int(item["source_row"]))
    payload = {
        "protocol": REGION_CACHE_PROTOCOL,
        "role": args.role,
        "source_region_indices": torch.tensor([item["source_row"] for item in records], dtype=torch.long),
        "image_ids": [str(item["image_id"]) for item in records],
        "bboxes": torch.tensor([item["bbox"] for item in records], dtype=torch.long),
        "areas": torch.tensor([item["area"] for item in records], dtype=torch.long),
        "image_shapes": torch.tensor([item["image_shape"] for item in records], dtype=torch.long),
        "packed_component_masks": [item["packed"] for item in records],
        "identity_mode": [item["match"].mode for item in records],
        "identity_key_cosine": torch.tensor([item["match"].key_cosine for item in records]),
        "identity_bbox_iou": torch.tensor([item["match"].bbox_iou for item in records]),
        "selected_source_rows": selected_count,
        "unresolved_rows": unresolved,
        "unresolved_fraction": unresolved_fraction,
        "mode_counts": mode_counts,
        "verified_fraction": verified_fraction,
        "p0_protocol": p0["metadata"]["protocol"],
        "p0_source_list_sha256": p0["metadata"].get("source_list_sha256"),
        "p3_checkpoint_sha256": checkpoint_hash,
        "region_source": "frozen_p3_predicted_connected_component_not_gt",
        "test_accessed": False,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"Saved              : {output}")
    print(f"Frozen/unresolved  : {len(records)}/{len(unresolved)}")
    print(f"Exact/verified     : {mode_counts['exact_bbox_area']}/{mode_counts['verified_geometry_plus_rawmean']}")
    print("Test accessed      : False")


if __name__ == "__main__":
    main()
