# -*- coding: utf-8 -*-
"""Build frozen 256-D ``fused_roi`` case cards for isolated CMR-C0.

The script reads the already-frozen V1 component cache.  It never replays a
component, never reads Test, and never changes P3/RawMean candidate selection.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import P4NativeSARDataset
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from model.cmr_c0.case_card import (
    CASE_CACHE_PROTOCOL,
    CRITICAL_VARIANTS,
    extract_fused_case,
    incident_to_disaster_type,
    region_fingerprint,
    validate_case_cache,
)
from model.cmr_v1.region_cache import (
    source_row_lookup,
    unpack_component,
    validate_region_cache,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(p0_path: str, candidate_path: str) -> tuple[dict, dict]:
    p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if p0.get("metadata", {}).get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("CMR-C0 requires the frozen CMR-P0 cache")
    if candidate.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("CMR-C0 requires frozen RawMean Top-K candidates")
    if bool(p0["metadata"].get("test_accessed", True)) or bool(candidate.get("test_accessed", True)):
        raise RuntimeError("CMR-C0 refuses Test-derived inputs")
    if candidate["query_labels"].numel() != p0["labels"].numel():
        raise RuntimeError("P0 and candidate query rows do not align")
    return p0, candidate


def _selected_rows(
    role: str, p0: dict, candidate: dict
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if role.endswith("query"):
        # Frozen and label-free: P3 says Intact and RawMean Top-20 contains D.
        has_damaged = (candidate["candidate_labels"].long() == 1).any(dim=1)
        proposal = (p0["base_predictions"].long() == 0) & has_damaged
        return torch.nonzero(proposal, as_tuple=False).flatten(), None
    memory_labels = candidate["memory_labels"].long()
    memory_local = torch.nonzero(
        (memory_labels == 0) | (memory_labels == 1), as_tuple=False
    ).flatten()
    source_rows = candidate["memory_indices_in_train"].long()[memory_local]
    return source_rows, memory_local


def _pack_case_lists(cases: list[Dict[str, object]]) -> Dict[str, object]:
    tensor_fields = (
        "features", "xy", "normalized_xy", "damage_logits", "di_margin",
        "building_confidence", "component_weight", "is_boundary",
    )
    dtypes = {
        "features": torch.float16,
        "xy": torch.int16,
        "normalized_xy": torch.float16,
        "damage_logits": torch.float16,
        "di_margin": torch.float16,
        "building_confidence": torch.float16,
        "component_weight": torch.float16,
        "is_boundary": torch.bool,
    }
    packed: Dict[str, object] = {
        name: [case[name].to(dtypes[name]).contiguous() for case in cases]
        for name in tensor_fields
    }
    packed["critical_indices"] = {
        variant: [
            case["critical_indices"][variant].to(torch.int16).contiguous()
            for case in cases
        ]
        for variant in CRITICAL_VARIANTS
    }
    packed["nms_backfill"] = {
        variant: torch.tensor(
            [int(case["nms_backfill"][variant]) for case in cases], dtype=torch.long
        )
        for variant in CRITICAL_VARIANTS
    }
    return packed


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("CMR-C0 frozen fused_roi critical case cache")
    parser.add_argument("--role", choices=("support", "train_query", "validation_query"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--p0_cache", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--region_cache", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mask_threshold", type=float, default=0.05)
    parser.add_argument("--critical_min", type=int, default=4)
    parser.add_argument("--critical_max", type=int, default=32)
    parser.add_argument("--nms_radius", type=float, default=1.5)
    args = parser.parse_args()

    p0, candidate = _load_inputs(args.p0_cache, args.candidate_cache)
    expected_p0_role = "query" if args.role == "validation_query" else "train"
    expected_candidate_role = "validation" if args.role == "validation_query" else "train"
    if p0["metadata"].get("role") != expected_p0_role:
        raise RuntimeError(f"CMR-C0 {args.role} requires P0 role {expected_p0_role}")
    if candidate.get("role") != expected_candidate_role:
        raise RuntimeError(f"CMR-C0 {args.role} requires {expected_candidate_role} candidates")
    observed_sha = sha256(args.p3_checkpoint)
    if observed_sha != p0["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("CMR-C0 P3 checkpoint hash differs from frozen P0")

    source_rows, memory_local = _selected_rows(args.role, p0, candidate)
    region = torch.load(args.region_cache, map_location="cpu", weights_only=False)
    validate_region_cache(region, args.role)
    if region.get("p3_checkpoint_sha256") != observed_sha:
        raise RuntimeError("CMR-C0 region cache and P0 use different P3 checkpoints")
    region_lookup = source_row_lookup(region, int(p0["labels"].numel()))
    available = region_lookup[source_rows] >= 0
    skipped_source_rows = source_rows[~available]
    if args.role == "support":
        assert memory_local is not None
        memory_local = memory_local[available]
    source_rows = source_rows[available]
    if source_rows.numel() == 0:
        raise RuntimeError("CMR-C0 found no identity-frozen selected regions")

    rows_by_image: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for output_row, source_row in enumerate(source_rows.tolist()):
        rows_by_image[str(p0["image_ids"][source_row])].append((output_row, source_row))
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

    cases: list[Dict[str, object] | None] = [None] * source_rows.numel()
    fingerprints: list[str | None] = [None] * source_rows.numel()
    for image_index, batch in enumerate(loader, 1):
        name = str(batch["name"][0])
        pre = batch["pre_img"].float().to(device, non_blocking=True)
        post = batch["teacher_post_img"].float().to(device, non_blocking=True)
        _, _, auxiliary = model(pre, post, return_aux=True)
        # The accepted P3 has residual scale 0.  This is precisely the frozen
        # same-position damage head that produced its pixel prediction.
        damage_logits_feature = model.clf_2(auxiliary["fused_roi"])
        for output_row, source_row in rows_by_image[name]:
            region_row = int(region_lookup[source_row])
            packed_mask = region["packed_component_masks"][region_row]
            bbox = region["bboxes"][region_row].tolist()
            image_shape = region["image_shapes"][region_row].tolist()
            component = unpack_component(packed_mask, bbox, image_shape)
            cases[output_row] = extract_fused_case(
                auxiliary["fused_roi"], auxiliary["p_building"],
                damage_logits_feature, component,
                mask_threshold=args.mask_threshold,
                minimum=args.critical_min, maximum=args.critical_max,
                nms_radius=args.nms_radius,
            )
            fingerprints[output_row] = region_fingerprint(packed_mask, bbox, image_shape)
        if image_index % 50 == 0 or image_index == len(loader):
            complete = sum(item is not None for item in cases)
            print(
                f"CMR-C0 {args.role}: images={image_index}/{len(loader)} "
                f"cases={complete}/{len(cases)}",
                flush=True,
            )
    if any(case is None for case in cases) or any(value is None for value in fingerprints):
        raise RuntimeError("CMR-C0 case extraction left unmatched rows")
    typed_cases = [case for case in cases if case is not None]
    packed = _pack_case_lists(typed_cases)
    token_count = torch.tensor(
        [int(case["features"].shape[0]) for case in typed_cases], dtype=torch.long
    )

    payload: Dict[str, object] = {
        "protocol": CASE_CACHE_PROTOCOL,
        "role": args.role,
        "feature_source": "frozen_p3_fused_roi_256d_same_position_clf2",
        "selection_source": "frozen_clf2_logit_margin_damaged_minus_intact",
        "selection_is_label_free": True,
        "selection_parameters": {
            "mask_threshold": float(args.mask_threshold),
            "critical_min": int(args.critical_min),
            "critical_max": int(args.critical_max),
            "nms_radius": float(args.nms_radius),
            "variants": list(CRITICAL_VARIANTS),
        },
        "source_region_indices": source_rows.long(),
        "token_count": token_count,
        **packed,
        "labels": p0["labels"][source_rows].long(),  # diagnostic only
        "rawmean_keys": p0["keys"][source_rows].float(),
        "base_predictions": p0["base_predictions"][source_rows].long(),
        "base_probabilities": p0["base_probabilities"][source_rows].float(),
        "events": [str(p0["events"][row]) for row in source_rows.tolist()],
        "disaster_types": [
            incident_to_disaster_type(str(p0["events"][row]))
            for row in source_rows.tolist()
        ],
        "image_ids": [str(p0["image_ids"][row]) for row in source_rows.tolist()],
        "bboxes": region["bboxes"][region_lookup[source_rows]].long(),
        "areas": region["areas"][region_lookup[source_rows]].long(),
        "image_shapes": region["image_shapes"][region_lookup[source_rows]].long(),
        "identity_mode": [
            region["identity_mode"][int(region_lookup[row])]
            for row in source_rows.tolist()
        ],
        "region_fingerprints": [str(value) for value in fingerprints],
        "p3_checkpoint_sha256": observed_sha,
        "proposal_rule": "frozen_p3_intact_and_any_damaged_in_frozen_rawmean_top20",
        "proposal_rule_uses_gt": False,
        "frozen_region_cache": {
            "protocol": region["protocol"],
            "path": str(Path(args.region_cache).resolve()),
            "selected_regions": int(source_rows.numel()),
            "skipped_unresolved_source_rows": skipped_source_rows.long(),
            "component_replay_during_case_build": False,
        },
        "upstream": {
            "p0_cache": str(Path(args.p0_cache).resolve()),
            "candidate_cache": str(Path(args.candidate_cache).resolve()),
            "p3_checkpoint": str(Path(args.p3_checkpoint).resolve()),
        },
        "test_accessed": False,
    }
    if args.role == "support":
        assert memory_local is not None
        payload.update({
            "memory_local_indices": memory_local.long(),
            "memory_labels": candidate["memory_labels"][memory_local].long(),
            "memory_events": [
                str(candidate["memory_events"][row]) for row in memory_local.tolist()
            ],
        })
    else:
        payload.update({
            "candidate_indices": candidate["candidate_indices"][source_rows].long(),
            "candidate_labels": candidate["candidate_labels"][source_rows].long(),
            "candidate_scores": candidate["candidate_scores"][source_rows].float(),
        })
    validate_case_cache(payload, args.role)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"Saved            : {output}")
    print(f"Cases/tokens     : {len(typed_cases)}/{int(token_count.sum())}")
    print(f"Feature          : fused_roi 256-D")
    print(f"Critical variants: {', '.join(CRITICAL_VARIANTS)}")
    print(f"Region skipped   : {int(skipped_source_rows.numel())}")
    print("Test accessed    : False")


if __name__ == "__main__":
    main()
