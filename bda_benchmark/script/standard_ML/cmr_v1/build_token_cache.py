# -*- coding: utf-8 -*-
"""Build label-free P3-Intact proposal and two-class support token caches for CMR-V1.

Differences from V0:
  1) a query is admitted when frozen P3 predicts Intact and Raw-Mean Top-K
     contains at least one Damaged case; Top-1 does NOT need to be Damaged;
  2) both Intact and Damaged train-memory support tokens are materialised;
  3) proposal selection does not use GT labels.  GT is stored only for audit.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import P4NativeSARDataset
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from model.cmr.pooling import (
    build_cmr_feature_maps,
    extract_cmr_local_tokens,
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
    if p0["metadata"].get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("V1 requires the frozen CMR-P0 cache")
    if candidate.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("V1 requires frozen Raw-Mean Top-K candidates")
    if bool(p0["metadata"].get("test_accessed", True)) or bool(candidate.get("test_accessed", True)):
        raise RuntimeError("V1 refuses Test-derived inputs")
    if candidate["query_labels"].numel() != p0["labels"].numel():
        raise RuntimeError("P0 and candidate query rows do not align")
    return p0, candidate


def _selected_rows(role: str, p0: dict, candidate: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    if role.endswith("query"):
        labels = candidate["candidate_labels"].long()
        has_damaged = (labels == 1).any(dim=1)
        proposal = (p0["base_predictions"].long() == 0) & has_damaged
        return torch.nonzero(proposal, as_tuple=False).flatten(), None

    # Materialise both sides of the counterfactual comparison from frozen Train Memory.
    memory_labels = candidate["memory_labels"].long()
    memory_local = torch.nonzero(
        (memory_labels == 0) | (memory_labels == 1), as_tuple=False
    ).flatten()
    source_rows = candidate["memory_indices_in_train"].long()[memory_local]
    return source_rows, memory_local


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("CMR-V1 two-class support/local-token cache")
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
    parser.add_argument(
        "--storage_layout", choices=("packed_list", "padded"), default="packed_list",
        help="packed_list avoids global max-token padding and is recommended for query caches",
    )
    args = parser.parse_args()

    p0, candidate = _load_inputs(args.p0_cache, args.candidate_cache)
    expected_p0_role = "query" if args.role == "validation_query" else "train"
    if p0["metadata"].get("role") != expected_p0_role:
        raise RuntimeError(f"V1 {args.role} requires P0 role {expected_p0_role}")
    expected_candidate_role = "validation" if args.role == "validation_query" else "train"
    if candidate.get("role") != expected_candidate_role:
        raise RuntimeError(f"V1 role expects {expected_candidate_role} candidates")
    observed_sha = sha256(args.p3_checkpoint)
    if observed_sha != p0["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("V1 P3 checkpoint hash differs from P0")

    source_rows, memory_local = _selected_rows(args.role, p0, candidate)
    if source_rows.numel() == 0:
        raise RuntimeError("V1 selected no regions")
    region = torch.load(args.region_cache, map_location="cpu", weights_only=False)
    validate_region_cache(region, args.role)
    if region.get("p3_checkpoint_sha256") != observed_sha:
        raise RuntimeError("V1 token/region caches use different P3 checkpoints")
    region_lookup = source_row_lookup(region, int(p0["labels"].numel()))
    available = region_lookup[source_rows] >= 0
    skipped_source_rows = source_rows[~available]
    if args.role == "support":
        memory_local = memory_local[available]
    source_rows = source_rows[available]
    if source_rows.numel() == 0:
        raise RuntimeError("V1 region cache contains none of the selected regions")

    rows_by_image = defaultdict(list)
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

    extracted: list[torch.Tensor | None] = [None] * source_rows.numel()
    for image_index, batch in enumerate(loader, 1):
        name = str(batch["name"][0])
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        _, _, aux = model(pre, post, return_aux=True)
        maps = build_cmr_feature_maps(
            model.micb_dsmil, aux["rgb_feature"], aux["sar_feature"], aux["fused_roi"]
        )
        for output_row, source_row in rows_by_image[name]:
            region_row = int(region_lookup[source_row])
            component = unpack_component(
                region["packed_component_masks"][region_row],
                region["bboxes"][region_row].tolist(),
                region["image_shapes"][region_row].tolist(),
            )
            extracted[output_row] = extract_cmr_local_tokens(
                model.micb_dsmil, maps, torch.from_numpy(component.astype("float32"))
            )
        if image_index % 50 == 0 or image_index == len(loader):
            print(f"CMR-V1 {args.role}: {image_index}/{len(loader)}", flush=True)

    if any(value is None for value in extracted):
        raise RuntimeError("V1 token extraction left unmatched rows")
    typed = [value for value in extracted if value is not None]
    token_count = torch.tensor([value.shape[0] for value in typed], dtype=torch.long)
    if args.storage_layout == "packed_list":
        # A single unusually large building must not force every query row to
        # allocate the same padded length.  audit_set_verifier indexes rows one
        # at a time, so a ragged list is lossless and directly consumable.
        tokens = [value.detach().cpu().to(torch.float16).contiguous() for value in typed]
        mask = [torch.ones(value.shape[0], dtype=torch.bool) for value in typed]
    else:
        maximum = max(value.shape[0] for value in typed)
        dimension = typed[0].shape[1]
        tokens = torch.zeros(len(typed), maximum, dimension, dtype=torch.float16)
        mask = torch.zeros(len(typed), maximum, dtype=torch.bool)
        for row, value in enumerate(typed):
            count = value.shape[0]
            tokens[row, :count] = value.to(torch.float16)
            mask[row, :count] = True

    payload = {
        "protocol": "CMR_V1_ISOLATED_counterfactual_local_token_cache_v2",
        "role": args.role,
        "source_region_indices": source_rows.long(),
        "tokens": tokens,
        "token_mask": mask,
        "token_count": token_count,
        "storage_layout": args.storage_layout,
        "labels": p0["labels"][source_rows].long(),
        "events": [p0["events"][row] for row in source_rows.tolist()],
        "image_ids": [p0["image_ids"][row] for row in source_rows.tolist()],
        "p3_checkpoint_sha256": observed_sha,
        "proposal_rule": "frozen_p3_pred_intact_and_any_damaged_in_rawmean_topk",
        "proposal_rule_uses_gt": False,
        "test_accessed": False,
        "frozen_region_cache": {
            "protocol": region["protocol"],
            "path": str(Path(args.region_cache).resolve()),
            "selected_regions": int(source_rows.numel()),
            "skipped_unresolved_source_rows": skipped_source_rows.long(),
            "component_replay_during_token_build": False,
        },
    }
    if args.role == "support":
        payload["memory_local_indices"] = memory_local.long()
        payload["memory_labels"] = candidate["memory_labels"][memory_local].long()
        payload["memory_events"] = [candidate["memory_events"][row] for row in memory_local.tolist()]
    else:
        payload.update({
            "base_predictions": p0["base_predictions"][source_rows].long(),
            "base_probabilities": p0["base_probabilities"][source_rows].float(),
            "candidate_indices": candidate["candidate_indices"][source_rows].long(),
            "candidate_labels": candidate["candidate_labels"][source_rows].long(),
            "candidate_scores": candidate["candidate_scores"][source_rows].float(),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"Saved          : {output}")
    print(f"Regions/tokens : {len(typed)}/{int(token_count.sum())}")
    print(f"Storage layout : {args.storage_layout}")
    print(f"Region skipped : {int(skipped_source_rows.numel())}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
