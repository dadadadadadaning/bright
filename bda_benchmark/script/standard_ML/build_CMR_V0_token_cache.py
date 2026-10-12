# -*- coding: utf-8 -*-
"""Build proposal-only local-token caches for CMR-V0.

Only Train/Validation proposals and their frozen train-memory Damaged supports
are materialised.  This avoids a multi-gigabyte all-region token cache.
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
from model.cmr.pooling import build_cmr_feature_maps, extract_cmr_local_tokens
from script.standard_ML.p3_hard_occurrence_runtime import match_cached_component


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
        raise RuntimeError("V0 requires the frozen CMR-P0 cache")
    if candidate.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("V0 requires the frozen Raw-Mean Top-K candidate cache")
    if bool(p0["metadata"].get("test_accessed", True)) or bool(candidate.get("test_accessed", True)):
        raise RuntimeError("V0 refuses Test-derived inputs")
    if candidate["query_labels"].numel() != p0["labels"].numel():
        raise RuntimeError("P0 and candidate query rows do not align")
    return p0, candidate


def _selected_rows(role: str, p0: dict, candidate: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    proposal = (
        (p0["base_predictions"].long() == 0)
        & (candidate["candidate_labels"][:, 0].long() == 1)
        & ((p0["labels"].long() == 0) | (p0["labels"].long() == 1))
    )
    proposal_rows = torch.nonzero(proposal, as_tuple=False).flatten()
    if role.endswith("query"):
        return proposal_rows, None
    # Materialise every Damaged item in the frozen train-only Memory.  Building
    # support from Train proposals alone would omit Damaged items that appear
    # only in Validation Top-20 lists and make the verifier cache incomplete.
    memory_local = torch.nonzero(
        candidate["memory_labels"].long() == 1, as_tuple=False
    ).flatten()
    source_rows = candidate["memory_indices_in_train"].long()[memory_local]
    return source_rows, memory_local


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("CMR-V0 proposal-only local token cache")
    parser.add_argument("--role", choices=("support", "train_query", "validation_query"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--p0_cache", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    p0, candidate = _load_inputs(args.p0_cache, args.candidate_cache)
    if args.role == "support" and p0["metadata"].get("role") != "train":
        raise RuntimeError("V0 support must be built from Train P0")
    expected_candidate_role = "validation" if args.role == "validation_query" else "train"
    if candidate.get("role") != expected_candidate_role:
        raise RuntimeError(f"V0 role expects {expected_candidate_role} candidates")
    observed_sha = sha256(args.p3_checkpoint)
    if observed_sha != p0["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("V0 P3 checkpoint hash differs from P0")

    source_rows, memory_local = _selected_rows(args.role, p0, candidate)
    if source_rows.numel() == 0:
        raise RuntimeError("V0 selected no regions")
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
        output_loc, _, aux = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        maps = build_cmr_feature_maps(
            model.micb_dsmil, aux["rgb_feature"], aux["sar_feature"], aux["fused_roi"]
        )
        for output_row, source_row in rows_by_image[name]:
            component = match_cached_component(
                building_mask, p0["bboxes"][source_row].tolist(), int(p0["areas"][source_row])
            )
            if component is None:
                raise RuntimeError(
                    f"V0 exact component replay failed for {name} row={source_row}; "
                    "do not substitute an approximate bbox"
                )
            extracted[output_row] = extract_cmr_local_tokens(
                model.micb_dsmil, maps, torch.from_numpy(component.astype("float32"))
            )
        if image_index % 50 == 0 or image_index == len(loader):
            print(f"CMR-V0 {args.role}: {image_index}/{len(loader)}", flush=True)
    if any(value is None for value in extracted):
        raise RuntimeError("V0 token extraction left unmatched rows")
    token_count = torch.tensor([value.shape[0] for value in extracted], dtype=torch.long)
    maximum = max(value.shape[0] for value in extracted)
    dimension = extracted[0].shape[1]
    tokens = torch.zeros(len(extracted), maximum, dimension, dtype=torch.float16)
    mask = torch.zeros(len(extracted), maximum, dtype=torch.bool)
    for row, value in enumerate(extracted):
        count = value.shape[0]
        tokens[row, :count] = value.to(torch.float16)
        mask[row, :count] = True

    payload = {
        "protocol": "CMR_V0_proposal_local_token_cache_v1",
        "role": args.role,
        "source_region_indices": source_rows.long(),
        "tokens": tokens,
        "token_mask": mask,
        "token_count": token_count,
        "labels": p0["labels"][source_rows].long(),
        "events": [p0["events"][row] for row in source_rows.tolist()],
        "image_ids": [p0["image_ids"][row] for row in source_rows.tolist()],
        "p3_checkpoint_sha256": observed_sha,
        "test_accessed": False,
    }
    if args.role == "support":
        payload["memory_local_indices"] = memory_local.long()
    else:
        payload.update({
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
    print(f"Regions/tokens : {len(extracted)}/{int(token_count.sum())}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
