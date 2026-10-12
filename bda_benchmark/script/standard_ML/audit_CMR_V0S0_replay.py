# -*- coding: utf-8 -*-
"""CMR-V0S0: forensic audit of old P0 region identity under current runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import P4NativeSARDataset
from model.DamageFormer_S7_SIRM_M2B import connected_components_4
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from script.standard_ML.cmr_v0_replay_runtime import bbox_iou, guarded_identity, replay_decision
from script.standard_ML.p3_hard_occurrence_runtime import (
    component_bbox,
    differentiable_region_mean_key,
    match_cached_component,
    region_conditional_probability,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_source_rows(role: str, p0: dict, candidate: dict) -> torch.Tensor:
    if role == "support":
        local = torch.nonzero(candidate["memory_labels"].long() == 1, as_tuple=False).flatten()
        return candidate["memory_indices_in_train"].long()[local]
    proposal = (
        (p0["base_predictions"].long() == 0)
        & (candidate["candidate_labels"][:, 0].long() == 1)
        & ((p0["labels"].long() == 0) | (p0["labels"].long() == 1))
    )
    return torch.nonzero(proposal, as_tuple=False).flatten()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("CMR-V0S0 exact component replay forensic audit")
    parser.add_argument("--role", choices=("support", "train_query", "validation_query"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--p0_cache", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_count", type=int, default=0, help="0 audits all selected rows")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    p0 = torch.load(args.p0_cache, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate_cache, map_location="cpu", weights_only=False)
    if p0["metadata"].get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("V0S0 requires a frozen P0 cache")
    if candidate.get("protocol") != "CMR_M3_frozen_mean_topk_candidates_v1":
        raise RuntimeError("V0S0 requires frozen Raw-Mean candidates")
    if bool(p0["metadata"].get("test_accessed", True)) or bool(candidate.get("test_accessed", True)):
        raise RuntimeError("V0S0 refuses Test-derived inputs")
    checkpoint_sha = sha256(args.p3_checkpoint)
    if checkpoint_sha != p0["metadata"].get("p3_checkpoint_sha256"):
        raise RuntimeError("V0S0 P3 checkpoint SHA differs from P0")
    source_rows = selected_source_rows(args.role, p0, candidate)
    if args.sample_count > 0 and source_rows.numel() > args.sample_count:
        generator = torch.Generator().manual_seed(args.seed)
        source_rows = source_rows[
            torch.randperm(source_rows.numel(), generator=generator)[: args.sample_count]
        ]
    rows_by_image = defaultdict(list)
    for source_row in source_rows.tolist():
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

    rows = []
    for image_index, batch in enumerate(loader, 1):
        name = str(batch["name"][0])
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        output_loc, output_damage, aux = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        current_components = connected_components_4(building_mask)
        for source_row in rows_by_image[name]:
            expected_bbox = p0["bboxes"][source_row].tolist()
            expected_area = int(p0["areas"][source_row])
            cached_key = p0["key_variants"]["mean"][source_row].float()
            cached_probability = p0["base_probabilities"][source_row].float()
            exact = match_cached_component(building_mask, expected_bbox, expected_area)
            candidates = [exact] if exact is not None else [
                component for component in current_components
                if bbox_iou(expected_bbox, component_bbox(component)) > 0
            ]
            candidate_rows = []
            for component in candidates:
                if component is None:
                    continue
                online_key = differentiable_region_mean_key(
                    aux["fused_roi"], component, batch_index=0
                ).cpu()
                online_probability = region_conditional_probability(
                    output_damage, component, batch_index=0
                ).cpu()
                area = int(component.sum())
                candidate_rows.append({
                    "bbox": list(component_bbox(component)),
                    "area": area,
                    "bbox_iou": bbox_iou(expected_bbox, component_bbox(component)),
                    "relative_area_error": abs(area - expected_area) / max(expected_area, 1),
                    "mean_key_cosine": float((online_key * cached_key).sum()),
                    "probability_max_abs_delta": float((online_probability - cached_probability).abs().max()),
                })
            best = max(
                candidate_rows,
                key=lambda item: (item["bbox_iou"], item["mean_key_cosine"], -item["relative_area_error"]),
                default={
                    "bbox": None, "area": None, "bbox_iou": 0.0,
                    "relative_area_error": 1.0, "mean_key_cosine": -1.0,
                    "probability_max_abs_delta": 1.0,
                },
            )
            row = {
                "source_row": source_row,
                "image_id": name,
                "event": str(p0["events"][source_row]),
                "expected_bbox": expected_bbox,
                "expected_area": expected_area,
                "exact_match": exact is not None,
                **best,
            }
            row["guarded_identity"] = guarded_identity(row)
            rows.append(row)
        if image_index % 50 == 0 or image_index == len(loader):
            print(f"V0-S0 {args.role}: {image_index}/{len(loader)} | rows={len(rows)}", flush=True)

    exact_rate = sum(bool(row["exact_match"]) for row in rows) / max(len(rows), 1)
    guarded_rate = sum(bool(row["guarded_identity"]) for row in rows) / max(len(rows), 1)
    failed_events = Counter(row["event"] for row in rows if not row["exact_match"])
    checks, decision = replay_decision(exact_rate=exact_rate, guarded_identity_rate=guarded_rate)
    payload = {
        "protocol": "CMR_V0S0_old_p0_current_runtime_region_replay_audit_v1",
        "role": args.role,
        "audited_rows": len(rows),
        "exact_match_rate": exact_rate,
        "guarded_identity_rate": guarded_rate,
        "failed_event_counts": dict(failed_events),
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "p3_checkpoint_sha256": checkpoint_sha,
            "model_source_sha256": sha256(str(Path(__file__).parents[2] / "model" / "DamageFormer_S8_P4_NativeSAR_MICB.py")),
            "dataset_source_sha256": sha256(str(Path(__file__).parents[2] / "dataset" / "make_data_loader_p4_native_sar.py")),
        },
        "checks": checks,
        "passed": decision == "ADMIT_TO_V0_TOKEN_CACHE_EXACT_REPLAY",
        "decision": decision,
        "rows": rows,
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print("CMR-V0S0 old-P0/current-runtime component replay audit")
    print(f"Role/rows       : {args.role}/{len(rows)}")
    print(f"Exact rate      : {100*exact_rate:.3f}%")
    print(f"Guarded identity: {100*guarded_rate:.3f}%")
    print(f"Failed events   : {dict(failed_events)}")
    print(f"Decision        : {decision}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()

