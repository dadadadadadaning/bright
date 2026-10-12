# -*- coding: utf-8 -*-
"""CMR-H1S0: frozen-P3 cache-to-online feature replay guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.make_data_loader_p4_native_sar import P4NativeSARDataset
from model.DamageFormer_S8_P4_NativeSAR_MICB import (
    DamageFormer_S8_P4_NativeSAR_MICB,
    load_p3_checkpoint,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache
from script.standard_ML.p3_hard_occurrence_runtime import (
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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("CMR-H1S0 frozen replay guard")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--h0_pairs", required=True)
    parser.add_argument("--h0b_manifest", required=True)
    parser.add_argument("--p3_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=640)
    args = parser.parse_args()

    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    pairs = torch.load(args.h0_pairs, map_location="cpu", weights_only=False)
    manifest = torch.load(args.h0b_manifest, map_location="cpu", weights_only=False)
    if pairs.get("protocol") != "CMR_H0_train_only_occurrence_hard_pairs_v1":
        raise RuntimeError("H1S0 refuses an unknown H0 pair protocol")
    if manifest.get("protocol") != "CMR_H0B_event_capped_rotating_train_manifest_v1":
        raise RuntimeError("H1S0 refuses an unknown H0B manifest protocol")
    if not bool(manifest.get("passed", False)):
        raise RuntimeError("H1S0 requires a passed H0B manifest")
    if any(bool(source.get("validation_accessed", True)) or bool(source.get("test_accessed", True)) for source in (pairs, manifest)):
        raise RuntimeError("H1S0 refuses Validation/Test-derived inputs")
    checkpoint_sha = sha256(args.p3_checkpoint)
    expected_sha = cache["metadata"].get("p3_checkpoint_sha256")
    if checkpoint_sha != expected_sha or manifest.get("p3_checkpoint_sha256") != expected_sha:
        raise RuntimeError("P3 checkpoint/cache/manifest SHA256 mismatch")
    if manifest.get("h0_pairs_sha256") != sha256(args.h0_pairs):
        raise RuntimeError("H0 pair file changed after H0B admission")

    anchor_indices = pairs["anchor_indices"].long().flatten()
    positive_indices = pairs["positive_indices"].long()
    negative_indices = pairs["negative_indices"].long()
    pair_contract = (
        bool((labels[anchor_indices] == 1).all())
        and bool((labels[positive_indices] == 1).all())
        and bool((labels[negative_indices] == 0).all())
        and not bool(
            (events[positive_indices] == events[anchor_indices, None]).any()
        )
        and not bool(
            (events[negative_indices] == events[anchor_indices, None]).any()
        )
    )
    preview_rows = manifest["preview_epoch_rows"].long().flatten()
    take = min(int(args.sample_count), int(preview_rows.numel()))
    if take < 256:
        raise RuntimeError(f"H1S0 requires at least 256 replay anchors, found {take}")
    selected_rows = preview_rows[:take]
    selected_train_indices = anchor_indices[selected_rows]
    rows_by_image = defaultdict(list)
    for row, train_index in zip(selected_rows.tolist(), selected_train_indices.tolist()):
        image_id = str(cache["image_ids"][train_index])
        rows_by_image[image_id].append((int(row), int(train_index)))

    dataset = P4NativeSARDataset(
        dataset_path=args.dataset_path,
        data_list=sorted(rows_by_image),
        crop_size=args.crop_size,
        split="cache",
        variant="a0",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    device = torch.device(args.device)
    model = DamageFormer_S8_P4_NativeSAR_MICB(variant="a0").to(device)
    load_p3_checkpoint(model, args.p3_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    requested = int(selected_rows.numel())
    found_image_rows = 0
    component_matches = 0
    key_cosines = []
    probability_deltas = []
    replay_rows = []
    for batch in loader:
        name = str(batch["name"][0])
        pre = batch["pre_img"].float().to(device)
        post = batch["teacher_post_img"].float().to(device)
        output_loc, output_damage, aux = model(pre, post, return_aux=True)
        building_mask = output_loc.argmax(dim=1)[0].cpu().numpy() == 1
        for pair_row, train_index in rows_by_image.get(name, []):
            found_image_rows += 1
            bbox = cache["bboxes"][train_index].tolist()
            area = int(cache["areas"][train_index])
            component = match_cached_component(building_mask, bbox, area)
            if component is None:
                replay_rows.append({
                    "pair_row": pair_row,
                    "train_index": train_index,
                    "image_id": name,
                    "component_matched": False,
                })
                continue
            component_matches += 1
            online_key = differentiable_region_mean_key(
                aux["fused_roi"], component, batch_index=0
            ).cpu()
            cached_key = feature[train_index]
            cosine = float((online_key * cached_key).sum())
            online_probability = region_conditional_probability(
                output_damage, component, batch_index=0
            ).cpu()
            probability_delta = float(
                (online_probability - cache["base_probabilities"][train_index]).abs().max()
            )
            key_cosines.append(cosine)
            probability_deltas.append(probability_delta)
            replay_rows.append({
                "pair_row": pair_row,
                "train_index": train_index,
                "image_id": name,
                "component_matched": True,
                "key_cosine": cosine,
                "conditional_probability_max_abs_delta": probability_delta,
            })

    image_match_rate = found_image_rows / max(requested, 1)
    component_match_rate = component_matches / max(requested, 1)
    key_min = min(key_cosines) if key_cosines else 0.0
    key_mean = sum(key_cosines) / max(len(key_cosines), 1)
    probability_max = max(probability_deltas) if probability_deltas else float("inf")
    checks = {
        "at_least_256_anchors_replayed": requested >= 256,
        "image_id_match_rate_is_one": image_match_rate >= 1.0,
        "component_match_rate_at_least_0_99": component_match_rate >= 0.99,
        "minimum_key_cosine_at_least_0_999": key_min >= 0.999,
        "conditional_probability_delta_at_most_1e_5": probability_max <= 1e-5,
        "positive_negative_contract_exact": pair_contract,
        "p3_checkpoint_sha256_exact": checkpoint_sha == expected_sha,
    }
    passed = all(checks.values())
    decision = (
        "ADMIT_TO_CMR_H1A_500_STEP_SCREEN"
        if passed
        else "STOP_BEFORE_H1_TRAINING_FIX_REPLAY_CONTRACT"
    )
    result = {
        "protocol": "CMR_H1S0_frozen_p3_feature_replay_guard_v1",
        "requested_anchors": requested,
        "found_image_rows": found_image_rows,
        "component_matches": component_matches,
        "image_id_match_rate": image_match_rate,
        "component_match_rate": component_match_rate,
        "key_cosine_mean": key_mean,
        "key_cosine_min": key_min,
        "conditional_probability_max_abs_delta": probability_max,
        "pair_contract_exact": pair_contract,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "rows": replay_rows,
        "validation_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print("CMR-H1S0 frozen P3 feature replay guard")
    print(f"Requested/found       : {requested}/{found_image_rows}")
    print(f"Component matches     : {component_matches} ({100*component_match_rate:.2f}%)")
    print(f"Key cosine mean/min   : {key_mean:.8f}/{key_min:.8f}")
    print(f"Probability max delta : {probability_max:.8g}")
    print(f"Checks                : {checks}")
    print(f"Decision              : {decision}")
    print(f"Saved                 : {output}")
    print("Validation accessed   : False")
    print("Test accessed         : False")


if __name__ == "__main__":
    main()
