from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import predicted_building_label_map
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.model.cmr.pooling import build_cmr_feature_maps, extract_cmr_pooling_keys
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import building_role_tokens, geometry_vector, masked_mean_std


TENSOR_FIELDS = (
    "image_index", "region_label", "areas", "bboxes", "geometry", "labels",
    "label_purity", "class_fractions", "p3_predictions", "p3_probabilities",
    "role_tokens", "role_token_mask", "role_fractions",
)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def empty_state() -> dict:
    return {
        "image_names": [], "image_events": [], "key_variants": {},
        "modality_keys": {"rgb": [], "sar": []},
        "image_index": [], "region_label": [], "areas": [], "bboxes": [],
        "geometry": [], "labels": [], "label_purity": [], "class_fractions": [],
        "p3_predictions": [], "p3_probabilities": [], "role_tokens": [],
        "role_token_mask": [], "role_fractions": [],
    }


def pack_state(state: dict, common: dict, start: int, stop: int) -> dict:
    if not state["labels"]:
        raise RuntimeError(f"card shard {start}:{stop} contains no buildings")
    return {
        **common, "processed_image_start": start, "processed_image_stop": stop,
        "image_names": state["image_names"], "image_events": state["image_events"],
        "key_variants": {k: torch.stack(v) for k, v in state["key_variants"].items()},
        "modality_keys": {k: torch.stack(v) for k, v in state["modality_keys"].items()},
        "image_index": torch.tensor(state["image_index"], dtype=torch.int32),
        "region_label": torch.tensor(state["region_label"], dtype=torch.int32),
        "areas": torch.tensor(state["areas"], dtype=torch.int32),
        "bboxes": torch.tensor(state["bboxes"], dtype=torch.int32),
        "geometry": torch.stack(state["geometry"]),
        "labels": torch.tensor(state["labels"], dtype=torch.long),
        "label_purity": torch.tensor(state["label_purity"]),
        "class_fractions": torch.stack(state["class_fractions"]),
        "p3_predictions": torch.tensor(state["p3_predictions"], dtype=torch.long),
        "p3_probabilities": torch.stack(state["p3_probabilities"]),
        "role_tokens": torch.stack(state["role_tokens"]),
        "role_token_mask": torch.stack(state["role_token_mask"]),
        "role_fractions": torch.stack(state["role_fractions"]),
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build sharded complete pseudo-building cards")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True); parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32); parser.add_argument("--tokens-per-role", type=int, default=2)
    parser.add_argument("--images-per-shard", type=int, default=25)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    output = Path(args.output); parts_dir = output.with_suffix(".parts"); parts_dir.mkdir(parents=True, exist_ok=True)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images}, observed {len(dataset)}")
    final_stop = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    signature = {
        "protocol": PROTOCOL, "split": args.split, "p3_sha256": file_sha256(args.checkpoint),
        "list_sha256": file_sha256(args.list_path), "minimum_area": args.minimum_area,
        "tokens_per_role": args.tokens_per_role, "entry_filter_uses_gt": False,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    existing = sorted(parts_dir.glob("part_*.pt")); resume = 0
    for part in existing:
        header = torch.load(part, map_location="cpu", weights_only=False)
        for key in ("protocol", "split", "p3_sha256", "list_sha256", "minimum_area", "tokens_per_role"):
            if header.get(key) != signature[key]:
                raise RuntimeError(f"shard signature mismatch at {part}; use a new CACHE_ROOT")
        if int(header["processed_image_start"]) != resume:
            raise RuntimeError("card shards are not contiguous")
        resume = int(header["processed_image_stop"])
    if resume > final_stop:
        raise RuntimeError("existing shards exceed requested image count")
    if resume:
        print(f"Resuming complete cards {args.split} at image {resume}/{final_stop}", flush=True)

    device = torch.device(args.device); model, _ = load_frozen_p3(args.checkpoint, device)
    loader = DataLoader(Subset(dataset, range(resume, final_stop)), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    state = empty_state(); shard_start = resume; part_index = len(existing); total_buildings = 0
    for part in existing:
        total_buildings += int(torch.load(part, map_location="cpu", weights_only=False)["labels"].numel())
    started = time.perf_counter()
    for offset, (rgb_cpu, sar_cpu, names) in enumerate(loader):
        global_image = resume + offset; local_image = len(state["image_names"])
        rgb = rgb_cpu.to(device, non_blocking=True); sar = sar_cpu.to(device, non_blocking=True)
        localization, damage, aux = model(rgb, p3_sar_view(sar), return_aux=True)
        probability = damage[0].float().softmax(0).cpu()
        regions = predicted_building_label_map(localization[0].argmax(0).cpu(), args.minimum_area)
        target = load_semantic_target(args.dataset_path, str(names[0])).long().cpu()
        maps = build_cmr_feature_maps(model.micb_dsmil, aux["rgb_feature"], aux["sar_feature"], aux["fused_roi"])
        label_np = np.asarray(regions.labels.cpu())
        for region_label, slices in enumerate(ndimage.find_objects(label_np), 1):
            if slices is None: continue
            local = torch.from_numpy(label_np[slices] == region_label)
            mask = torch.zeros_like(regions.labels, dtype=torch.bool); mask[slices] = local
            area = int(mask.sum())
            if area < args.minimum_area: continue
            keys = extract_cmr_pooling_keys(model.micb_dsmil, maps, mask.float())
            for key, value in keys.items(): state["key_variants"].setdefault(key, []).append(value.half())
            state["modality_keys"]["rgb"].append(masked_mean_std(aux["rgb_feature"][0], mask).half())
            state["modality_keys"]["sar"].append(masked_mean_std(aux["sar_feature"][0], mask).half())
            tokens, token_mask, fractions = building_role_tokens(
                maps["token_map"][0], maps["change_energy"][0], mask,
                tokens_per_role=args.tokens_per_role,
            )
            region_target = target[mask]; region_target = region_target[(region_target >= 0) & (region_target <= 3)]
            counts = torch.bincount(region_target, minlength=4).float() if region_target.numel() else torch.tensor([1., 0., 0., 0.])
            class_fraction = counts / counts.sum().clamp_min(1); region_probability = probability[:, mask].mean(1)
            y, x = torch.nonzero(mask, as_tuple=True)
            state["image_index"].append(local_image); state["region_label"].append(region_label); state["areas"].append(area)
            state["bboxes"].append([int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)])
            state["geometry"].append(geometry_vector(mask)); state["labels"].append(int(counts.argmax()))
            state["label_purity"].append(float(class_fraction.max())); state["class_fractions"].append(class_fraction)
            state["p3_predictions"].append(int(region_probability.argmax())); state["p3_probabilities"].append(region_probability)
            state["role_tokens"].append(tokens); state["role_token_mask"].append(token_mask); state["role_fractions"].append(fractions)
        name = str(names[0]); state["image_names"].append(name); state["image_events"].append(name.rsplit("_", 1)[0])
        processed = global_image + 1
        should_flush = len(state["image_names"]) >= args.images_per_shard or processed == final_stop
        if should_flush:
            part_path = parts_dir / f"part_{part_index:05d}.pt"
            payload = pack_state(state, signature, shard_start, processed)
            count = int(payload["labels"].numel()); atomic_save(payload, part_path)
            existing.append(part_path); total_buildings += count
            print(f"Saved card shard {part_index}: images={shard_start}:{processed} buildings={count}", flush=True)
            part_index += 1; shard_start = processed; state = empty_state(); gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        if processed % args.progress_every == 0 or processed == final_stop:
            rate = (processed - resume) / max(time.perf_counter() - started, 1e-9)
            print(f"Complete cards {args.split}: {processed}/{final_stop} | {rate:.2f} img/s | buildings={total_buildings + len(state['labels'])}", flush=True)
    manifest = {
        **signature, "sharded": True, "processed_images": final_stop,
        "buildings": total_buildings, "shards": [str(path.resolve()) for path in existing],
    }
    atomic_save(manifest, output)
    output.with_suffix(".json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved sharded manifest: {output} | shards={len(existing)} buildings={total_buildings}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
