from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.model.cmr.pooling import build_cmr_feature_maps
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import edge_statistics, pool_feature, region_geometry, semantic_label_and_purity, structural_regions


TENSOR_FIELDS = (
    "image_index", "region_id", "labels", "label_purity", "areas", "geometry",
    "p3_predictions", "p3_probabilities", "rgb_key", "sar_key", "fused_key", "structure_key",
)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def empty_state() -> dict:
    return {key: [] for key in ("image_names", "image_events", "row_events", *TENSOR_FIELDS)}


def pack_state(state: dict) -> dict:
    """Move a bounded image shard from Python lists into compact CPU tensors."""
    payload = {}
    for key, values in state.items():
        if key in ("image_names", "image_events", "row_events"):
            payload[key] = list(values)
        elif key in ("image_index", "region_id", "areas", "p3_predictions"):
            dtype = torch.int32 if key in ("image_index", "region_id", "areas") else torch.long
            payload[key] = torch.tensor(values, dtype=dtype)
        elif key == "labels":
            payload[key] = torch.tensor(values, dtype=torch.long)
        elif key == "label_purity":
            payload[key] = torch.tensor(values, dtype=torch.float32)
        else:
            payload[key] = torch.stack(values).contiguous()
    return payload


def part_paths(parts_dir: Path) -> list[Path]:
    return sorted(parts_dir.glob("part_*.pt"))


def recover_completed(parts_dir: Path, signature: dict) -> int:
    """Return the next image index, accepting only contiguous compatible shards."""
    next_index = 0
    for path in part_paths(parts_dir):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if any(value.get(key) != expected for key, expected in signature.items()):
            raise RuntimeError(f"incompatible existing shard: {path}")
        if int(value["processed_image_start"]) != next_index:
            raise RuntimeError(f"non-contiguous local-cache shards at {path}; use a new CACHE_ROOT")
        next_index = int(value["processed_image_stop"])
    return next_index


def flush_shard(state: dict, signature: dict, parts_dir: Path, start: int, stop: int) -> int:
    if stop <= start:
        return 0
    payload = {**signature, **pack_state(state), "processed_image_start": start, "processed_image_stop": stop}
    destination = parts_dir / f"part_{start:05d}_{stop:05d}.pt"
    atomic_save(payload, destination)
    return len(payload["labels"])


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build RGB-structural local Memory units as resumable shards")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True); parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--segments", type=int, default=192); parser.add_argument("--compactness", type=float, default=8.)
    parser.add_argument("--minimum-area", type=int, default=32); parser.add_argument("--images-per-shard", type=int, default=10)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    if args.images_per_shard < 1:
        raise ValueError("--images-per-shard must be positive")
    output = Path(args.output)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    total = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    signature = {
        "protocol": PROTOCOL, "split": args.split, "p3_sha256": file_sha256(args.checkpoint),
        "list_sha256": file_sha256(args.list_path), "segments": args.segments, "compactness": args.compactness,
        "minimum_area": args.minimum_area, "entry_filter_uses_gt": False,
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    parts_dir = output.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    start_index = recover_completed(parts_dir, signature)
    if start_index > total:
        raise RuntimeError("existing shards exceed requested image count; use a new CACHE_ROOT")
    if start_index == total:
        print(f"LSM cache already complete: {output}", flush=True)
        return 0

    device = torch.device(args.device)
    model, _ = load_frozen_p3(args.checkpoint, device)
    loader = DataLoader(Subset(dataset, range(start_index, total)), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    state = empty_state(); shard_start = start_index; total_regions = 0
    for local_index, (rgb_cpu, sar_cpu, names) in enumerate(loader):
        index = start_index + local_index
        rgb = rgb_cpu.to(device, non_blocking=True); sar = sar_cpu.to(device, non_blocking=True)
        _localization, damage, aux = model(rgb, p3_sar_view(sar), return_aux=True)
        probability = damage[0].float().softmax(0).cpu()
        target = load_semantic_target(args.dataset_path, str(names[0])).long().cpu()
        regions = structural_regions(rgb[0], n_segments=args.segments, compactness=args.compactness).cpu()
        maps = build_cmr_feature_maps(model.micb_dsmil, aux["rgb_feature"], aux["sar_feature"], aux["fused_roi"])
        token_map = maps["token_map"][0]
        name = str(names[0]); event = name.rsplit("_", 1)[0]
        state["image_names"].append(name); state["image_events"].append(event)
        labels_np = np.asarray(regions)
        for value, slices in enumerate(ndimage.find_objects(labels_np), 1):
            if slices is None:
                continue
            local = torch.from_numpy(labels_np[slices] == value)
            mask = torch.zeros_like(regions, dtype=torch.bool); mask[slices] = local
            area = int(mask.sum())
            if area < args.minimum_area:
                continue
            label, label_purity = semantic_label_and_purity(target, mask)
            p3_prob = probability[:, mask].mean(1)
            state["image_index"].append(index); state["region_id"].append(value); state["row_events"].append(event)
            state["labels"].append(label); state["label_purity"].append(label_purity); state["areas"].append(area)
            state["geometry"].append(region_geometry(mask)); state["p3_probabilities"].append(p3_prob)
            state["p3_predictions"].append(int(p3_prob.argmax()))
            state["rgb_key"].append(pool_feature(aux["rgb_feature"][0], mask).half().cpu())
            state["sar_key"].append(pool_feature(aux["sar_feature"][0], mask).half().cpu())
            state["fused_key"].append(pool_feature(token_map, mask).half().cpu())
            state["structure_key"].append(torch.cat((edge_statistics(rgb[0], mask), region_geometry(mask))).half().cpu())
        del _localization, damage, aux, maps, token_map, rgb, sar, probability, target, regions
        if device.type == "cuda":
            torch.cuda.empty_cache()
        processed = index + 1
        if processed % args.images_per_shard == 0 or processed == total:
            written = flush_shard(state, signature, parts_dir, shard_start, processed)
            total_regions += written
            print(f"LSM structural shard {args.split}: {shard_start}:{processed}/{total} | shard_regions={written}", flush=True)
            state = empty_state(); shard_start = processed
        if processed % args.progress_every == 0 or processed == total:
            print(f"LSM structural cache {args.split}: {processed}/{total}", flush=True)

    shards = []
    regions = 0
    for path in part_paths(parts_dir):
        part = torch.load(path, map_location="cpu", weights_only=False)
        shards.append(str(path.resolve())); regions += len(part["labels"])
    manifest = {**signature, "sharded": True, "processed_images": total, "regions": regions, "shards": shards}
    atomic_save(manifest, output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved sharded local cache: {output} | parts={len(shards)} | regions={regions}", flush=True)
    print("Meta-confirm/Val450/test accessed: NEVER / NEVER / NEVER", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
