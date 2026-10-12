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

from bda_benchmark.experiments.spi_complete_building_memory_v1.core import geometry_vector
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import predicted_building_label_map
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import build_independent_card, independent_evidence_maps


FIELDS = ("image_index", "region_label", "areas", "geometry", "labels", "label_purity",
          "class_fractions", "p3_predictions", "p3_probabilities", "rgb_structure",
          "sar_physics", "relative", "quality", "role_tokens", "role_token_mask")


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def empty_state() -> dict:
    return {"image_names": [], "image_events": [], **{name: [] for name in FIELDS}}


def pack(state: dict, signature: dict, start: int, stop: int) -> dict:
    if not state["labels"]:
        raise RuntimeError("independent card shard contains no buildings")
    integer = {"image_index": torch.int32, "region_label": torch.int32,
               "areas": torch.int32, "labels": torch.long, "p3_predictions": torch.long}
    output = {**signature, "processed_image_start": start, "processed_image_stop": stop,
              "image_names": state["image_names"], "image_events": state["image_events"]}
    for name in FIELDS:
        if name in integer:
            output[name] = torch.tensor(state[name], dtype=integer[name])
        else:
            output[name] = torch.stack(state[name])
    return output


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build P3-independent RGB/SAR pseudo-building cards")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True); parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32); parser.add_argument("--evidence-side", type=int, default=256)
    parser.add_argument("--tokens-per-role", type=int, default=3); parser.add_argument("--images-per-shard", type=int, default=25)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    if args.split not in ("meta_train", "meta_calibration"):
        raise RuntimeError("this development experiment may only access meta_train/meta_calibration")
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images}, observed {len(dataset)}")
    stop_at = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    output = Path(args.output); parts = output.with_suffix(".parts"); parts.mkdir(parents=True, exist_ok=True)
    signature = {"protocol": PROTOCOL, "split": args.split, "p3_sha256": file_sha256(args.checkpoint),
                 "list_sha256": file_sha256(args.list_path), "minimum_area": args.minimum_area,
                 "evidence_side": args.evidence_side, "tokens_per_role": args.tokens_per_role,
                 "retrieval_features_use_p3": False, "p3_used_for_mask_only": True,
                 "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    shard_paths = sorted(parts.glob("part_*.pt")); resume = 0; total = 0
    for path in shard_paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("protocol", "split", "p3_sha256", "list_sha256", "minimum_area", "evidence_side", "tokens_per_role"):
            if item.get(key) != signature[key]:
                raise RuntimeError(f"cache signature mismatch at {path}; use a new CACHE_ROOT")
        if int(item["processed_image_start"]) != resume:
            raise RuntimeError("independent card shards are not contiguous")
        resume = int(item["processed_image_stop"]); total += int(item["labels"].numel())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_frozen_p3(args.checkpoint, device)
    loader = DataLoader(Subset(dataset, range(resume, stop_at)), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    state = empty_state(); shard_start = resume; started = time.perf_counter()
    for offset, (rgb_cpu, sar_cpu, names) in enumerate(loader):
        image_id = resume + offset; local_image = len(state["image_names"])
        rgb = rgb_cpu.to(device, non_blocking=True); sar = sar_cpu.to(device, non_blocking=True)
        localization, damage = model(rgb, p3_sar_view(sar))
        probability = damage[0].float().softmax(0).cpu()
        regions = predicted_building_label_map(localization[0].argmax(0).cpu(), args.minimum_area)
        target = load_semantic_target(args.dataset_path, str(names[0])).long().cpu()
        rgb_maps, sar_maps = independent_evidence_maps(rgb, sar, args.evidence_side)
        labels_np = np.asarray(regions.labels.cpu())
        for region_label, slices in enumerate(ndimage.find_objects(labels_np), 1):
            if slices is None: continue
            mask = torch.zeros_like(regions.labels, dtype=torch.bool)
            mask[slices] = torch.from_numpy(labels_np[slices] == region_label)
            area = int(mask.sum())
            if area < args.minimum_area: continue
            geometry = geometry_vector(mask)
            card = build_independent_card(rgb_maps, sar_maps, mask.to(device), geometry.to(device), args.tokens_per_role)
            valid_target = target[mask]; valid_target = valid_target[(valid_target >= 0) & (valid_target <= 3)]
            counts = torch.bincount(valid_target, minlength=4).float() if valid_target.numel() else torch.tensor([1., 0., 0., 0.])
            fractions = counts / counts.sum().clamp_min(1); p3 = probability[:, mask].mean(1)
            state["image_index"].append(local_image); state["region_label"].append(region_label); state["areas"].append(area)
            state["geometry"].append(geometry); state["labels"].append(int(counts.argmax()))
            state["label_purity"].append(torch.tensor(float(fractions.max()))); state["class_fractions"].append(fractions)
            state["p3_predictions"].append(int(p3.argmax())); state["p3_probabilities"].append(p3)
            for key, value in card.items(): state[key].append(value.detach().cpu().half() if value.dtype.is_floating_point else value.cpu())
        name = str(names[0]); state["image_names"].append(name); state["image_events"].append(name.rsplit("_", 1)[0])
        processed = image_id + 1
        if len(state["image_names"]) >= args.images_per_shard or processed == stop_at:
            path = parts / f"part_{len(shard_paths):05d}.pt"; payload = pack(state, signature, shard_start, processed)
            atomic_save(payload, path); shard_paths.append(path); total += int(payload["labels"].numel())
            print(f"Saved independent shard {len(shard_paths)-1}: images={shard_start}:{processed} buildings={len(payload['labels'])}", flush=True)
            state = empty_state(); shard_start = processed; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        if processed % args.progress_every == 0 or processed == stop_at:
            rate = (processed - resume) / max(time.perf_counter() - started, 1e-9)
            print(f"Independent cards {args.split}: {processed}/{stop_at} | {rate:.2f} img/s | buildings={total}", flush=True)
    manifest = {**signature, "sharded": True, "processed_images": stop_at, "buildings": total,
                "shards": [str(path.resolve()) for path in shard_paths]}
    atomic_save(manifest, output)
    output.with_suffix(".json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2)+"\n", encoding="utf-8")
    print(f"Saved: {output}"); print("Meta-confirm/Val450/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
