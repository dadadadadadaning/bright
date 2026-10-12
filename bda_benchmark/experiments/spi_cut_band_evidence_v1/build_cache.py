from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_evidence_v2.core import matching_evidence
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import predicted_building_label_map
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_rgid_e0_v1.core import s2_prediction
from bda_benchmark.experiments.spi_selective_rgid_e0_v1.runtime import load_rgid
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import fast_histogram, file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import (
    candidate_cut_band, scharr_magnitude, build_feature_groups, split_utility,
    feature_names, compressed_feature_channels, fast_geometry_features,
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build candidate-cut-band evidence cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgid-checkpoint", required=True)
    parser.add_argument("--rgid-provenance", required=True)
    parser.add_argument("--rgid-train-list", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration", "meta_confirm"), required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--band-radius", type=int, default=2)
    parser.add_argument("--minimum-band-pixels", type=int, default=8)
    parser.add_argument("--matching-max-side", type=int, default=64)
    parser.add_argument("--matching-window", type=int, default=5)
    parser.add_argument("--matching-channels", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--state-every", type=int, default=25)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    p3, p3_provenance = load_frozen_p3(args.checkpoint, device)
    head, decoder, boundary_threshold, rgid_provenance = load_rgid(
        args.rgid_checkpoint, args.rgid_provenance, args.checkpoint,
        args.rgid_train_list, device,
    )
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    output = Path(args.output)
    state_path = output.with_suffix(output.suffix + ".partial.pt")
    signature = {
        "protocol": PROTOCOL, "split": args.split,
        "list_sha256": file_sha256(args.list_path), "p3_sha256": file_sha256(args.checkpoint),
        "rgid_sha256": file_sha256(args.rgid_checkpoint), "band_radius": args.band_radius,
        "matching_max_side": args.matching_max_side, "matching_window": args.matching_window,
        "matching_channels": args.matching_channels,
    }
    state = {
        "processed": 0, "features": {}, "utility": [], "image_index": [],
        "parent_id": [], "delta_hist": [], "image_names": [], "image_events": [],
        "r0_histograms": [], "r1_histograms": [], "counts": {"splittable": 0, "stored": 0, "empty_band": 0},
        "timing_seconds": {"p3": 0.0, "decode": 0.0, "matching": 0.0, "other": 0.0},
    }
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("cut-band resume signature mismatch")
        state = saved["state"]
        print(f"Resuming {args.split} after {state['processed']} images", flush=True)
    initial_processed = int(state["processed"])
    remaining = Subset(dataset, range(initial_processed, len(dataset)))
    loader = DataLoader(remaining, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    started = time.perf_counter()
    matching_names = None
    def clock() -> float:
        if args.profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()
    for offset, (rgb, sar, names) in enumerate(loader):
        image_started = clock()
        image_index = initial_processed + offset
        if args.max_images and image_index >= args.max_images:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits, auxiliary = p3(rgb, sar, return_aux=True)
        p3_finished = clock()
        probabilities = damage_logits[0].float().softmax(0).cpu()
        parents = predicted_building_label_map(loc_logits[0].argmax(0).cpu(), args.minimum_area)
        parent_np = parents.labels.numpy()
        support = parents.labels > 0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            geometry_prediction = head(auxiliary["fused_roi"].float())
        center_logits = geometry_prediction["center_logits"][0, 0]
        boundary_logits = geometry_prediction["boundary_logits"][0, 0]
        decoded = decoder.decode(
            center_logits, boundary_logits, support, mode="g3", inputs_are_logits=True,
            boundary_gate_threshold=boundary_threshold,
        )
        children = decoded.instance_labels.astype("int32", copy=False)
        if not torch.equal(torch.from_numpy(children > 0), support.cpu()):
            raise RuntimeError("RGID changed P3/LocCC building support")
        boundary_full = torch.nn.functional.interpolate(
            boundary_logits.detach().float().sigmoid()[None, None], size=parents.labels.shape,
            mode="bilinear", align_corners=False,
        )[0, 0].cpu()
        decode_finished = clock()
        r0, r1 = s2_prediction(probabilities, parent_np), s2_prediction(probabilities, children)
        sample_name = str(names[0])
        target = load_semantic_target(args.dataset_path, sample_name).long()
        state["r0_histograms"].append(fast_histogram(r0, target).long())
        state["r1_histograms"].append(fast_histogram(r1, target).long())
        state["image_names"].append(sample_name)
        state["image_events"].append(sample_name.rsplit("_", 1)[0])
        support_np = parent_np > 0
        parent_child_pairs = np.unique(
            np.stack((parent_np[support_np], children[support_np]), axis=1), axis=0
        )
        parent_ids = [int(value) for value in np.unique(parent_np[support_np])]
        child_count_array = np.bincount(parent_child_pairs[:, 0], minlength=max(parent_ids) + 1)
        child_counts = {pid: int(child_count_array[pid]) for pid in parent_ids}
        splittable = [pid for pid in parent_ids if child_counts[pid] >= 2]
        state["counts"]["splittable"] += len(splittable)
        matching_elapsed = 0.0
        if splittable:
            rgb_edge = scharr_magnitude(rgb)
            sar_edge = scharr_magnitude(sar)
            rgb_matching = compressed_feature_channels(auxiliary["rgb_feature"], args.matching_channels)
            sar_matching = compressed_feature_channels(auxiliary["sar_feature"], args.matching_channels)
            matching = matching_evidence(
                rgb_matching, sar_matching, tuple(r0.shape),
                max_side=args.matching_max_side, window_size=args.matching_window,
            )
            matching_names = matching.channel_names
            matching_cpu = matching.values[0].float().cpu()
            matching_finished = clock()
            matching_elapsed = matching_finished - decode_finished
            generator = torch.Generator().manual_seed(args.seed + image_index)
            shift_y = int(torch.randint(max(1, r0.shape[0] // 4), max(2, 3 * r0.shape[0] // 4), (1,), generator=generator))
            shift_x = int(torch.randint(max(1, r0.shape[1] // 4), max(2, 3 * r0.shape[1] // 4), (1,), generator=generator))
            shuffled = torch.roll(matching_cpu, shifts=(shift_y, shift_x), dims=(-2, -1))
            for parent_id in splittable:
                band = candidate_cut_band(parent_np, children, parent_id, args.band_radius)
                if int(band.sum()) < args.minimum_band_pixels:
                    state["counts"]["empty_band"] += 1
                    continue
                parent_mask = parents.labels == parent_id
                geometry = fast_geometry_features(
                    parent_mask=parent_mask, probabilities=probabilities,
                    boundary_probability=boundary_full, child_count=child_counts[parent_id],
                )
                groups = build_feature_groups(
                    geometry=geometry, parent_mask=parent_mask, band=band,
                    rgb_edge=rgb_edge[0], sar_edge=sar_edge[0], matching_maps=matching_cpu,
                    shuffled_matching_maps=shuffled,
                )
                for group, values in groups.items():
                    state["features"].setdefault(group, []).append(values.half())
                state["utility"].append(split_utility(r0, r1, target, parent_mask))
                state["image_index"].append(image_index)
                state["parent_id"].append(parent_id)
                masked_target = torch.where(parent_mask, target, torch.full_like(target, 255))
                state["delta_hist"].append((fast_histogram(r1, masked_target) - fast_histogram(r0, masked_target)).to(torch.int64))
                state["counts"]["stored"] += 1
        image_finished = clock()
        state["timing_seconds"]["p3"] += p3_finished - image_started
        state["timing_seconds"]["decode"] += decode_finished - p3_finished
        state["timing_seconds"]["matching"] += matching_elapsed
        state["timing_seconds"]["other"] += max(
            0.0, image_finished - image_started
            - (p3_finished - image_started) - (decode_finished - p3_finished) - matching_elapsed
        )
        state["processed"] = image_index + 1
        if state["processed"] % args.progress_every == 0:
            speed = (state["processed"] - initial_processed) / max(time.perf_counter() - started, 1e-6)
            print(f"Cut-band cache {args.split}: {state['processed']}/{len(dataset)} | {speed:.2f} img/s | candidates={state['counts']['stored']}", flush=True)
            if args.profile_timing:
                count = max(1, state["processed"])
                row = {key: round(1000.0 * value / count, 1) for key, value in state["timing_seconds"].items()}
                print(f"Cut-band timing mean ms/image: {row}", flush=True)
        if state["processed"] % args.state_every == 0:
            _atomic_save({"signature": signature, "state": state}, state_path)
    if not state["utility"] or matching_names is None:
        raise RuntimeError("no valid candidate cut bands were found")
    utility_keys = tuple(state["utility"][0])
    payload = {
        **signature, "configuration": vars(args), "processed_images": state["processed"],
        "p3_provenance": p3_provenance, "rgid_provenance": rgid_provenance,
        "feature_names": feature_names(tuple(matching_names)),
        "features": {name: torch.stack(rows) for name, rows in state["features"].items()},
        "utility": {key: torch.tensor([row[key] for row in state["utility"]]) for key in utility_keys},
        "image_index": torch.tensor(state["image_index"], dtype=torch.int32),
        "parent_id": torch.tensor(state["parent_id"], dtype=torch.int32),
        "delta_hist": torch.stack(state["delta_hist"]),
        "image_names": state["image_names"], "image_events": state["image_events"],
        "r0_histograms": torch.stack(state["r0_histograms"]),
        "r1_histograms": torch.stack(state["r1_histograms"]), "counts": state["counts"],
        "timing_seconds": state["timing_seconds"],
        "meta_confirm_accessed": args.split == "meta_confirm",
        "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    _atomic_save(payload, output)
    if state_path.exists():
        state_path.unlink()
    print(f"Saved: {output}", flush=True)
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
