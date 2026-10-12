from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels,
    predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import fast_histogram, file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import FEATURE_NAMES, feature_groups, utility_counts


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build fixed-S2 spatial modification-pattern cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration", "meta_confirm"), required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--state-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    output = Path(args.output)
    state_path = output.with_suffix(output.suffix + ".partial.pt")
    signature = {
        "protocol": PROTOCOL, "split": args.split,
        "list_path_sha256": file_sha256(args.list_path),
        "p3_sha256": file_sha256(args.checkpoint), "minimum_area": args.minimum_area,
        "seed": args.seed,
    }
    state = {
        "processed": 0, "features": {name: [] for name in FEATURE_NAMES},
        "utility": [], "image_index": [], "region_label": [], "delta_hist": [],
        "image_names": [], "image_events": [], "p3_histograms": [], "s2_histograms": [],
        "counts": {"regions_all": 0, "regions_changed": 0, "changed_pixels": 0},
    }
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("SMP cache resume signature mismatch")
        state = saved["state"]
        print(f"Resuming SMP {args.split} after {state['processed']} images", flush=True)
    initial = int(state["processed"])
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    remaining = Subset(dataset, range(initial, stop))
    loader = DataLoader(remaining, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    config = SelectiveContextConfig()
    started = time.perf_counter()
    for offset, (rgb, sar, names) in enumerate(loader):
        image_index = initial + offset
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        loc_logits, damage_logits = p3(rgb, sar)
        probabilities = damage_logits[0].float().softmax(0).cpu()
        p3_prediction = probabilities.argmax(0)
        regions = predicted_building_label_map(loc_logits[0].argmax(0).cpu(), args.minimum_area)
        candidate = fixed_s2_candidate_from_labels(probabilities, regions, config)
        candidate_probabilities = candidate.probabilities.float().cpu()
        candidate_prediction = candidate_probabilities.argmax(0)
        sample_name = str(names[0])
        target = load_semantic_target(args.dataset_path, sample_name).long()
        state["image_names"].append(sample_name)
        state["image_events"].append(sample_name.rsplit("_", 1)[0])
        state["p3_histograms"].append(fast_histogram(p3_prediction, target).long())
        state["s2_histograms"].append(fast_histogram(candidate_prediction, target).long())
        state["counts"]["regions_all"] += regions.count
        for label, record in enumerate(candidate.records, start=1):
            if int(record["changed_pixels"]) <= 0:
                continue
            mask = regions.labels == label
            utility = utility_counts(p3_prediction, candidate_prediction, target, mask)
            if int(utility["changed"]) <= 0:
                continue
            groups = feature_groups(
                building=mask, base_probabilities=probabilities,
                candidate_probabilities=candidate_probabilities,
                base_prediction=p3_prediction, candidate_prediction=candidate_prediction,
                shuffle_seed=args.seed * 1000003 + image_index * 1009 + label,
            )
            for name, values in groups.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"non-finite {name} feature")
                state["features"][name].append(values.half())
            state["utility"].append(utility)
            state["image_index"].append(image_index)
            state["region_label"].append(label)
            masked_target = torch.where(mask, target, torch.full_like(target, 255))
            delta = fast_histogram(candidate_prediction, masked_target) - fast_histogram(p3_prediction, masked_target)
            state["delta_hist"].append(delta.to(torch.int64))
            state["counts"]["regions_changed"] += 1
            state["counts"]["changed_pixels"] += int(utility["changed"])
        state["processed"] = image_index + 1
        if state["processed"] % args.progress_every == 0:
            speed = (state["processed"] - initial) / max(time.perf_counter() - started, 1e-6)
            print(f"SMP cache {args.split}: {state['processed']}/{stop} | {speed:.2f} img/s | buildings={state['counts']['regions_changed']}", flush=True)
        if state["processed"] % args.state_every == 0:
            _atomic_save({"signature": signature, "state": state}, state_path)
    if not state["utility"]:
        raise RuntimeError("no fixed-S2 changed buildings found")
    utility_keys = tuple(state["utility"][0])
    payload = {
        **signature, "configuration": vars(args), "processed_images": state["processed"],
        "p3_provenance": provenance, "s2_configuration": config.__dict__,
        "feature_names": FEATURE_NAMES,
        "features": {name: torch.stack(rows) for name, rows in state["features"].items()},
        "utility": {key: torch.tensor([row[key] for row in state["utility"]]) for key in utility_keys},
        "image_index": torch.tensor(state["image_index"], dtype=torch.int32),
        "region_label": torch.tensor(state["region_label"], dtype=torch.int32),
        "delta_hist": torch.stack(state["delta_hist"]),
        "image_names": state["image_names"], "image_events": state["image_events"],
        "p3_histograms": torch.stack(state["p3_histograms"]),
        "s2_histograms": torch.stack(state["s2_histograms"]), "counts": state["counts"],
        "meta_confirm_accessed": args.split == "meta_confirm", "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    _atomic_save(payload, output)
    if state_path.exists():
        state_path.unlink()
    print(json.dumps({"split": args.split, "processed": state["processed"], **state["counts"]}, indent=2))
    print(f"Saved: {output}")
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

