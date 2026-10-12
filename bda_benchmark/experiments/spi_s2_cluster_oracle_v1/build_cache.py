from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels,
    predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram,
    file_sha256,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import oracle_predictions


METHODS = ("p3", "fixed_s2", "building_oracle", "cluster_oracle", "pixel_oracle")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build S2 modification-cluster oracle cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--state-every", type=int, default=50)
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
        "protocol": PROTOCOL,
        "split": args.split,
        "list_path_sha256": file_sha256(args.list_path),
        "p3_sha256": file_sha256(args.checkpoint),
        "minimum_area": args.minimum_area,
        "cluster_connectivity": 8,
        "cluster_partition": "same_P3_to_S2_transition",
        "oracle_objective": "positive_corrected_minus_harmed_count",
        "seed": args.seed,
    }
    state = {
        "processed": 0,
        "image_names": [],
        "image_events": [],
        "histograms": {name: [] for name in METHODS},
        "counts": Counter(),
        "transition_counts": Counter(),
    }
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("cluster-oracle resume signature mismatch")
        state = saved["state"]
        state["counts"] = Counter(state["counts"])
        state["transition_counts"] = Counter(state["transition_counts"])
        print(f"Resuming Cluster Oracle {args.split} after {state['processed']} images", flush=True)

    initial = int(state["processed"])
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    loader = DataLoader(
        Subset(dataset, range(initial, stop)), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    config = SelectiveContextConfig()
    started = time.perf_counter()
    for offset, (rgb, sar, names) in enumerate(loader):
        image_index = initial + offset
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        localization, damage = p3(rgb, sar)
        probabilities = damage[0].float().softmax(0).cpu()
        base = probabilities.argmax(0)
        regions = predicted_building_label_map(localization[0].argmax(0).cpu(), args.minimum_area)
        candidate = fixed_s2_candidate_from_labels(probabilities, regions, config).probabilities.argmax(0).cpu()
        sample_name = str(names[0])
        target = load_semantic_target(args.dataset_path, sample_name).long()
        building, cluster, pixel, counts = oracle_predictions(base, candidate, target, regions.labels)
        predictions = {
            "p3": base, "fixed_s2": candidate, "building_oracle": building,
            "cluster_oracle": cluster, "pixel_oracle": pixel,
        }
        for name, prediction in predictions.items():
            state["histograms"][name].append(fast_histogram(prediction, target).long())
        state["image_names"].append(sample_name)
        state["image_events"].append(sample_name.rsplit("_", 1)[0])
        for key, value in counts.items():
            if key == "transition_counts":
                state["transition_counts"].update(value)
            else:
                state["counts"][key] += int(value)
        state["counts"]["regions_all"] += int(regions.count)
        state["processed"] = image_index + 1
        if state["processed"] % args.progress_every == 0:
            speed = (state["processed"] - initial) / max(time.perf_counter() - started, 1e-6)
            print(
                f"Cluster Oracle cache {args.split}: {state['processed']}/{stop} | "
                f"{speed:.2f} img/s | clusters={state['counts']['clusters_total']}",
                flush=True,
            )
        if state["processed"] % args.state_every == 0:
            serializable = {**state, "counts": dict(state["counts"]),
                            "transition_counts": dict(state["transition_counts"])}
            _atomic_save({"signature": signature, "state": serializable}, state_path)

    payload = {
        **signature,
        "configuration": vars(args),
        "processed_images": state["processed"],
        "p3_provenance": provenance,
        "s2_configuration": config.__dict__,
        "image_names": state["image_names"],
        "image_events": state["image_events"],
        "histograms": {name: torch.stack(rows) for name, rows in state["histograms"].items()},
        "counts": dict(state["counts"]),
        "transition_counts": dict(state["transition_counts"]),
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    _atomic_save(payload, output)
    if state_path.exists():
        state_path.unlink()
    print(json.dumps({"split": args.split, "processed": state["processed"],
                      "counts": payload["counts"]}, indent=2))
    print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

