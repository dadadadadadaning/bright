from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.core import (
    iter_transition_clusters, to_nchw_stage,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels, predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset

from . import PROTOCOL
from .core import hard_token_coverage


class ShapeCapture:
    def __init__(self, model) -> None:
        self.shapes = {}
        self.handles = [
            model.encoder_1.register_forward_hook(self._hook("rgb")),
            model.encoder_2.register_forward_hook(self._hook("sar")),
        ]

    def _hook(self, name):
        def hook(_module, _inputs, output):
            stages = (
                to_nchw_stage(output[2], 192), to_nchw_stage(output[3], 384),
                to_nchw_stage(output[4], 768),
            )
            self.shapes[name] = tuple(tuple(row.shape[-2:]) for row in stages)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Exact Stage2/3/4 token coverage for S2 clusters")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    if reference.get("split") != "meta_calibration" or reference.get("meta_confirm_accessed", True):
        raise RuntimeError("D0-B requires the untouched meta_calibration cluster cache")
    device = torch.device(args.device)
    model, provenance = load_frozen_p3(args.checkpoint, device)
    if provenance["p3_sha256"] != reference["p3_sha256"]:
        raise RuntimeError("P3 fingerprint differs from Cluster-E0 cache")
    capture = ShapeCapture(model)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images}, observed {len(dataset)}")
    stop = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    loader = DataLoader(
        Subset(dataset, range(stop)), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    rows = {name: [] for name in (
        "image_index", "region_label", "cluster_index", "direction_code", "area",
        "bbox_height", "bbox_width", "stage2_tokens", "stage3_tokens", "stage4_tokens",
    )}
    config = SelectiveContextConfig()
    try:
        for image_index, (rgb, sar, _names) in enumerate(loader):
            rgb, sar = rgb.to(device), p3_sar_view(sar.to(device))
            localization, damage = model(rgb, sar)
            probabilities = damage[0].softmax(0).cpu()
            base = probabilities.argmax(0)
            regions = predicted_building_label_map(
                localization[0].argmax(0).cpu(), args.minimum_area
            )
            candidate = fixed_s2_candidate_from_labels(
                probabilities, regions, config
            ).probabilities.argmax(0)
            if capture.shapes["rgb"] != capture.shapes["sar"]:
                raise RuntimeError("RGB/SAR stage shapes differ")
            image_shape = tuple(base.shape)
            for spec in iter_transition_clusters(base, candidate, regions.labels):
                yy, xx = torch.nonzero(spec.cluster, as_tuple=True)
                rows["image_index"].append(image_index)
                rows["region_label"].append(spec.region_label)
                rows["cluster_index"].append(spec.cluster_index)
                rows["direction_code"].append(spec.transition)
                rows["area"].append(int(spec.cluster.sum()))
                rows["bbox_height"].append(int(yy.max() - yy.min() + 1))
                rows["bbox_width"].append(int(xx.max() - xx.min() + 1))
                for stage_index, stage in enumerate(("stage2", "stage3", "stage4")):
                    rows[f"{stage}_tokens"].append(hard_token_coverage(
                        spec.cluster, spec.slices, image_shape,
                        capture.shapes["rgb"][stage_index],
                    ))
            if (image_index + 1) % args.progress_every == 0:
                print(
                    f"Cluster Detail geometry: {image_index + 1}/{stop} | "
                    f"clusters={len(rows['area'])}", flush=True,
                )
    finally:
        capture.close()

    payload = {
        "protocol": PROTOCOL, "split": "meta_calibration",
        "processed_images": stop, **provenance,
        "rows": {name: torch.tensor(value, dtype=torch.int32) for name, value in rows.items()},
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    if stop == len(dataset):
        checks = (
            ("image_index", reference["image_index"]),
            ("region_label", reference["region_label"]),
            ("cluster_index", reference["cluster_index"]),
            ("direction_code", reference["direction_code"]),
            ("area", reference["cluster_area"]),
        )
        for name, expected in checks:
            if not torch.equal(payload["rows"][name].long(), expected.long()):
                raise RuntimeError(f"geometry/reference cluster alignment failed: {name}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = {
        "protocol": PROTOCOL, "processed_images": stop,
        "clusters": len(rows["area"]), "aligned_to_reference": stop == len(dataset),
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
