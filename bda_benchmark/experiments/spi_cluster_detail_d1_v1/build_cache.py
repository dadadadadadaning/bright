from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import RGB_MEAN, RGB_STD
from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.core import (
    cluster_delta_histogram, compress_channels, iter_transition_clusters,
    pool_map_channels, pool_map_names, role_feature_vector, role_masks,
    shuffled_cluster_mask, spatial_probability_features, to_nchw_stage,
    utility_counts,
)
from bda_benchmark.experiments.spi_s2_evidence_v2.core import (
    direct_difference_evidence, matching_evidence,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels, predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram, file_sha256, p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import PROTOCOL
from .core import (
    fixed_edge_maps, fixed_sar_texture_maps, pooled_map_features,
    structural_relation_features,
)


GROUPS = (
    "spatial", "stage2_role", "rgb_edge", "sar_texture", "stage2_direct",
    "stage2_matching", "structure_relation", "shuffled_rgb_edge",
    "shuffled_sar_texture", "shuffled_stage2_matching", "shuffled_structure_relation",
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Stage2Capture:
    def __init__(self, model) -> None:
        self.outputs = {}
        self.handles = [
            model.encoder_1.register_forward_hook(self._hook("rgb")),
            model.encoder_2.register_forward_hook(self._hook("sar")),
        ]

    def _hook(self, name):
        def capture(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise RuntimeError("P3 encoder did not expose Stage2")
            self.outputs[name] = to_nchw_stage(output[2].detach(), 192)
        return capture

    def stages(self):
        if set(self.outputs) != {"rgb", "sar"}:
            raise RuntimeError("both frozen encoders must execute")
        return self.outputs

    def close(self):
        for handle in self.handles:
            handle.remove()


def _expanded(spec, shape, radius):
    h, w = shape
    sy, sx = spec.slices
    y0, y1 = max(0, sy.start - radius), min(h, sy.stop + radius)
    x0, x1 = max(0, sx.start - radius), min(w, sx.stop + radius)
    cluster = torch.zeros((y1 - y0, x1 - x0), dtype=torch.bool)
    cluster[sy.start-y0:sy.stop-y0, sx.start-x0:sx.stop-x0] = spec.cluster
    return slice(y0, y1), slice(x0, x1), cluster


def _rgb_gray(normalized_rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(RGB_MEAN, device=normalized_rgb.device)[None, :, None, None]
    std = torch.as_tensor(RGB_STD, device=normalized_rgb.device)[None, :, None, None]
    rgb = (normalized_rgb.float() * std + mean) / 255.0
    weights = torch.tensor((0.299, 0.587, 0.114), device=rgb.device)[None, :, None, None]
    return (rgb * weights).sum(1, keepdim=True)


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build D1 high-resolution evidence cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=("meta_train", "meta_calibration", "meta_confirm"),
        required=True,
    )
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--minimum-building-area", type=int, default=32)
    parser.add_argument("--minimum-cluster-area", type=int, default=33)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--matching-window", type=int, default=5)
    parser.add_argument("--halo-radius", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    capture = Stage2Capture(p3)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    config = SelectiveContextConfig()
    output = Path(args.output)
    state_path = output.with_suffix(output.suffix + ".partial.pt")
    signature = {
        "protocol": PROTOCOL, "split": args.split,
        "p3_sha256": file_sha256(args.checkpoint),
        "list_path_sha256": file_sha256(args.list_path),
        "minimum_building_area": args.minimum_building_area,
        "minimum_cluster_area": args.minimum_cluster_area,
        "feature_channels": args.feature_channels, "matching_window": args.matching_window,
        "halo_radius": args.halo_radius, "stop": stop, "seed": args.seed,
    }
    state = {
        "processed": 0,
        "image_names": [], "image_events": [],
        "features": {name: [] for name in GROUPS}, "feature_names": {},
        "utility": {name: [] for name in ("changed", "corrected", "harmed", "neutral", "net", "utility_ratio")},
        "image_index": [], "region_label": [], "cluster_index": [],
        "direction_code": [], "cluster_area": [], "delta_hist": [],
        "p3_histograms": [], "fixed_s2_histograms": [],
        "counts": {"regions_all": 0, "clusters_seen": 0, "clusters_eligible": 0, "eligible_pixels": 0},
    }
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("D1 resume signature mismatch; use a new CACHE_ROOT")
        state = saved["state"]
        print(f"Resuming Cluster D1 {args.split} after {state['processed']} images", flush=True)
    initial = int(state["processed"])
    loader = DataLoader(
        Subset(dataset, range(initial, stop)), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    started = time.perf_counter()
    try:
        for offset, (rgb_cpu, sar_cpu, names) in enumerate(loader):
            image_index = initial + offset
            rgb = rgb_cpu.to(device, non_blocking=True)
            sar_native = sar_cpu.to(device, non_blocking=True)
            localization, damage, _ = p3(rgb, p3_sar_view(sar_native), return_aux=True)
            probabilities = damage[0].float().softmax(0).cpu()
            base = probabilities.argmax(0)
            regions = predicted_building_label_map(
                localization[0].argmax(0).cpu(), args.minimum_building_area
            )
            candidate_result = fixed_s2_candidate_from_labels(probabilities, regions, config)
            candidate_probabilities = candidate_result.probabilities.float().cpu()
            candidate = candidate_probabilities.argmax(0)
            target = load_semantic_target(args.dataset_path, str(names[0])).long()
            output_size = tuple(base.shape)

            stages = {name: compress_channels(value, args.feature_channels) for name, value in capture.stages().items()}
            stage_maps = {
                name: F.interpolate(value.float(), output_size, mode="bilinear", align_corners=False).half()
                for name, value in stages.items()
            }
            direct_item = direct_difference_evidence(stages["rgb"], stages["sar"], stages["rgb"].shape[-2:])
            matching_item = matching_evidence(
                stages["rgb"], stages["sar"], stages["rgb"].shape[-2:],
                max_side=max(stages["rgb"].shape[-2:]), window_size=args.matching_window,
            )
            direct_map = F.interpolate(direct_item.values.float(), output_size, mode="bilinear", align_corners=False).half()
            matching_map = F.interpolate(matching_item.values.float(), output_size, mode="bilinear", align_corners=False).half()
            rgb_edge, rgb_edge_names = fixed_edge_maps(_rgb_gray(rgb))
            # channel 1 is the native/raw SAR view used for robust texture evidence.
            sar_raw = sar_native[:, 1:2] if sar_native.shape[1] > 1 else sar_native[:, :1]
            sar_texture, sar_texture_names = fixed_sar_texture_maps(sar_raw)

            margin = probabilities.topk(2, dim=0).values
            margin = margin[0] - margin[1]
            entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(0) / 1.3862943611198906
            unchanged = base == candidate
            state["counts"]["regions_all"] += int(regions.count)
            state["p3_histograms"].append(fast_histogram(base, target).long())
            state["fixed_s2_histograms"].append(fast_histogram(candidate, target).long())
            specs = list(iter_transition_clusters(base, candidate, regions.labels))
            state["counts"]["clusters_seen"] += len(specs)
            distance_cache = {}
            for spec in specs:
                area = int(spec.cluster.sum())
                if area < args.minimum_cluster_area:
                    continue
                cy, cx, local_cluster = _expanded(spec, output_size, args.halo_radius)
                local_building = regions.labels[cy, cx] == spec.region_label
                roles, metadata = role_masks(
                    local_cluster, local_building, margin[cy, cx], unchanged[cy, cx], args.halo_radius
                )
                full_cluster = torch.zeros(output_size, dtype=torch.bool)
                full_cluster[cy, cx] = local_cluster
                utility = utility_counts(base, candidate, target, full_cluster)
                if not utility["changed"]:
                    continue
                if spec.region_label not in distance_cache:
                    distance_cache[spec.region_label] = torch.from_numpy(
                        ndimage.distance_transform_edt(np.asarray(spec.building))
                    ).float()
                spatial, spatial_names = spatial_probability_features(
                    spec, probabilities, candidate_probabilities,
                    distance_map=distance_cache[spec.region_label], margin_map=margin,
                    entropy_map=entropy,
                )
                spatial = torch.cat((spatial, torch.tensor(tuple(metadata.values()))))
                state["feature_names"].setdefault("spatial", spatial_names + tuple(metadata.keys()))
                features = {"spatial": spatial}

                role_rows, role_names = [], []
                for modality in ("rgb", "sar"):
                    row, names_role, _ = role_feature_vector(
                        stage_maps[modality][:, :, cy, cx], roles, f"{modality}_stage2"
                    )
                    role_rows.append(row.cpu()); role_names.extend(names_role)
                features["stage2_role"] = torch.cat(role_rows)
                state["feature_names"].setdefault("stage2_role", tuple(role_names))

                features["rgb_edge"], names_edge = pooled_map_features(
                    rgb_edge[:, :, cy, cx], roles, "rgb", rgb_edge_names
                )
                features["sar_texture"], names_sar = pooled_map_features(
                    sar_texture[:, :, cy, cx], roles, "sar", sar_texture_names
                )
                features["stage2_direct"] = pool_map_channels(direct_map[:, :, cy, cx], roles).cpu()
                features["stage2_matching"] = pool_map_channels(matching_map[:, :, cy, cx], roles).cpu()
                features["structure_relation"], names_relation = structural_relation_features(
                    rgb_edge[:, :, cy, cx], sar_texture[:, :, cy, cx], roles
                )
                state["feature_names"].setdefault("rgb_edge", names_edge)
                state["feature_names"].setdefault("sar_texture", names_sar)
                state["feature_names"].setdefault("stage2_direct", pool_map_names("stage2", tuple(direct_item.channel_names)))
                state["feature_names"].setdefault("stage2_matching", pool_map_names("stage2", tuple(matching_item.channel_names)))
                state["feature_names"].setdefault("structure_relation", names_relation)

                shuffled = shuffled_cluster_mask(
                    local_building, area, args.seed + image_index * 1000003 + len(state["image_index"])
                )
                shuffled_roles, _ = role_masks(
                    shuffled, local_building, margin[cy, cx], unchanged[cy, cx], args.halo_radius
                )
                for source, key in ((rgb_edge, "rgb_edge"), (sar_texture, "sar_texture")):
                    features[f"shuffled_{key}"] = pool_map_channels(source[:, :, cy, cx], shuffled_roles).cpu()
                    state["feature_names"].setdefault(f"shuffled_{key}", state["feature_names"][key])
                features["shuffled_stage2_matching"] = pool_map_channels(
                    matching_map[:, :, cy, cx], shuffled_roles
                ).cpu()
                features["shuffled_structure_relation"], _ = structural_relation_features(
                    rgb_edge[:, :, cy, cx], sar_texture[:, :, cy, cx], shuffled_roles
                )
                state["feature_names"].setdefault("shuffled_stage2_matching", state["feature_names"]["stage2_matching"])
                state["feature_names"].setdefault("shuffled_structure_relation", names_relation)

                for group in GROUPS:
                    row = features[group].float().cpu()
                    if row.numel() != len(state["feature_names"][group]) or not bool(torch.isfinite(row).all()):
                        raise RuntimeError(f"invalid D1 feature {group}")
                    # The cache is stored in float16 for size.  Saturate large
                    # finite area-like features before conversion so they do
                    # not silently become +/-inf and poison downstream
                    # standardization.
                    half_max = float(torch.finfo(torch.float16).max)
                    state["features"][group].append(row.clamp(-half_max, half_max).half())
                for key in state["utility"]:
                    state["utility"][key].append(utility[key])
                state["image_index"].append(image_index)
                state["region_label"].append(spec.region_label)
                state["cluster_index"].append(spec.cluster_index)
                state["direction_code"].append(spec.transition)
                state["cluster_area"].append(area)
                state["delta_hist"].append(cluster_delta_histogram(base, candidate, target, full_cluster))
                state["counts"]["clusters_eligible"] += 1
                state["counts"]["eligible_pixels"] += area

            sample_name = str(names[0])
            state["image_names"].append(sample_name)
            state["image_events"].append(sample_name.rsplit("_", 1)[0])
            state["processed"] = image_index + 1
            if (image_index + 1) % args.progress_every == 0:
                speed = (image_index + 1 - initial) / max(time.perf_counter() - started, 1e-9)
                print(f"Cluster D1 cache {args.split}: {image_index+1}/{stop} | {speed:.2f} img/s | eligible={state['counts']['clusters_eligible']}", flush=True)
            if (image_index + 1) % args.checkpoint_every == 0:
                _atomic_save({"signature": signature, "state": state}, state_path)
    finally:
        capture.close()

    if not state["image_index"]:
        raise RuntimeError("D1 found no eligible clusters")
    payload = {
        **signature, "configuration": vars(args), **provenance,
        "s2_configuration": config.__dict__,
        "processed_images": stop, "image_names": state["image_names"],
        "image_events": state["image_events"], "feature_names": state["feature_names"],
        "features": {name: torch.stack(rows) for name, rows in state["features"].items()},
        "utility": {name: torch.tensor(rows) for name, rows in state["utility"].items()},
        "image_index": torch.tensor(state["image_index"], dtype=torch.int32),
        "region_label": torch.tensor(state["region_label"], dtype=torch.int32),
        "cluster_index": torch.tensor(state["cluster_index"], dtype=torch.int32),
        "direction_code": torch.tensor(state["direction_code"], dtype=torch.uint8),
        "cluster_area": torch.tensor(state["cluster_area"], dtype=torch.int32),
        "delta_hist": torch.stack(state["delta_hist"]).to(torch.int32),
        "p3_histograms": torch.stack(state["p3_histograms"]),
        "fixed_s2_histograms": torch.stack(state["fixed_s2_histograms"]),
        "counts": state["counts"],
        "meta_confirm_accessed": args.split == "meta_confirm",
        "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    _atomic_save(payload, output)
    if state_path.exists():
        state_path.unlink()
    manifest = {"protocol": PROTOCOL, "split": args.split, "processed_images": stop,
                "counts": state["counts"], "feature_dimensions": {k: int(v.shape[1]) for k, v in payload["features"].items()},
                "meta_confirm_accessed": payload["meta_confirm_accessed"], "val450_accessed": False,
                "official_instances_accessed": False, "test_accessed": False}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    if args.split == "meta_confirm":
        print(
            "Meta-confirm accessed: ONCE. "
            "Val450/official instances/test accessed: NEVER / NEVER / NEVER",
            flush=True,
        )
    else:
        print(
            "Meta-confirm/Val450/official instances/test accessed: "
            "NEVER / NEVER / NEVER / NEVER",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
