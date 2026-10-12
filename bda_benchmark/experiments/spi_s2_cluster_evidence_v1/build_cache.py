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

from bda_benchmark.experiments.spi_s2_evidence_v2.core import (
    direct_difference_evidence,
    matching_evidence,
    relation_evidence,
)
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
from .core import (
    ROLE_NAMES,
    cluster_delta_histogram,
    compress_channels,
    iter_transition_clusters,
    pool_map_channels,
    pool_map_names,
    role_feature_vector,
    role_masks,
    shuffled_cluster_mask,
    spatial_probability_features,
    to_nchw_stage,
    utility_counts,
)


GROUPS = ("spatial", "rgb_role", "sar_role", "direct", "matching", "relation")
DEFAULT_GROUPS = ",".join(GROUPS)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class StageCapture:
    """Read-only capture of the two frozen P3 encoder outputs."""

    def __init__(self, model) -> None:
        self.outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = [
            model.encoder_1.register_forward_hook(self._hook("rgb")),
            model.encoder_2.register_forward_hook(self._hook("sar")),
        ]

    def _hook(self, name: str):
        def capture(_module, _inputs, output) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 5:
                raise RuntimeError("P3 encoder did not expose stage1--stage4 outputs")
            self.outputs[name] = (
                to_nchw_stage(output[3].detach(), 384),
                to_nchw_stage(output[4].detach(), 768),
            )
        return capture

    def stages(self) -> dict[str, dict[str, torch.Tensor]]:
        if set(self.outputs) != {"rgb", "sar"}:
            raise RuntimeError("both frozen P3 encoders must run before reading stages")
        return {
            name: {"stage3": rows[0], "stage4": rows[1]}
            for name, rows in self.outputs.items()
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _expanded_crop(spec, shape: tuple[int, int], radius: int) -> tuple[slice, slice, torch.Tensor]:
    height, width = shape
    source_y, source_x = spec.slices
    y0, y1 = max(0, source_y.start - radius), min(height, source_y.stop + radius)
    x0, x1 = max(0, source_x.start - radius), min(width, source_x.stop + radius)
    expanded = torch.zeros((y1 - y0, x1 - x0), dtype=torch.bool)
    oy, ox = source_y.start - y0, source_x.start - x0
    expanded[oy:oy + spec.cluster.shape[0], ox:ox + spec.cluster.shape[1]] = spec.cluster
    return slice(y0, y1), slice(x0, x1), expanded


def _upsample(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(feature.float(), size=size, mode="bilinear", align_corners=False).half()


def _image_maps(
    stages: dict[str, dict[str, torch.Tensor]],
    output_size: tuple[int, int],
    channels: int,
    matching_window: int,
    enabled: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, tuple[str, ...]]],
]:
    compressed = {
        modality: {
            stage: compress_channels(value, channels)
            for stage, value in rows.items()
        }
        for modality, rows in stages.items()
    }
    role_maps = {
        modality: {stage: _upsample(value, output_size) for stage, value in rows.items()}
        for modality, rows in compressed.items()
    }
    evidence_maps: dict[str, dict[str, torch.Tensor]] = {}
    names: dict[str, dict[str, tuple[str, ...]]] = {}
    builders = {}
    if "direct" in enabled:
        builders["direct"] = lambda r, s: direct_difference_evidence(r, s, r.shape[-2:])
    if "matching" in enabled:
        builders["matching"] = lambda r, s: matching_evidence(
            r, s, r.shape[-2:], max_side=max(r.shape[-2:]), window_size=matching_window
        )
    if "relation" in enabled:
        builders["relation"] = lambda r, s: relation_evidence(r, s, r.shape[-2:])
    for group, builder in builders.items():
        evidence_maps[group], names[group] = {}, {}
        for stage in ("stage3", "stage4"):
            item = builder(compressed["rgb"][stage], compressed["sar"][stage])
            evidence_maps[group][stage] = _upsample(item.values, output_size)
            names[group][stage] = tuple(item.channel_names)
    return role_maps, evidence_maps, names


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build cluster-level RGB-SAR utility evidence cache")
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--matching-window", type=int, default=5)
    parser.add_argument("--halo-radius", type=int, default=2)
    parser.add_argument("--max-clusters-per-image", type=int, default=0)
    parser.add_argument("--groups", default=DEFAULT_GROUPS)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--state-every", type=int, default=50)
    parser.add_argument(
        "--profile-timing", action="store_true",
        default=os.environ.get("PROFILE_TIMING", "0") == "1",
    )
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    raw_groups = str(args.groups).strip()
    if raw_groups.lower() in {"", "0", "false", "off", "none", "null"}:
        raw_groups = DEFAULT_GROUPS
        print(
            f"Normalized disabled/empty --groups={args.groups!r} to {DEFAULT_GROUPS}",
            flush=True,
        )
    requested = tuple(
        dict.fromkeys(value.strip() for value in raw_groups.split(",") if value.strip())
    )
    unknown = [name for name in requested if name not in GROUPS]
    if unknown:
        raise ValueError(f"unknown cluster evidence groups: {unknown}")
    if "spatial" not in requested:
        requested = ("spatial",) + requested
    enabled_evidence = tuple(name for name in ("direct", "matching", "relation") if name in requested)
    # Stage3 is the preregistered main hypothesis because most modification
    # clusters are tiny. Stage4 is cached as a separately named diagnostic and
    # is never silently concatenated into the main evidence group.
    stage4_groups = tuple(
        f"{name}_stage4" for name in requested if name != "spatial"
    )
    shuffled_groups = tuple(f"shuffled_{name}" for name in enabled_evidence)
    shuffled_stage4_groups = tuple(
        f"shuffled_{name}_stage4" for name in enabled_evidence
    )
    stored_groups = tuple(requested) + stage4_groups + shuffled_groups + shuffled_stage4_groups

    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    capture = StageCapture(p3)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, observed {len(dataset)}")
    output = Path(args.output)
    state_path = output.with_suffix(output.suffix + ".partial.pt")
    signature = {
        "protocol": PROTOCOL,
        "split": args.split,
        "p3_sha256": file_sha256(args.checkpoint),
        "list_path_sha256": file_sha256(args.list_path),
        "minimum_area": args.minimum_area,
        "feature_channels": args.feature_channels,
        "matching_window": args.matching_window,
        "halo_radius": args.halo_radius,
        "max_clusters_per_image": args.max_clusters_per_image,
        "groups": stored_groups,
        "seed": args.seed,
    }
    state = {
        "processed": 0,
        "image_names": [], "image_events": [],
        "features": {name: [] for name in stored_groups},
        "feature_names": {}, "memory_key": [],
        "utility": {name: [] for name in ("changed", "corrected", "harmed", "neutral", "net", "utility_ratio")},
        "image_index": [], "region_label": [], "cluster_index": [], "direction_code": [], "cluster_area": [],
        "delta_hist": [], "p3_histograms": [], "fixed_s2_histograms": [],
        "counts": {
            "regions_all": 0, "changed_buildings": 0,
            "clusters_seen": 0, "clusters": 0, "changed_pixels": 0,
        },
    }
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("cluster-evidence resume signature mismatch")
        state = saved["state"]
        print(f"Resuming Cluster Evidence {args.split} after {state['processed']} images", flush=True)

    initial = int(state["processed"])
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    loader = DataLoader(
        Subset(dataset, range(initial, stop)), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    config = SelectiveContextConfig()
    started = time.perf_counter()
    timing = {name: 0.0 for name in ("p3", "candidate", "evidence", "clusters", "total")}
    try:
        for offset, (rgb, sar, names) in enumerate(loader):
            image_started = time.perf_counter()
            image_index = initial + offset
            rgb = rgb.to(device, non_blocking=True)
            sar = p3_sar_view(sar.to(device, non_blocking=True))
            localization, damage, _auxiliary = p3(rgb, sar, return_aux=True)
            p3_finished = time.perf_counter()
            probabilities = damage[0].float().softmax(0).cpu()
            base = probabilities.argmax(0)
            regions = predicted_building_label_map(localization[0].argmax(0).cpu(), args.minimum_area)
            candidate_result = fixed_s2_candidate_from_labels(probabilities, regions, config)
            candidate_probabilities = candidate_result.probabilities.float().cpu()
            candidate = candidate_probabilities.argmax(0)
            sample_name = str(names[0])
            target = load_semantic_target(args.dataset_path, sample_name).long()
            candidate_finished = time.perf_counter()
            output_size = tuple(base.shape)
            role_feature_maps, evidence_maps, evidence_names = _image_maps(
                capture.stages(), output_size, args.feature_channels,
                args.matching_window, enabled_evidence,
            )
            evidence_finished = time.perf_counter()
            margin = probabilities.topk(2, dim=0).values
            margin = margin[0] - margin[1]
            entropy = -(
                probabilities.clamp_min(1e-8)
                * probabilities.clamp_min(1e-8).log()
            ).sum(0) / 1.3862943611198906
            unchanged = base == candidate
            changed_labels = torch.unique(regions.labels[(regions.labels > 0) & (base != candidate)])
            state["counts"]["regions_all"] += int(regions.count)
            state["counts"]["changed_buildings"] += int(changed_labels.numel())
            state["p3_histograms"].append(fast_histogram(base, target).long())
            state["fixed_s2_histograms"].append(fast_histogram(candidate, target).long())

            specs = list(iter_transition_clusters(base, candidate, regions.labels))
            state["counts"]["clusters_seen"] += len(specs)
            if args.max_clusters_per_image > 0 and len(specs) > args.max_clusters_per_image:
                generator = torch.Generator().manual_seed(args.seed + image_index * 1000003)
                selected = torch.randperm(len(specs), generator=generator)[
                    : args.max_clusters_per_image
                ].tolist()
                specs = [specs[index] for index in sorted(selected)]
            distance_cache: dict[int, torch.Tensor] = {}
            for spec in specs:
                crop_y, crop_x, cluster_local = _expanded_crop(spec, output_size, args.halo_radius)
                building_local = regions.labels[crop_y, crop_x] == spec.region_label
                local_roles, role_metadata = role_masks(
                    cluster_local, building_local, margin[crop_y, crop_x],
                    unchanged[crop_y, crop_x], args.halo_radius,
                )
                full_cluster = torch.zeros(output_size, dtype=torch.bool)
                full_cluster[crop_y, crop_x] = cluster_local
                utility = utility_counts(base, candidate, target, full_cluster)
                if int(utility["changed"]) <= 0:
                    continue

                features: dict[str, torch.Tensor] = {}
                if spec.region_label not in distance_cache:
                    distance_cache[spec.region_label] = torch.from_numpy(
                        ndimage.distance_transform_edt(np.asarray(spec.building))
                    ).float()
                spatial, spatial_names = spatial_probability_features(
                    spec, probabilities, candidate_probabilities,
                    distance_map=distance_cache[spec.region_label],
                    margin_map=margin, entropy_map=entropy,
                )
                spatial = torch.cat((spatial, torch.tensor(tuple(role_metadata.values()))))
                features["spatial"] = spatial
                current_spatial_names = spatial_names + tuple(role_metadata.keys())
                state["feature_names"].setdefault("spatial", current_spatial_names)

                memory_rows = []
                for modality, group in (("rgb", "rgb_role"), ("sar", "sar_role")):
                    for stage in ("stage3", "stage4"):
                        feature_crop = role_feature_maps[modality][stage][:, :, crop_y, crop_x]
                        summary, summary_names, memory = role_feature_vector(
                            feature_crop, local_roles, f"{stage}"
                        )
                        key = group if stage == "stage3" else f"{group}_stage4"
                        features[key] = summary.cpu()
                        state["feature_names"].setdefault(key, tuple(summary_names))
                        memory_rows.append(F.normalize(memory.float(), dim=0).cpu())

                shuffled = shuffled_cluster_mask(
                    building_local, int(cluster_local.sum()),
                    args.seed + image_index * 1000003 + state["counts"]["clusters"],
                )
                shuffled_roles, _ = role_masks(
                    shuffled, building_local, margin[crop_y, crop_x],
                    unchanged[crop_y, crop_x], args.halo_radius,
                )
                for group in enabled_evidence:
                    for stage in ("stage3", "stage4"):
                        suffix = "" if stage == "stage3" else "_stage4"
                        key = f"{group}{suffix}"
                        control_key = f"shuffled_{group}{suffix}"
                        evidence_crop = evidence_maps[group][stage][:, :, crop_y, crop_x]
                        features[key] = pool_map_channels(evidence_crop, local_roles).cpu()
                        features[control_key] = pool_map_channels(
                            evidence_crop, shuffled_roles
                        ).cpu()
                        feature_names = pool_map_names(
                            stage, evidence_names[group][stage]
                        )
                        state["feature_names"].setdefault(key, feature_names)
                        state["feature_names"].setdefault(control_key, feature_names)

                for group in stored_groups:
                    row = features[group].float()
                    if not bool(torch.isfinite(row).all()):
                        raise FloatingPointError(f"non-finite {group} cluster feature")
                    expected_names = state["feature_names"][group]
                    if row.numel() != len(expected_names):
                        raise RuntimeError(
                            f"{group} feature/name mismatch: {row.numel()} vs {len(expected_names)}"
                        )
                    state["features"][group].append(row.half())
                state["memory_key"].append(torch.cat(memory_rows).half())
                for key in state["utility"]:
                    state["utility"][key].append(utility[key])
                state["image_index"].append(image_index)
                state["region_label"].append(spec.region_label)
                state["cluster_index"].append(spec.cluster_index)
                state["direction_code"].append(spec.transition)
                state["cluster_area"].append(int(cluster_local.sum()))
                state["delta_hist"].append(cluster_delta_histogram(base, candidate, target, full_cluster))
                state["counts"]["clusters"] += 1
                state["counts"]["changed_pixels"] += int(utility["changed"])

            state["image_names"].append(sample_name)
            state["image_events"].append(sample_name.rsplit("_", 1)[0])
            state["processed"] = image_index + 1
            image_finished = time.perf_counter()
            timing["p3"] += p3_finished - image_started
            timing["candidate"] += candidate_finished - p3_finished
            timing["evidence"] += evidence_finished - candidate_finished
            timing["clusters"] += image_finished - evidence_finished
            timing["total"] += image_finished - image_started
            if state["processed"] % args.progress_every == 0:
                speed = (state["processed"] - initial) / max(time.perf_counter() - started, 1e-9)
                print(
                    f"Cluster Evidence cache {args.split}: {state['processed']}/{stop} | "
                    f"{speed:.2f} img/s | clusters={state['counts']['clusters']}", flush=True
                )
                if args.profile_timing:
                    processed_now = max(state["processed"] - initial, 1)
                    means = {
                        name: round(1000.0 * value / processed_now, 2)
                        for name, value in timing.items()
                    }
                    print(f"Cluster Evidence timing mean ms/image: {means}", flush=True)
            if state["processed"] % args.state_every == 0:
                _atomic_save({"signature": signature, "state": state}, state_path)
    finally:
        capture.close()

    if not state["image_index"]:
        raise RuntimeError("no modification clusters were cached")
    payload = {
        **signature,
        "configuration": vars(args), **provenance,
        "s2_configuration": config.__dict__,
        "processed_images": state["processed"],
        "image_names": state["image_names"], "image_events": state["image_events"],
        "feature_names": {name: tuple(rows) for name, rows in state["feature_names"].items()},
        "features": {name: torch.stack(rows) for name, rows in state["features"].items()},
        "memory_key": torch.stack(state["memory_key"]),
        "utility": {
            "changed": torch.tensor(state["utility"]["changed"], dtype=torch.int32),
            "corrected": torch.tensor(state["utility"]["corrected"], dtype=torch.int32),
            "harmed": torch.tensor(state["utility"]["harmed"], dtype=torch.int32),
            "neutral": torch.tensor(state["utility"]["neutral"], dtype=torch.int32),
            "net": torch.tensor(state["utility"]["net"], dtype=torch.int32),
            "utility_ratio": torch.tensor(state["utility"]["utility_ratio"], dtype=torch.float32),
        },
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
    if args.max_clusters_per_image == 0:
        reconstructed = payload["p3_histograms"].sum(0) + payload["delta_hist"].sum(0)
        expected_fixed = payload["fixed_s2_histograms"].sum(0)
        if not torch.equal(reconstructed.to(expected_fixed.dtype), expected_fixed):
            raise RuntimeError(
                "cluster deltas do not exactly reconstruct fixed S2; cache is incomplete"
            )
    _atomic_save(payload, output)
    if state_path.exists():
        state_path.unlink()
    manifest = {
        "protocol": PROTOCOL, "split": args.split, "processed_images": state["processed"],
        "counts": state["counts"],
        "feature_dimensions": {name: int(value.shape[1]) for name, value in payload["features"].items()},
        "memory_key_dimension": int(payload["memory_key"].shape[1]),
        "meta_confirm_accessed": payload["meta_confirm_accessed"],
        "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    print("Val450/official instances/test accessed: NEVER / NEVER / NEVER", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
