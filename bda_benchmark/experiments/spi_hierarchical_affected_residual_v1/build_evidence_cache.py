from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import RGB_MEAN, RGB_STD
from bda_benchmark.experiments.spi_cluster_detail_d1_v1.build_cache import Stage2Capture
from bda_benchmark.experiments.spi_cluster_detail_d1_v1.core import fixed_edge_maps, fixed_sar_texture_maps
from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.core import compress_channels
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    fixed_s2_candidate_from_labels, predicted_building_label_map,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.runtime import load_frozen_p3
from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256, p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from . import EVIDENCE_CACHE_PROTOCOL
from .core import (
    AO_OUTCOME_CORRECTED, AO_OUTCOME_HARMED, affected_candidate_outcomes,
    deterministic_subsample, minimum_affected_residual, severity_choice,
)


GROUPS = ("probability", "rgb_structure", "sar_texture", "instance_relative", "cross_relation", "shuffled_relation")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rgb_gray(normalized_rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(RGB_MEAN, device=normalized_rgb.device)[None, :, None, None]
    std = torch.as_tensor(RGB_STD, device=normalized_rgb.device)[None, :, None, None]
    rgb = (normalized_rgb.float() * std + mean) / 255.0
    weights = torch.tensor((0.299, 0.587, 0.114), device=rgb.device)[None, :, None, None]
    return (rgb * weights).sum(1, keepdim=True)


def _anchor_relative(maps: torch.Tensor, labels: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense feature-minus-stable-anchor maps and anchor vectors."""

    result = torch.zeros_like(maps.float())
    anchors = torch.zeros_like(maps.float())
    labels_device = labels.to(maps.device)
    anchor_device = anchor.to(maps.device)
    for label in range(1, int(labels.max()) + 1):
        region = labels_device == label
        reference = region & anchor_device
        if not bool(reference.any()):
            reference = region
        vector = maps[0, :, reference].float().mean(1)
        result[0, :, region] = maps[0, :, region].float() - vector[:, None]
        anchors[0, :, region] = vector[:, None]
    return result, anchors


def _summary(relative: torch.Tensor, original: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    absolute = relative.abs()
    cosine = F.cosine_similarity(original.float(), anchor.float(), dim=1, eps=1e-6)[:, None]
    return torch.cat((
        absolute.mean(1, keepdim=True),
        absolute.square().mean(1, keepdim=True).sqrt(),
        absolute.amax(1, keepdim=True), cosine,
    ), dim=1)


def _probability_maps(p3: torch.Tensor, fixed: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    entropy = -(fixed.clamp_min(1e-8) * fixed.clamp_min(1e-8).log()).sum(0, keepdim=True)
    severity = fixed[2:4]
    affected = severity.sum(0, keepdim=True)
    nonaffected = fixed[:2].amax(0, keepdim=True)
    severity_margin = (severity[0] - severity[1]).abs()[None]
    maps = torch.cat((p3, fixed, affected, nonaffected, affected - nonaffected, entropy, severity_margin), dim=0)[None]
    names = tuple([f"p3_p{n}" for n in range(4)] + [f"s2_p{n}" for n in range(4)] + [
        "affected_probability", "nonaffected_probability", "affected_margin", "entropy", "severity_margin",
    ])
    return maps, names


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build AO-E1 prediction-only candidate evidence cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--detail-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("meta_train", "meta_calibration"), required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--minimum-building-area", type=int, default=32)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--maximum-candidates-per-image", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    detail = json.loads(Path(args.detail_json).read_text(encoding="utf-8"))
    if detail.get("decision") != "ADMIT_AO_E1_EVIDENCE_AUDIT":
        raise RuntimeError("AO-E1 refused: AO-D0 did not pass concentration/localization checks")
    device = torch.device(args.device)
    p3, provenance = load_frozen_p3(args.checkpoint, device)
    capture = Stage2Capture(p3)
    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, got {len(dataset)}")
    stop = min(len(dataset), args.max_images) if args.max_images > 0 else len(dataset)
    output = Path(args.output); state_path = output.with_suffix(output.suffix + ".partial.pt")
    signature = {
        "protocol": EVIDENCE_CACHE_PROTOCOL, "split": args.split,
        "p3_sha256": file_sha256(args.checkpoint), "list_sha256": file_sha256(args.list_path),
        "minimum_building_area": args.minimum_building_area, "feature_channels": args.feature_channels,
        "maximum_candidates_per_image": args.maximum_candidates_per_image, "stop": stop, "seed": args.seed,
    }
    state = {
        "processed": 0, "image_names": [], "image_events": [],
        "features": {name: [] for name in GROUPS}, "feature_names": {},
        "outcome": [], "image_index": [], "region_label": [], "direction_code": [],
        "required_residual": [], "target_class": [],
        "counts": {"candidates_seen": 0, "candidates_stored": 0, "corrected": 0, "harmed": 0, "neutral": 0},
    }
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("AO-E1 resume signature mismatch; choose a new CACHE_ROOT")
        state = saved["state"]
        print(f"Resuming AO-E1 {args.split} after {state['processed']} images", flush=True)
    initial = int(state["processed"])
    loader = DataLoader(Subset(dataset, range(initial, stop)), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0)
    config = SelectiveContextConfig(); started = time.perf_counter()
    try:
        for offset, (rgb_cpu, sar_cpu, names) in enumerate(loader):
            image_index = initial + offset
            rgb = rgb_cpu.to(device, non_blocking=True); sar = sar_cpu.to(device, non_blocking=True)
            localization, damage, _ = p3(rgb, p3_sar_view(sar), return_aux=True)
            p3_probability = damage[0].float().softmax(0).cpu()
            regions = predicted_building_label_map(localization[0].argmax(0).cpu(), args.minimum_building_area)
            fixed_probability = fixed_s2_candidate_from_labels(p3_probability, regions, config).probabilities.float().cpu()
            target = load_semantic_target(args.dataset_path, str(names[0])).long()
            candidate, outcome = affected_candidate_outcomes(fixed_probability, regions.labels > 0, target)
            sample = deterministic_subsample(candidate, args.maximum_candidates_per_image,
                                             args.seed + image_index * 1000003)
            state["counts"]["candidates_seen"] += int(candidate.sum())
            state["counts"]["candidates_stored"] += int(sample.sum())
            state["counts"]["corrected"] += int((outcome[sample] == AO_OUTCOME_CORRECTED).sum())
            state["counts"]["harmed"] += int((outcome[sample] == AO_OUTCOME_HARMED).sum())
            state["counts"]["neutral"] += int((outcome[sample] == 0).sum())

            output_size = tuple(target.shape)
            stages = {k: F.interpolate(compress_channels(v, args.feature_channels).float(), output_size,
                                       mode="bilinear", align_corners=False)
                      for k, v in capture.stages().items()}
            rgb_edge, rgb_names = fixed_edge_maps(_rgb_gray(rgb))
            sar_raw = sar[:, 1:2] if sar.shape[1] > 1 else sar[:, :1]
            sar_maps, sar_names = fixed_sar_texture_maps(sar_raw)
            fixed_label = fixed_probability.argmax(0)
            confidence = fixed_probability.topk(2, dim=0).values
            stable_anchor = (fixed_label == 1) & ((confidence[0] - confidence[1]) >= 0.20)
            rgb_relative, rgb_anchor = _anchor_relative(stages["rgb"], regions.labels, stable_anchor)
            sar_relative, sar_anchor = _anchor_relative(stages["sar"], regions.labels, stable_anchor)
            edge_relative, _ = _anchor_relative(rgb_edge, regions.labels, stable_anchor)
            texture_relative, _ = _anchor_relative(sar_maps, regions.labels, stable_anchor)
            rgb_summary = _summary(rgb_relative, stages["rgb"], rgb_anchor)
            sar_summary = _summary(sar_relative, stages["sar"], sar_anchor)
            relation = torch.cat((
                (edge_relative[:, :4].abs() - texture_relative[:, :4].abs()).abs(),
                edge_relative[:, :4].abs(), texture_relative[:, :4].abs(),
            ), dim=1)
            # Deterministic negative control: break local correspondence while
            # preserving each image's marginal SAR-relation distribution.
            shuffled = torch.roll(texture_relative[:, :4].abs(), shifts=(17, 29), dims=(2, 3))
            shuffled_relation = torch.cat(((edge_relative[:, :4].abs() - shuffled).abs(),
                                           edge_relative[:, :4].abs(), shuffled), dim=1)
            probability_maps, probability_names = _probability_maps(p3_probability, fixed_probability)
            maps = {
                "probability": probability_maps.to(device),
                "rgb_structure": torch.cat((rgb_edge, edge_relative.abs()), dim=1),
                "sar_texture": torch.cat((sar_maps, texture_relative.abs()), dim=1),
                "instance_relative": torch.cat((rgb_summary, sar_summary,
                                                  (rgb_summary - sar_summary).abs()), dim=1),
                "cross_relation": relation, "shuffled_relation": shuffled_relation,
            }
            state["feature_names"].setdefault("probability", probability_names)
            state["feature_names"].setdefault("rgb_structure", tuple([f"rgb_{x}" for x in rgb_names] + [f"rgb_relative_{x}" for x in rgb_names]))
            state["feature_names"].setdefault("sar_texture", tuple([f"sar_{x}" for x in sar_names] + [f"sar_relative_{x}" for x in sar_names]))
            summary_names = ("mean_abs", "rms", "max_abs", "anchor_cosine")
            state["feature_names"].setdefault("instance_relative", tuple([f"rgb_{x}" for x in summary_names] + [f"sar_{x}" for x in summary_names] + [f"cross_gap_{x}" for x in summary_names]))
            relation_names = tuple([f"cross_gap_{x}" for x in rgb_names[:4]] + [f"rgb_internal_{x}" for x in rgb_names[:4]] + [f"sar_internal_{x}" for x in sar_names[:4]])
            state["feature_names"].setdefault("cross_relation", relation_names)
            state["feature_names"].setdefault("shuffled_relation", relation_names)
            ys, xs = torch.where(sample)
            ys_device, xs_device = ys.to(device), xs.to(device)
            for group in GROUPS:
                rows = maps[group][0, :, ys_device, xs_device].t().float().cpu()
                if not bool(torch.isfinite(rows).all()):
                    raise RuntimeError(f"non-finite AO-E1 feature: {group}")
                state["features"][group].append(rows.half())
            severity = severity_choice(fixed_probability)
            state["outcome"].append(outcome[sample].cpu())
            state["image_index"].append(torch.full((len(ys),), image_index, dtype=torch.int32))
            state["region_label"].append(regions.labels[sample].to(torch.int32))
            state["direction_code"].append((fixed_label[sample] * 2 + severity[sample] - 2).to(torch.uint8))
            state["required_residual"].append(minimum_affected_residual(fixed_probability)[sample].half())
            state["target_class"].append(target[sample].to(torch.int8))
            sample_name = str(names[0]); state["image_names"].append(sample_name)
            state["image_events"].append(sample_name.rsplit("_", 1)[0]); state["processed"] = image_index + 1
            if (image_index + 1) % args.progress_every == 0:
                rate = (image_index + 1 - initial) / max(time.perf_counter() - started, 1e-9)
                print(f"AO-E1 cache {args.split}: {image_index+1}/{stop} | {rate:.2f} img/s | stored={state['counts']['candidates_stored']}", flush=True)
            if (image_index + 1) % args.checkpoint_every == 0:
                _atomic_save({"signature": signature, "state": state}, state_path)
    finally:
        capture.close()

    if not state["outcome"]:
        raise RuntimeError("AO-E1 found no candidates")
    payload = {
        **signature, **provenance, "configuration": vars(args),
        "processed_images": stop, "image_names": state["image_names"], "image_events": state["image_events"],
        "features": {k: torch.cat(v) for k, v in state["features"].items()},
        "feature_names": state["feature_names"],
        "outcome": torch.cat(state["outcome"]), "image_index": torch.cat(state["image_index"]),
        "region_label": torch.cat(state["region_label"]), "direction_code": torch.cat(state["direction_code"]),
        "required_residual": torch.cat(state["required_residual"]), "target_class": torch.cat(state["target_class"]),
        "counts": state["counts"], "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    _atomic_save(payload, output)
    if state_path.exists(): state_path.unlink()
    manifest = {"protocol": EVIDENCE_CACHE_PROTOCOL, "split": args.split, "processed_images": stop,
                "counts": state["counts"], "feature_dimensions": {k: int(v.shape[1]) for k, v in payload["features"].items()},
                "meta_confirm_accessed": False, "val450_accessed": False,
                "official_instances_accessed": False, "test_accessed": False}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2)); print(f"Saved: {output}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
