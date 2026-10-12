#!/usr/bin/env python3
"""BIOS E0: validation-only, parameter-free building relation audit.

The audit compares intra-RGB and intra-SAR self-similarity inside building
ROIs.  It deliberately trains no projection, classifier, threshold or fusion
module and refuses a test split.  A second pass applies controlled SAR-token
shifts and a deterministic spatial shuffle to the validation-selected signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
)
from bda_benchmark.model.region_ops import _connected_components_4  # noqa: E402
from bda_benchmark.script.standard_ML.bios_e0_runtime import (  # noqa: E402
    aggregate_token_scores,
    complete_signal_metrics,
    deterministic_valid_shuffle,
    direct_token_anomaly,
    relation_token_anomaly,
    shift_token_grid,
)
from bda_benchmark.script.standard_ML.train_Q0_SAR_building_readout import (  # noqa: E402
    BEST_PROTOCOL as Q0_PROTOCOL,
    file_sha256,
    load_locked_source,
)


EVENT_RE = re.compile(r"_\d+$")
AGGREGATES = ("mean", "std", "max", "q75", "q90", "top20", "area", "concentration")
SPATIAL_MODES = ("local", "global", "local_global")
RELATION_MODES = ("sim", "sd")
PRIMARY_SIGNAL = "s3_local_sim_mean"
PRIMARY_DIRECT = "s3_direct_mean"

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class BuildingROI:
    batch_index: int
    instance_id: int
    box: tuple[int, int, int, int]
    label: int
    p3_prediction: int
    event: str
    area: int
    purity: float
    building_fraction: float


def event_name(sample: str) -> str:
    return EVENT_RE.sub("", str(sample))


def atomic_json(payload: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def load_preregistration(path: str | os.PathLike[str], args) -> tuple[dict, str]:
    source = Path(path)
    raw = source.read_bytes()
    registration = json.loads(raw.decode("utf-8"))
    if registration.get("protocol") != "BIOS_E0_preregistration_v1":
        raise RuntimeError("unexpected BIOS E0 preregistration protocol")
    primary = registration.get("primary", {})
    roi = registration.get("roi_protocol", {})
    secondary = registration.get("secondary_statistic_protocol", {})
    stress = registration.get("stress", {})
    gate = registration.get("advancement_gate", {})
    expected = {
        "roi_grid_size": args.grid_size,
        "feature_stage": "s3",
        "spatial_relation": "local_3x3",
        "relation_type": "cosine_similarity_discrepancy",
        "aggregate": "mean",
        "signal_key": PRIMARY_SIGNAL,
        "metric": "p3_nonaffected_miss.auroc",
    }
    if any(primary.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"BIOS E0 primary setting differs from preregistration: {primary}")
    locked_roi = {
        "minimum_area": args.minimum_area,
        "p3_proposal_threshold": args.proposal_threshold,
        "minimum_building_fraction": args.minimum_building_fraction,
        "minimum_class_purity": args.minimum_class_purity,
        "minimum_token_mass": args.minimum_token_mass,
        "minimum_valid_tokens": args.minimum_valid_tokens,
    }
    if any(roi.get(key) != value for key, value in locked_roi.items()):
        raise RuntimeError("BIOS E0 ROI protocol differs from preregistration")
    if (
        secondary.get("similarity_dissimilarity_neighbours") != args.neighbour_count
        or secondary.get("area_threshold") != args.relation_threshold
    ):
        raise RuntimeError("BIOS E0 secondary-statistic protocol differs from preregistration")
    if list(stress.get("shift_magnitudes", [])) != list(args.shift_magnitudes):
        raise RuntimeError("BIOS E0 shift settings differ from preregistration")
    locked_gates = {
        "minimum_miss_auroc": 0.60,
        "minimum_event_macro_auroc": 0.58,
        "minimum_direction_correct_events": 10,
        "maximum_gt_p3_auc_gap": args.maximum_gt_p3_auc_gap,
    }
    if any(gate.get(key) != value for key, value in locked_gates.items()):
        raise RuntimeError("BIOS E0 gate differs from preregistration")
    if registration.get("training") != "none" or registration.get("test_access") != "forbidden":
        raise RuntimeError("BIOS E0 preregistration violates no-train/no-test contract")
    return registration, hashlib.sha256(raw).hexdigest()


def load_feature_sources(args, device: torch.device):
    """Load P3 and H6-A2; audit Q0 only as an immutable feature-source lock."""
    p3_checkpoint = torch.load(args.p3_checkpoint, map_location="cpu", weights_only=False)
    p3_state = p3_checkpoint.get("model", p3_checkpoint)
    if not isinstance(p3_state, Mapping) or not p3_state:
        raise RuntimeError("P3 checkpoint has no model state")
    p3 = DamageFormer_S8_MICB_DSMIL()
    incompatible = p3.load_state_dict(p3_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    source, a2_provenance = load_locked_source(args.a2_checkpoint, device)

    feature_lock = torch.load(
        args.feature_lock_checkpoint, map_location="cpu", weights_only=False
    )
    if feature_lock.get("protocol") != Q0_PROTOCOL:
        raise RuntimeError("feature lock is not a Q0 validated checkpoint")
    if feature_lock.get("test_accessed") is not False:
        raise RuntimeError("BIOS refuses a feature lock that accessed test")
    if str(feature_lock.get("variant", "")).lower() != "q0_b0":
        raise RuntimeError("BIOS feature lock must be the practical Q0-B0 checkpoint")
    if not bool(feature_lock.get("metrics", {}).get("advancement_gate", {}).get("passed")):
        raise RuntimeError("BIOS feature lock did not pass its declared validation gate")
    if feature_lock.get("source", {}).get("sha256") != a2_provenance["sha256"]:
        raise RuntimeError("BIOS feature lock/H6-A2 fingerprint mismatch")

    p3.to(device).eval()
    source.eval()
    for model in (p3, source):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    provenance = {
        "p3": {
            "path": str(Path(args.p3_checkpoint).resolve()),
            "sha256": file_sha256(args.p3_checkpoint),
            "step": int(p3_checkpoint.get("step", 0)),
            "strict_load": True,
        },
        "a2_feature_source": a2_provenance,
        "feature_lock": {
            "path": str(Path(args.feature_lock_checkpoint).resolve()),
            "sha256": file_sha256(args.feature_lock_checkpoint),
            "protocol": feature_lock.get("protocol"),
            "variant": feature_lock.get("variant"),
            "step": int(feature_lock.get("step", 0)),
            "q0_head_instantiated": False,
            "q0_probabilities_used": False,
        },
    }
    return p3, source, provenance


def components_numpy(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        count, labels = cv2.connectedComponents(mask, connectivity=4)
    else:
        labels = _connected_components_4(mask)
        count = int(labels.max()) + 1
    output = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for old_id in range(1, int(count)):
        selected = labels == old_id
        if int(selected.sum()) < minimum_area:
            continue
        output[selected] = next_id
        next_id += 1
    return output


def build_p3_roi_map(
    localization_logits: torch.Tensor,
    threshold: float,
    minimum_area: int,
) -> torch.Tensor:
    if localization_logits.ndim != 4 or localization_logits.shape[1] != 2:
        raise ValueError("P3 localization logits must have shape [B,2,H,W]")
    probability = localization_logits.detach().float().softmax(1)[:, 1]
    binary = (probability >= float(threshold)).cpu().numpy()
    maps = [components_numpy(value, minimum_area) for value in binary]
    return torch.from_numpy(np.stack(maps)).to(
        localization_logits.device, dtype=torch.long
    )


def p3_sar_view(post_sar: torch.Tensor) -> torch.Tensor:
    if post_sar.ndim != 4 or post_sar.shape[1] != 2:
        raise RuntimeError("BIOS requires native and display SAR channels")
    display = post_sar[:, 1:2].float()
    mean = display.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = display.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    return ((display.repeat(1, 3, 1, 1) - mean) / std).to(post_sar.dtype)


def gt_instance_map(damage: torch.Tensor, minimum_area: int) -> torch.Tensor:
    maps = []
    for value in damage.detach().cpu().numpy():
        building = ((value >= 1) & (value <= 3)).astype(np.uint8)
        maps.append(components_numpy(building, minimum_area))
    return torch.from_numpy(np.stack(maps)).to(damage.device, dtype=torch.long)


def collect_rois(
    instance_map: torch.Tensor,
    damage: torch.Tensor,
    p3_logits: torch.Tensor,
    names: Sequence[str],
    *,
    grid_size: int,
    minimum_building_fraction: float,
    minimum_class_purity: float,
) -> tuple[list[BuildingROI], torch.Tensor, torch.Tensor]:
    records: list[BuildingROI] = []
    boxes: list[list[float]] = []
    masks: list[torch.Tensor] = []
    for batch_index in range(instance_map.shape[0]):
        for instance_id in range(1, int(instance_map[batch_index].max()) + 1):
            mask = instance_map[batch_index] == instance_id
            area = int(mask.sum())
            if area <= 0:
                continue
            values = damage[batch_index, mask]
            valid = values != 255
            values = values[valid]
            if values.numel() == 0:
                continue
            building = (values >= 1) & (values <= 3)
            building_fraction = float(building.float().mean())
            if building_fraction < minimum_building_fraction or int(building.sum()) == 0:
                continue
            class_counts = torch.stack([(values == index).sum() for index in (1, 2, 3)])
            label = int(class_counts.argmax()) + 1
            purity = float(class_counts.max() / class_counts.sum().clamp_min(1))
            if purity < minimum_class_purity:
                continue
            ys, xs = torch.where(mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            prediction = int(p3_logits[batch_index, :, mask].mean(1).argmax())
            pooled_mask = F.adaptive_avg_pool2d(
                mask[y0:y1, x0:x1].float()[None, None], (grid_size, grid_size)
            )[0, 0]
            records.append(BuildingROI(
                batch_index=batch_index,
                instance_id=instance_id,
                box=(x0, y0, x1, y1),
                label=label,
                p3_prediction=prediction,
                event=event_name(names[batch_index]),
                area=area,
                purity=purity,
                building_fraction=building_fraction,
            ))
            boxes.append([float(batch_index), float(x0), float(y0), float(x1), float(y1)])
            masks.append(pooled_mask.reshape(-1))
    if not records:
        return records, instance_map.new_empty((0, 5), dtype=torch.float32), instance_map.new_empty(
            (0, grid_size * grid_size), dtype=torch.float32
        )
    return records, torch.tensor(boxes, device=instance_map.device), torch.stack(masks)


def roi_tokens(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    input_size: tuple[int, int],
    grid_size: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return feature.new_empty((0, grid_size * grid_size, feature.shape[1]))
    input_height, input_width = input_size
    scaled = boxes.clone().to(feature)
    scaled[:, (1, 3)] *= feature.shape[-1] / float(input_width)
    scaled[:, (2, 4)] *= feature.shape[-2] / float(input_height)
    pooled = roi_align(
        feature.float(), scaled, output_size=(grid_size, grid_size),
        spatial_scale=1.0, sampling_ratio=2, aligned=True,
    )
    return pooled.permute(0, 2, 3, 1).reshape(pooled.shape[0], grid_size * grid_size, -1)


def add_stage_signals(
    row: dict[str, object],
    stage: str,
    rgb: torch.Tensor,
    sar: torch.Tensor,
    valid: torch.Tensor,
    args,
) -> None:
    if stage == "s3":
        direct = direct_token_anomaly(rgb, sar, valid)
        row[f"{stage}_direct_mean"] = aggregate_token_scores(
            direct, args.relation_threshold, statistics=("mean",)
        )["mean"]

    configurations = [("local", "sim")]
    if stage == "s3":
        configurations.extend([
            ("global", "sim"), ("local_global", "sim"), ("local", "sd")
        ])
    for spatial, relation in configurations:
        values = relation_token_anomaly(
            rgb, sar, valid, grid_size=args.grid_size,
            spatial=spatial, relation=relation,
            neighbour_count=args.neighbour_count,
        )
        requested = AGGREGATES if (stage, spatial, relation) == ("s3", "local", "sim") else ("mean",)
        statistics = aggregate_token_scores(
            values, args.relation_threshold, valid=valid, grid_size=args.grid_size,
            statistics=requested,
        )
        for aggregate, value in statistics.items():
            row[f"{stage}_{spatial}_{relation}_{aggregate}"] = value


def add_multiscale(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    suffixes = sorted(
        key[3:] for key in rows[0]
        if key.startswith("s3_") and f"s4_{key[3:]}" in rows[0]
    )
    normalization: dict[str, dict[str, float]] = {}
    for suffix in suffixes:
        s3 = np.asarray([float(row[f"s3_{suffix}"]) for row in rows], dtype=np.float64)
        s4 = np.asarray([float(row[f"s4_{suffix}"]) for row in rows], dtype=np.float64)
        finite = np.isfinite(s3) & np.isfinite(s4)
        mean3 = float(np.mean(s3[finite])) if finite.any() else 0.0
        mean4 = float(np.mean(s4[finite])) if finite.any() else 0.0
        std3 = max(float(np.std(s3[finite])) if finite.any() else 1.0, 1e-8)
        std4 = max(float(np.std(s4[finite])) if finite.any() else 1.0, 1e-8)
        normalization[suffix] = {"s3_mean": mean3, "s3_std": std3, "s4_mean": mean4, "s4_std": std4}
        for row in rows:
            value3, value4 = float(row[f"s3_{suffix}"]), float(row[f"s4_{suffix}"])
            row[f"ms_{suffix}"] = 0.5 * ((value3 - mean3) / std3 + (value4 - mean4) / std4)
    return normalization


def score_one_configuration(
    stage: str,
    suffix: str,
    rgb_by_stage: Mapping[str, torch.Tensor],
    sar_by_stage: Mapping[str, torch.Tensor],
    valid: torch.Tensor,
    args,
    normalization: Mapping[str, Mapping[str, float]],
) -> float:
    def raw(one_stage: str) -> float:
        rgb, sar = rgb_by_stage[one_stage], sar_by_stage[one_stage]
        if suffix.startswith("direct_"):
            values = direct_token_anomaly(rgb, sar, valid)
            aggregate = suffix.split("_")[-1]
        else:
            spatial, relation, aggregate = suffix.rsplit("_", 2)
            values = relation_token_anomaly(
                rgb, sar, valid, grid_size=args.grid_size,
                spatial=spatial, relation=relation,
                neighbour_count=args.neighbour_count,
            )
        return aggregate_token_scores(values, args.relation_threshold)[aggregate]

    if stage != "ms":
        return raw(stage)
    value3, value4 = raw("s3"), raw("s4")
    norm = normalization[suffix]
    return 0.5 * (
        (value3 - norm["s3_mean"]) / norm["s3_std"]
        + (value4 - norm["s4_mean"]) / norm["s4_std"]
    )


def preregistered_selection() -> dict[str, str]:
    """Return the immutable E0 primary hypothesis; labels never select it."""
    return {
        "primary_signal": PRIMARY_SIGNAL,
        "stage": "s3",
        "spatial": "local",
        "relation": "sim",
        "aggregate": "mean",
        "direct": PRIMARY_DIRECT,
    }


@torch.inference_mode()
def base_pass(p3, source, loader, device, args):
    support_rows: dict[str, list[dict[str, object]]] = {key: [] for key in args.supports}
    processed = 0
    for rgb, sar, _, damage, names in loader:
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        localization, p3_damage = p3(rgb, p3_sar_view(sar))
        rgb_stages = source.rgb_encoder(rgb)
        sar_stages = source.sar_encoder(sar[:, :1])
        maps = {
            "gt": gt_instance_map(damage, args.minimum_area),
            "p3": build_p3_roi_map(localization, args.proposal_threshold, args.minimum_area),
        }
        for support in args.supports:
            records, boxes, token_masks = collect_rois(
                maps[support], damage, p3_damage, names,
                grid_size=args.grid_size,
                minimum_building_fraction=(1.0 if support == "gt" else args.minimum_building_fraction),
                minimum_class_purity=args.minimum_class_purity,
            )
            if not records:
                continue
            stage_tokens = {}
            for stage, index in (("s3", 2), ("s4", 3)):
                stage_tokens[(stage, "rgb")] = roi_tokens(
                    rgb_stages[index], boxes, damage.shape[-2:], args.grid_size
                )
                stage_tokens[(stage, "sar")] = roi_tokens(
                    sar_stages[index], boxes, damage.shape[-2:], args.grid_size
                )
            for index, record in enumerate(records):
                valid = token_masks[index] >= args.minimum_token_mass
                if int(valid.sum()) < args.minimum_valid_tokens:
                    continue
                row: dict[str, object] = {
                    "sample": str(names[record.batch_index]),
                    "event": record.event,
                    "instance_id": record.instance_id,
                    "label": record.label,
                    "p3_prediction": record.p3_prediction,
                    "area": record.area,
                    "purity": record.purity,
                    "building_fraction": record.building_fraction,
                    "valid_tokens": int(valid.sum()),
                }
                for stage in ("s3", "s4"):
                    add_stage_signals(
                        row, stage,
                        stage_tokens[(stage, "rgb")][index],
                        stage_tokens[(stage, "sar")][index],
                        valid, args,
                    )
                support_rows[support].append(row)
        processed += len(names)
        if processed % 50 == 0:
            print(f"BIOS E0 base audit: {processed}", flush=True)
        if args.max_images and processed >= args.max_images:
            break
    normalizations = {key: add_multiscale(rows) for key, rows in support_rows.items()}
    return support_rows, normalizations, processed


def _stress_scores_for_region(
    rgb_by_stage: Mapping[str, torch.Tensor],
    sar_by_stage: Mapping[str, torch.Tensor],
    valid: torch.Tensor,
    selection: Mapping[str, str],
    normalization: Mapping[str, Mapping[str, float]],
    args,
    shuffle_seed: int,
) -> dict[str, float]:
    stage = selection["stage"]
    relation_suffix = f"{selection['spatial']}_{selection['relation']}_mean"
    direct_suffix = "direct_mean"
    result = {
        "relation_shift_0": score_one_configuration(
            stage, relation_suffix, rgb_by_stage, sar_by_stage, valid, args, normalization
        ),
        "direct_shift_0": score_one_configuration(
            stage, direct_suffix, rgb_by_stage, sar_by_stage, valid, args, normalization
        ),
    }
    for magnitude in args.shift_magnitudes:
        relation_values, direct_values = [], []
        for dy, dx in ((magnitude, 0), (-magnitude, 0), (0, magnitude), (0, -magnitude)):
            shifted, shifted_valid = {}, {}
            for one_stage in ("s3", "s4"):
                shifted[one_stage], shifted_valid[one_stage] = shift_token_grid(
                    sar_by_stage[one_stage], valid, args.grid_size, dy, dx
                )
            common_valid = valid & shifted_valid["s3"] & shifted_valid["s4"]
            if int(common_valid.sum()) < args.minimum_valid_tokens:
                continue
            relation_values.append(score_one_configuration(
                stage, relation_suffix, rgb_by_stage, shifted, common_valid, args, normalization
            ))
            direct_values.append(score_one_configuration(
                stage, direct_suffix, rgb_by_stage, shifted, common_valid, args, normalization
            ))
        result[f"relation_shift_{magnitude}"] = float(np.mean(relation_values)) if relation_values else float("nan")
        result[f"direct_shift_{magnitude}"] = float(np.mean(direct_values)) if direct_values else float("nan")
    shuffled = {
        one_stage: deterministic_valid_shuffle(sar_by_stage[one_stage], valid, shuffle_seed)
        for one_stage in ("s3", "s4")
    }
    result["relation_shuffle"] = score_one_configuration(
        stage, relation_suffix, rgb_by_stage, shuffled, valid, args, normalization
    )
    result["direct_shuffle"] = score_one_configuration(
        stage, direct_suffix, rgb_by_stage, shuffled, valid, args, normalization
    )
    return result


@torch.inference_mode()
def stress_pass(p3, source, loader, device, args, selections, normalizations):
    rows: dict[str, list[dict[str, object]]] = {key: [] for key in args.supports}
    processed = 0
    for rgb, sar, _, damage, names in loader:
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        localization, p3_damage = p3(rgb, p3_sar_view(sar))
        rgb_stages = source.rgb_encoder(rgb)
        sar_stages = source.sar_encoder(sar[:, :1])
        maps = {
            "gt": gt_instance_map(damage, args.minimum_area),
            "p3": build_p3_roi_map(localization, args.proposal_threshold, args.minimum_area),
        }
        for support in args.supports:
            records, boxes, token_masks = collect_rois(
                maps[support], damage, p3_damage, names,
                grid_size=args.grid_size,
                minimum_building_fraction=(1.0 if support == "gt" else args.minimum_building_fraction),
                minimum_class_purity=args.minimum_class_purity,
            )
            if not records:
                continue
            token_sets = {}
            for stage, index in (("s3", 2), ("s4", 3)):
                token_sets[(stage, "rgb")] = roi_tokens(rgb_stages[index], boxes, damage.shape[-2:], args.grid_size)
                token_sets[(stage, "sar")] = roi_tokens(sar_stages[index], boxes, damage.shape[-2:], args.grid_size)
            for index, record in enumerate(records):
                valid = token_masks[index] >= args.minimum_token_mass
                if int(valid.sum()) < args.minimum_valid_tokens:
                    continue
                rgb_by_stage = {stage: token_sets[(stage, "rgb")][index] for stage in ("s3", "s4")}
                sar_by_stage = {stage: token_sets[(stage, "sar")][index] for stage in ("s3", "s4")}
                identity = f"{names[record.batch_index]}:{record.instance_id}:{support}:{args.seed}"
                shuffle_seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
                row: dict[str, object] = {
                    "sample": str(names[record.batch_index]), "event": record.event,
                    "label": record.label, "p3_prediction": record.p3_prediction,
                }
                row.update(_stress_scores_for_region(
                    rgb_by_stage, sar_by_stage, valid, selections[support],
                    normalizations[support], args, shuffle_seed,
                ))
                rows[support].append(row)
        processed += len(names)
        if processed % 50 == 0:
            print(f"BIOS E0 stress audit: {processed}", flush=True)
        if args.max_images and processed >= args.max_images:
            break
    return rows, processed


def summarize_support(base_rows, stress_rows, normalization, args):
    signal_names = sorted(
        key for key in base_rows[0]
        if key.startswith(("s3_", "s4_", "ms_"))
    ) if base_rows else []
    metrics = {key: complete_signal_metrics(base_rows, key) for key in signal_names}
    selection = preregistered_selection()
    stress_names = sorted(
        key for key in stress_rows[0] if key.startswith(("relation_", "direct_"))
    ) if stress_rows else []
    stress_metrics = {key: complete_signal_metrics(stress_rows, key) for key in stress_names}
    relation = metrics[selection["primary_signal"]]["p3_nonaffected_miss"]
    direct = metrics[selection["direct"]]["p3_nonaffected_miss"]

    def auc(block):
        value = block.get("auroc")
        return None if value is None else float(value)

    relation_auc, direct_auc = auc(relation), auc(direct)
    shuffle_relation_auc = auc(
        stress_metrics["relation_shuffle"]["p3_nonaffected_miss"]
    )
    shift_checks = {}
    for magnitude in args.shift_magnitudes:
        shifted_relation = auc(stress_metrics[f"relation_shift_{magnitude}"]["p3_nonaffected_miss"])
        shifted_direct = auc(stress_metrics[f"direct_shift_{magnitude}"]["p3_nonaffected_miss"])
        relation_drop = None if relation_auc is None or shifted_relation is None else relation_auc - shifted_relation
        direct_drop = None if direct_auc is None or shifted_direct is None else direct_auc - shifted_direct
        shift_checks[str(magnitude)] = {
            "relation_auroc": shifted_relation,
            "direct_auroc": shifted_direct,
            "relation_drop": relation_drop,
            "direct_drop": direct_drop,
            "relation_drop_smaller": bool(
                relation_drop is not None and direct_drop is not None and relation_drop < direct_drop
            ),
        }
    shift1 = shift_checks.get("1", {})
    shift2 = shift_checks.get("2", {})
    checks = {
        "miss_auroc_at_least_0p60": relation_auc is not None and relation_auc >= 0.60,
        "event_macro_at_least_0p58": (
            relation.get("event_macro_auroc") is not None
            and float(relation["event_macro_auroc"]) >= 0.58
        ),
        "at_least_10_nonnegative_events": int(relation.get("nonnegative_events", 0)) >= 10,
        "relation_beats_direct_unshifted": (
            relation_auc is not None and direct_auc is not None and relation_auc > direct_auc
        ),
        "relation_beats_direct_shift_1": bool(
            shift1.get("relation_auroc") is not None and shift1.get("direct_auroc") is not None
            and shift1["relation_auroc"] > shift1["direct_auroc"]
        ),
        "relation_beats_direct_shift_2": bool(
            shift2.get("relation_auroc") is not None and shift2.get("direct_auroc") is not None
            and shift2["relation_auroc"] > shift2["direct_auroc"]
        ),
        "shift_1_more_robust_than_direct": shift1.get("relation_drop_smaller", False),
        "shift_2_more_robust_than_direct": shift2.get("relation_drop_smaller", False),
        "unshuffled_beats_shuffle": bool(
            relation_auc is not None and shuffle_relation_auc is not None
            and relation_auc > shuffle_relation_auc
        ),
    }
    magnitude_medians = metrics[selection["primary_signal"]].get("class_medians", {})
    heterogeneity_medians = metrics.get("s3_local_sim_std", {}).get("class_medians", {})

    def finite_triplet(values):
        triplet = [values.get(key) for key in ("intact", "damaged", "destroyed")]
        return triplet if all(value is not None and np.isfinite(float(value)) for value in triplet) else None

    magnitude_triplet = finite_triplet(magnitude_medians)
    heterogeneity_triplet = finite_triplet(heterogeneity_medians)
    magnitude_monotonic = bool(
        magnitude_triplet is not None
        and magnitude_triplet[0] < magnitude_triplet[1] < magnitude_triplet[2]
    )
    damaged_heterogeneity_peak = bool(
        heterogeneity_triplet is not None
        and heterogeneity_triplet[1] > heterogeneity_triplet[0]
        and heterogeneity_triplet[1] > heterogeneity_triplet[2]
    )
    if magnitude_monotonic:
        e1_hypothesis = "ORDINAL_MAGNITUDE_CANDIDATE"
    elif damaged_heterogeneity_peak:
        e1_hypothesis = "MAGNITUDE_HETEROGENEITY_DUAL_CANDIDATE"
    elif checks["miss_auroc_at_least_0p60"]:
        e1_hypothesis = "COMPLEMENTARY_MISSED_DAMAGE_ONLY"
    else:
        e1_hypothesis = "NO_SUPPORTED_E1_FORMULATION"
    return {
        "regions": len(base_rows),
        "selection_protocol": "preregistered; no validation-driven signal selection",
        "selection": selection,
        "primary_relation_metrics": metrics[selection["primary_signal"]],
        "matched_direct_metrics": metrics[selection["direct"]],
        "secondary_diagnostics": {
            "heterogeneity": metrics.get("s3_local_sim_std"),
            "upper_tail": metrics.get("s3_local_sim_q90"),
            "stage4": metrics.get("s4_local_sim_mean"),
            "multiscale": metrics.get("ms_local_sim_mean"),
            "global": metrics.get("s3_global_sim_mean"),
            "local_global": metrics.get("s3_local_global_sim_mean"),
            "similarity_dissimilarity": metrics.get("s3_local_sd_mean"),
        },
        "structural_pattern": {
            "magnitude_medians_i_d_x": magnitude_triplet,
            "heterogeneity_medians_i_d_x": heterogeneity_triplet,
            "magnitude_is_strictly_ordinal": magnitude_monotonic,
            "damaged_has_peak_heterogeneity": damaged_heterogeneity_peak,
            "conditional_e1_hypothesis": e1_hypothesis,
            "note": "diagnostic only; this does not authorize E1 unless every gate passes",
        },
        "all_signal_metrics": metrics,
        "stress_metrics": stress_metrics,
        "shift_robustness": shift_checks,
        "shuffle_control": {
            "unshuffled_relation_auroc": relation_auc,
            "shuffled_relation_auroc": shuffle_relation_auc,
            "unshuffled_minus_shuffled": (
                None if relation_auc is None or shuffle_relation_auc is None
                else relation_auc - shuffle_relation_auc
            ),
            "supports_spatial_organization": bool(
                relation_auc is not None and shuffle_relation_auc is not None
                and relation_auc > shuffle_relation_auc
            ),
        },
        "normalization": normalization,
        "gate": {"passed": bool(all(checks.values())), "checks": checks},
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser("BIOS E0 parameter-free relation evidence audit")
    value.add_argument("--p3_checkpoint", required=True)
    value.add_argument("--a2_checkpoint", required=True)
    value.add_argument("--feature_lock_checkpoint", required=True)
    value.add_argument("--dataset_path", required=True)
    value.add_argument("--val_list", required=True)
    value.add_argument("--sar_stats_path", required=True)
    value.add_argument("--output_json", required=True)
    value.add_argument("--preregistration", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--crop_size", type=int, default=640)
    value.add_argument("--batch_size", type=int, default=1)
    value.add_argument("--num_workers", type=int, default=8)
    value.add_argument("--supports", nargs="+", choices=("gt", "p3"), default=("gt", "p3"))
    value.add_argument("--grid_size", type=int, default=7)
    value.add_argument("--minimum_area", type=int, default=16)
    value.add_argument("--proposal_threshold", type=float, default=0.5)
    value.add_argument("--minimum_building_fraction", type=float, default=0.5)
    value.add_argument("--minimum_class_purity", type=float, default=0.75)
    value.add_argument("--minimum_token_mass", type=float, default=0.25)
    value.add_argument("--minimum_valid_tokens", type=int, default=4)
    value.add_argument("--neighbour_count", type=int, default=4)
    value.add_argument("--relation_threshold", type=float, default=0.25)
    value.add_argument("--shift_magnitudes", nargs="+", type=int, default=(1, 2, 3))
    value.add_argument("--max_images", type=int, default=0)
    value.add_argument("--seed", type=int, default=3407)
    value.add_argument("--maximum_gt_p3_auc_gap", type=float, default=0.10)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.grid_size < 3 or args.minimum_valid_tokens < 2:
        raise ValueError("invalid BIOS grid/token contract")
    if any(value <= 0 or value >= args.grid_size for value in args.shift_magnitudes):
        raise ValueError("shift magnitudes must lie in [1, grid_size-1]")
    registration, registration_sha256 = load_preregistration(args.preregistration, args)
    device = torch.device(args.device)
    p3, source, provenance = load_feature_sources(args, device)
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path, args.val_list, args.crop_size,
        split="val", variant="h4", sar_stats_path=args.sar_stats_path,
    )
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    print("=" * 96)
    print("Protocol       : BIOS E0 validation-only parameter-free relation evidence audit")
    print(f"Supports       : {','.join(args.supports)}")
    print(f"Feature lock   : {Path(args.feature_lock_checkpoint).resolve()}")
    print("Q0 probability : NOT USED")
    print(f"Grid/stages    : {args.grid_size}x{args.grid_size} / Stage3+Stage4")
    print("Training       : NONE")
    print("Test accessed  : NEVER")
    print(f"Output         : {Path(args.output_json).resolve()}")
    print("=" * 96, flush=True)
    base_rows, normalizations, base_images = base_pass(
        p3, source, DataLoader(dataset, **loader_args), device, args
    )
    selections = {support: preregistered_selection() for support in args.supports}
    stress_rows, stress_images = stress_pass(
        p3, source, DataLoader(dataset, **loader_args), device, args,
        selections, normalizations,
    )
    support_reports = {}
    for support in args.supports:
        report = summarize_support(
            base_rows[support], stress_rows[support], normalizations[support], args
        )
        support_reports[support] = report
    full_run = args.max_images == 0 or base_images >= len(dataset)
    p3_gate = support_reports.get("p3", {}).get("gate", {}).get("passed", False)
    gt_gate = support_reports.get("gt", {}).get("gate", {}).get("passed", False)
    gt_auc = support_reports.get("gt", {}).get("primary_relation_metrics", {}).get(
        "p3_nonaffected_miss", {}
    ).get("auroc")
    p3_auc = support_reports.get("p3", {}).get("primary_relation_metrics", {}).get(
        "p3_nonaffected_miss", {}
    ).get("auroc")
    gt_p3_gap = None if gt_auc is None or p3_auc is None else abs(float(gt_auc) - float(p3_auc))
    cross_support_gate = bool(
        gt_p3_gap is not None and gt_p3_gap <= args.maximum_gt_p3_auc_gap
    )
    if not full_run:
        decision = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
    elif p3_gate and gt_gate and cross_support_gate:
        decision = "PROCEED_TO_E1_CONTROLLED_READOUT"
    elif gt_gate:
        decision = "STOP_PRACTICAL_BIOS_OR_REDESIGN_PROPOSALS"
    else:
        decision = "STOP_BIOS_RELATION_ROUTE"
    payload = {
        "protocol": "BIOS_E0_relation_evidence_audit_v1",
        "provenance": provenance,
        "val_list": str(Path(args.val_list).resolve()),
        "processed_images_base": base_images,
        "processed_images_stress": stress_images,
        "supports": support_reports,
        "configuration": {
            "grid_size": args.grid_size,
            "minimum_area": args.minimum_area,
            "proposal_threshold": args.proposal_threshold,
            "minimum_building_fraction": args.minimum_building_fraction,
            "minimum_class_purity": args.minimum_class_purity,
            "minimum_token_mass": args.minimum_token_mass,
            "minimum_valid_tokens": args.minimum_valid_tokens,
            "neighbour_count": args.neighbour_count,
            "relation_threshold": args.relation_threshold,
            "shift_magnitudes": args.shift_magnitudes,
            "maximum_gt_p3_auc_gap": args.maximum_gt_p3_auc_gap,
        },
        "preregistration": {
            "path": str(Path(args.preregistration).resolve()),
            "sha256": registration_sha256,
            "declaration": registration,
            "primary_hypothesis": registration["primary_hypothesis"],
            "primary_signal": PRIMARY_SIGNAL,
            "primary_metric": "p3_nonaffected_miss.auroc",
            "secondary_signals_do_not_select_e1": True,
        },
        "cross_support_gate": {
            "gt_auc": gt_auc,
            "p3_auc": p3_auc,
            "absolute_gap": gt_p3_gap,
            "maximum_gap": args.maximum_gt_p3_auc_gap,
            "passed": cross_support_gate,
        },
        "training": "none",
        "test_accessed": False,
        "decision": decision,
    }
    atomic_json(payload, args.output_json)
    print("=" * 112)
    print("BIOS E0 relation evidence summary")
    for support, report in support_reports.items():
        selected = report["selection"]
        relation = report["primary_relation_metrics"]["p3_nonaffected_miss"]
        direct = report["matched_direct_metrics"]["p3_nonaffected_miss"]
        def fmt(value):
            return "NA" if value is None else f"{float(value):.4f}"
        print(
            f"{support:>7} | regions={report['regions']} | signal={selected['primary_signal']} "
            f"| missAUC={fmt(relation['auroc'])} | event={fmt(relation['event_macro_auroc'])} "
            f"| events={relation['nonnegative_events']}/{relation['comparable_events']} "
            f"| direct={fmt(direct['auroc'])} | gate={report['gate']['passed']}"
        )
        shuffle = report["shuffle_control"]
        print(
            f"        shuffle AUC={fmt(shuffle['shuffled_relation_auroc'])} "
            f"| normal-shuffle={fmt(shuffle['unshuffled_minus_shuffled'])} "
            f"| spatial={shuffle['supports_spatial_organization']}"
        )
        for magnitude, values in report["shift_robustness"].items():
            print(
                f"        shift={magnitude} | relation={fmt(values['relation_auroc'])} "
                f"drop={fmt(values['relation_drop'])} | direct={fmt(values['direct_auroc'])} "
                f"drop={fmt(values['direct_drop'])} "
                f"| robust={values['relation_drop_smaller']}"
            )
        events = relation.get("event_auroc", {})
        if events:
            rendered = " | ".join(
                f"{event}={fmt(value)}" for event, value in sorted(events.items())
            )
            print(f"        event AUROC: {rendered}")
        pattern = report["structural_pattern"]
        print(
            f"        structural pattern | ordinal={pattern['magnitude_is_strictly_ordinal']} "
            f"| damaged-heterogeneity-peak={pattern['damaged_has_peak_heterogeneity']} "
            f"| hypothesis={pattern['conditional_e1_hypothesis']}"
        )
    print(
        f"GT/P3 AUC gap : {fmt(gt_p3_gap)} / max={args.maximum_gt_p3_auc_gap:.4f} "
        f"| passed={cross_support_gate}"
    )
    print(f"Decision       : {decision}")
    print(f"Saved          : {Path(args.output_json).resolve()}")
    print("Test accessed  : False")


if __name__ == "__main__":
    main()
