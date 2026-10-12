# -*- coding: utf-8 -*-
"""Validation-only shallow evidence audit before H12-DGB training.

The audit freezes H6-A2 and optionally frozen SARMAE / P3 experts.  It does
not train, write weights, or accept a test split.  Ground truth is used only
to score whether Stage1/2/3 RGB--SAR differences or SARMAE blocks 3/6/8/11
can rank:

* Intact vs Damaged
* A2 correct vs A2 error
* Damaged gate-miss (GT Damaged predicted Intact)
* P3-wrong/A2-right and A2-wrong/P3-right, when a P3 checkpoint is supplied
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H10_H3D import (  # noqa: E402
    factorize_building_logits,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H11_SATR import (  # noqa: E402
    SARMAEEncoder,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    build_h6_sdcr_damageformer,
)
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (  # noqa: E402
    DamageFormer_S8_MICB_DSMIL,
    connected_components_4,
)
from bda_benchmark.script.standard_ML.audit_CCU_local_correspondence import (  # noqa: E402
    BinaryScoreBuffer,
    binary_auroc,
    event_name,
    pure_stage_labels,
)
from bda_benchmark.script.standard_ML.h12_dgb_runtime import (  # noqa: E402
    binary_auprc,
    parameter_free_difference_scores,
    parameter_free_unimodal_scores,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)
from bda_benchmark.script.standard_ML.train_H11_SATR import (  # noqa: E402
    migrate_sarmae_encoder_state,
)


A2_STAGES = ("stage1", "stage2", "stage3")
SARMAE_BLOCKS = (3, 6, 8, 11)
A2_SCORE_NAMES = ("abs_diff_mean", "cosine_distance", "local_std")
SARMAE_SCORE_NAMES = ("energy", "local_std", "rgb_cosine_distance")
TASK_NAMES = (
    "intact_vs_damaged",
    "a2_error",
    "damaged_gate_miss",
    "p3_wrong_a2_right",
    "a2_wrong_p3_right",
)
VALID_A2_PROTOCOLS = ("H6_SDCR_best_weights_v1", "H6_SDCR_resume_v1")
EXPECTED_P3_SHA256 = (
    "45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7"
)
SARMAE_NEAREST_RGB = {
    3: "stage2",
    6: "stage2",
    8: "stage3",
    11: "stage4",
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    value = checkpoint.get("model", checkpoint)
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("checkpoint does not contain a model state dictionary")
    return value  # type: ignore[return-value]


def interpolate_to(value: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    return F.interpolate(value, size=tuple(size), mode="bilinear", align_corners=False)


class ProbeBuffers:
    def __init__(self) -> None:
        self.global_buffers: Dict[str, BinaryScoreBuffer] = defaultdict(BinaryScoreBuffer)
        self.event_buffers: Dict[str, Dict[str, BinaryScoreBuffer]] = defaultdict(
            lambda: defaultdict(BinaryScoreBuffer)
        )

    def update(
        self,
        probe: str,
        score: torch.Tensor,
        label: torch.Tensor,
        events: Sequence[str],
    ) -> None:
        valid = torch.isfinite(score) & (label >= 0)
        if not valid.any():
            return
        self.global_buffers[probe].update(score[valid], label[valid])
        for sample_index, event in enumerate(events):
            sample_valid = valid[sample_index]
            if not sample_valid.any():
                continue
            self.event_buffers[event][probe].update(
                score[sample_index][sample_valid],
                label[sample_index][sample_valid],
            )


def summarize_buffer(buffer: BinaryScoreBuffer) -> Dict[str, float | int | None]:
    raw = buffer.summary()
    if not buffer.scores:
        raw["auprc"] = None
        return raw
    score = np.concatenate(buffer.scores)
    label = np.concatenate(buffer.labels).astype(np.uint8, copy=False)
    raw["auprc"] = binary_auprc(score, label)
    return raw


def summarize_probe(
    global_buffer: BinaryScoreBuffer,
    event_buffers: Mapping[str, BinaryScoreBuffer],
) -> Dict[str, object]:
    events = {}
    aurocs = []
    for event, buffer in sorted(event_buffers.items()):
        record = summarize_buffer(buffer)
        events[event] = record
        if record["auroc"] is not None:
            aurocs.append(float(record["auroc"]))
    global_record = summarize_buffer(global_buffer)
    event_worst = min(aurocs) if aurocs else None
    event_macro = sum(aurocs) / len(aurocs) if aurocs else None
    consistent = sum(1 for value in aurocs if value > 0.5)
    return {
        **global_record,
        "event_macro_auroc": event_macro,
        "event_worst_auroc": event_worst,
        "event_direction_consistent": consistent,
        "comparable_events": len(aurocs),
        "events": events,
    }


def task_labels(
    damage: torch.Tensor,
    a2_prediction: torch.Tensor,
    a2_occurrence: torch.Tensor,
    p3_prediction: torch.Tensor | None,
    output_size: Sequence[int],
    min_class_purity: float,
) -> Dict[str, torch.Tensor]:
    labels = pure_stage_labels(damage, output_size, min_class_purity)
    a2_pred = F.interpolate(
        a2_prediction.float().unsqueeze(1), size=tuple(output_size), mode="nearest"
    ).long().squeeze(1)
    occurrence = interpolate_to(a2_occurrence.float(), output_size)[:, 0] >= 0
    building = (labels >= 1) & (labels <= 3)
    intact_vs_damaged = torch.full_like(labels, -1)
    intact_vs_damaged = torch.where(labels == 1, torch.zeros_like(labels), intact_vs_damaged)
    intact_vs_damaged = torch.where(labels == 2, torch.ones_like(labels), intact_vs_damaged)
    a2_error = torch.full_like(labels, -1)
    a2_error = torch.where(building, (a2_pred != labels).long(), a2_error)
    gate_miss = torch.full_like(labels, -1)
    damaged = labels == 2
    gate_miss = torch.where(damaged, (~occurrence).long(), gate_miss)
    result = {
        "intact_vs_damaged": intact_vs_damaged,
        "a2_error": a2_error,
        "damaged_gate_miss": gate_miss,
    }
    if p3_prediction is None:
        none = torch.full_like(labels, -1)
        result["p3_wrong_a2_right"] = none
        result["a2_wrong_p3_right"] = none
        return result
    p3_pred = F.interpolate(
        p3_prediction.float().unsqueeze(1), size=tuple(output_size), mode="nearest"
    ).long().squeeze(1)
    p3_wrong_a2_right = torch.full_like(labels, -1)
    a2_wrong_p3_right = torch.full_like(labels, -1)
    comparable = building & (p3_pred != a2_pred)
    p3_wrong_a2_right = torch.where(
        comparable, ((p3_pred != labels) & (a2_pred == labels)).long(), p3_wrong_a2_right
    )
    a2_wrong_p3_right = torch.where(
        comparable, ((a2_pred != labels) & (p3_pred == labels)).long(), a2_wrong_p3_right
    )
    result["p3_wrong_a2_right"] = p3_wrong_a2_right
    result["a2_wrong_p3_right"] = a2_wrong_p3_right
    return result


def p3_pseudo_rgb(display_sar: torch.Tensor) -> torch.Tensor:
    mean = display_sar.new_tensor([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = display_sar.new_tensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    repeated = display_sar.float().repeat(1, 3, 1, 1)
    return (repeated - mean) / std


def load_a2(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    protocol = str(checkpoint.get("protocol", ""))
    variant = str(checkpoint.get("variant", "")).lower()
    if protocol not in VALID_A2_PROTOCOLS or variant != "a2":
        raise RuntimeError(
            "H12 audit requires a locked H6-A2 checkpoint; "
            f"received protocol={protocol!r}, variant={variant!r}"
        )
    model = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    incompatible = model.load_state_dict(_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"A2 strict load failed: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "protocol": protocol,
        "variant": variant,
        "step": int(checkpoint.get("step", 0)),
    }


def load_optional_p3(path: str, device: torch.device, expected_sha256: str):
    digest = sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(
            f"P3 checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"P3 strict load failed: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(Path(path).resolve()),
        "sha256": digest,
        "protocol": checkpoint.get("protocol"),
    }


def official_sarmae_input(display_sar: torch.Tensor) -> torch.Tensor:
    mean = display_sar.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = display_sar.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    unit = display_sar.float().clamp(0.0, 255.0) / 255.0
    return (unit.repeat(1, 3, 1, 1) - mean) / std


def load_optional_sarmae(path: str, device: torch.device):
    encoder = SARMAEEncoder(in_channels=3, tap_blocks=SARMAE_BLOCKS)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    report = migrate_sarmae_encoder_state(encoder, checkpoint)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        **report,
    }


def decide(tasks: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    gate = tasks.get("damaged_gate_miss", {})
    assert isinstance(gate, Mapping)

    def best(prefix: str, names: Sequence[str], score_names: Sequence[str]) -> Dict[str, object]:
        winner = {"probe": None, "event_macro_auroc": None, "auroc": None}
        for name in names:
            for score_name in score_names:
                probe = f"{prefix}_{name}_{score_name}"
                record = gate.get(probe)
                if not isinstance(record, Mapping):
                    continue
                macro = record.get("event_macro_auroc")
                auroc = record.get("auroc")
                if macro is None:
                    continue
                current = winner["event_macro_auroc"]
                if current is None or float(macro) > float(current):
                    winner = {
                        "probe": probe,
                        "event_macro_auroc": float(macro),
                        "auroc": None if auroc is None else float(auroc),
                        "event_worst_auroc": record.get("event_worst_auroc"),
                        "comparable_events": record.get("comparable_events"),
                        "event_direction_consistent": record.get(
                            "event_direction_consistent"
                        ),
                    }
        return winner

    a2_best = best("a2", A2_STAGES, A2_SCORE_NAMES)
    shallow_sarmae = best("sarmae", ("block3", "block6"), SARMAE_SCORE_NAMES)
    deep_sarmae = best("sarmae", ("block8", "block11"), SARMAE_SCORE_NAMES)
    building_best = best("building", A2_STAGES, A2_SCORE_NAMES)
    a2_macro = a2_best["event_macro_auroc"]
    shallow_macro = shallow_sarmae["event_macro_auroc"]
    deep_macro = deep_sarmae["event_macro_auroc"]
    building_macro = building_best["event_macro_auroc"]

    recommendation = "STOP_SARMAE_AND_BYPASS_USE_BUILDING_POOL_AUDIT"
    reason = "no probe ranked Damaged gate-miss above chance with event-macro AUROC"
    if (
        shallow_macro is not None
        and (deep_macro is None or shallow_macro >= float(deep_macro) + 0.03)
        and (a2_macro is None or shallow_macro >= float(a2_macro) + 0.02)
        and float(shallow_macro) >= 0.55
    ):
        recommendation = "KEEP_SHALLOW_SARMAE_ADAPTER"
        reason = "SARMAE Block3/6 outranks Block8/11 and A2 shallow stages on gate-miss"
    elif a2_macro is not None and float(a2_macro) >= 0.55:
        recommendation = "IMPLEMENT_H12_D1_STAGE2_BYPASS"
        reason = "A2 Stage1/2/3 RGB-SAR differences rank Damaged gate-miss above chance"
    elif (
        building_macro is not None
        and a2_macro is not None
        and float(building_macro) >= float(a2_macro) + 0.05
    ):
        recommendation = "STOP_PIXEL_BYPASS_USE_BUILDING_POOLING"
        reason = "building-pooled A2 scores outrank pixel scores; try a pooling head"
    elif a2_macro is None and shallow_macro is None:
        recommendation = "STOP_SARMAE_AND_BYPASS"
        reason = "neither A2 shallow stages nor SARMAE blocks produced usable scores"

    return {
        "recommendation": recommendation,
        "reason": reason,
        "a2_best_gate_miss": a2_best,
        "sarmae_shallow_best_gate_miss": shallow_sarmae,
        "sarmae_deep_best_gate_miss": deep_sarmae,
        "building_best_gate_miss": building_best,
    }


@torch.no_grad()
def run_audit(args) -> Dict[str, object]:
    if args.split != "val":
        raise RuntimeError("H12 shallow audit refuses any split other than val")
    set_seed(args.seed)
    device = torch.device(args.device)
    a2, a2_meta = load_a2(args.checkpoint, device)
    p3 = None
    p3_meta = None
    if args.p3_checkpoint:
        p3, p3_meta = load_optional_p3(
            args.p3_checkpoint, device, args.expected_p3_sha256
        )
    sarmae = None
    sarmae_meta = None
    if args.sarmae_checkpoint:
        sarmae, sarmae_meta = load_optional_sarmae(args.sarmae_checkpoint, device)

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split="val",
        variant="h4",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    pixel_buffers = ProbeBuffers()
    building_buffers = ProbeBuffers()
    processed = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb, sar, _location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        events = [event_name(name) for name in names]
        a2_out = a2(rgb, sar[:, :1], return_dict=True)
        a2_pred = a2_out["output_dam"].argmax(1)
        occurrence, _severity = factorize_building_logits(a2_out["output_dam"])
        p3_pred = None
        if p3 is not None:
            loc_logits, dam_logits = p3(rgb, p3_pseudo_rgb(sar[:, 1:2]))
            del loc_logits
            p3_pred = dam_logits.argmax(1)

        for stage_name in A2_STAGES:
            rgb_common = a2_out[f"{stage_name}_rgb_common"]
            sar_common = a2_out[f"{stage_name}_sar_common"]
            scores = parameter_free_difference_scores(rgb_common, sar_common)
            labels = task_labels(
                damage,
                a2_pred,
                occurrence,
                p3_pred,
                rgb_common.shape[-2:],
                args.min_class_purity,
            )
            for score_name, score_map in scores.items():
                if score_name not in A2_SCORE_NAMES:
                    continue
                for task_name, label_map in labels.items():
                    pixel_buffers.update(
                        f"{task_name}/a2_{stage_name}_{score_name}",
                        score_map[:, 0],
                        label_map,
                        events,
                    )

            for sample_index, event in enumerate(events):
                building = (damage[sample_index] >= 1) & (damage[sample_index] <= 3)
                components = connected_components_4(building)
                if not components:
                    continue
                sample_labels = {
                    task_name: label_map[sample_index]
                    for task_name, label_map in labels.items()
                }
                for score_name, score_map in scores.items():
                    if score_name not in A2_SCORE_NAMES:
                        continue
                    pooled_score = []
                    pooled_label = {task_name: [] for task_name in TASK_NAMES}
                    score_full = interpolate_to(
                        score_map[sample_index : sample_index + 1],
                        damage.shape[-2:],
                    )[0, 0]
                    for component in components:
                        mask = torch.as_tensor(
                            component, device=score_full.device, dtype=torch.bool
                        )
                        if not mask.any():
                            continue
                        pooled_score.append(score_full[mask].mean())
                        for task_name in TASK_NAMES:
                            label_full = F.interpolate(
                                sample_labels[task_name][None, None].float(),
                                size=damage.shape[-2:],
                                mode="nearest",
                            )[0, 0]
                            values = label_full[mask]
                            values = values[values >= 0]
                            if values.numel() == 0:
                                pooled_label[task_name].append(
                                    values.new_tensor(-1.0)
                                )
                            else:
                                pooled_label[task_name].append(values.float().mean())
                    if not pooled_score:
                        continue
                    score_tensor = torch.stack(pooled_score).reshape(1, -1)
                    for task_name in TASK_NAMES:
                        label_tensor = torch.stack(pooled_label[task_name]).reshape(1, -1)
                        binary = torch.where(
                            label_tensor < 0,
                            label_tensor,
                            (label_tensor >= 0.5).float(),
                        )
                        building_buffers.update(
                            f"{task_name}/building_{stage_name}_{score_name}",
                            score_tensor,
                            binary,
                            [event],
                        )

        if sarmae is not None:
            mae_input = official_sarmae_input(sar[:, 1:2])
            taps = sarmae(mae_input)
            for block_id, feature in zip(SARMAE_BLOCKS, taps):
                scores = parameter_free_unimodal_scores(feature)
                rgb_stage = SARMAE_NEAREST_RGB[block_id]
                rgb_common = interpolate_to(
                    a2_out[f"{rgb_stage}_rgb_common"], feature.shape[-2:]
                )
                rgb_unit = F.normalize(rgb_common.float(), dim=1, eps=1e-6)
                sar_unit = F.normalize(feature.float(), dim=1, eps=1e-6)
                # Channel counts differ; compare after a parameter-free
                # per-location energy match via pooled cosine on spatially
                # aligned L2-normalised maps reduced by adaptive pooling.
                rgb_pool = F.adaptive_avg_pool2d(rgb_unit, feature.shape[-2:])
                rgb_pool = F.normalize(rgb_pool.mean(1, keepdim=True), dim=1, eps=1e-6)
                sar_pool = F.normalize(sar_unit.mean(1, keepdim=True), dim=1, eps=1e-6)
                scores["rgb_cosine_distance"] = 1.0 - (rgb_pool * sar_pool).sum(
                    1, keepdim=True
                ).clamp(-1.0, 1.0)
                labels = task_labels(
                    damage,
                    a2_pred,
                    occurrence,
                    p3_pred,
                    feature.shape[-2:],
                    args.min_class_purity,
                )
                for score_name, score_map in scores.items():
                    if score_name not in SARMAE_SCORE_NAMES:
                        continue
                    for task_name, label_map in labels.items():
                        pixel_buffers.update(
                            f"{task_name}/sarmae_block{block_id}_{score_name}",
                            score_map[:, 0],
                            label_map,
                            events,
                        )
        processed += int(rgb.shape[0])

    tasks: Dict[str, Dict[str, object]] = {}
    for task_name in TASK_NAMES:
        probes: Dict[str, object] = {}
        for source in (pixel_buffers, building_buffers):
            for probe, buffer in source.global_buffers.items():
                stored_task, stored_probe = probe.split("/", 1)
                if stored_task != task_name:
                    continue
                probes[stored_probe] = summarize_probe(
                    buffer,
                    {
                        event: event_probes[probe]
                        for event, event_probes in source.event_buffers.items()
                        if probe in event_probes
                    },
                )
        tasks[task_name] = probes

    payload = {
        "protocol": "H12_DGB_shallow_evidence_audit_v1",
        "split": "val",
        "test_accessed": False,
        "processed_images": processed,
        "a2": a2_meta,
        "p3": p3_meta,
        "sarmae": sarmae_meta,
        "min_class_purity": args.min_class_purity,
        "tasks": tasks,
        "decision": decide(tasks),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("Audit H12 shallow damage-gate evidence")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--p3_checkpoint", default="")
    parser.add_argument("--sarmae_checkpoint", default="")
    parser.add_argument("--expected_p3_sha256", default=EXPECTED_P3_SHA256)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--min_class_purity", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if "test" in Path(args.data_list_path).name.lower():
        raise RuntimeError("H12 shallow audit refuses a test list")

    payload = run_audit(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    decision = payload["decision"]
    print(json.dumps(decision, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    print(f"Recommendation: {decision['recommendation']}", flush=True)


if __name__ == "__main__":
    main()
