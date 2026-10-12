#!/usr/bin/env python3
"""Causal diagnostics for the V3 aligned-change partition branch.

This audit does not train, change checkpoints, or access Test.  It answers:
1. whether the 3x3 matcher is discriminative;
2. whether the learned gate follows alignment confidence;
3. whether the residual really enters association logits;
4. whether the learned scalar gain is too small;
5. which loss terms push the scalar gain up or down.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.association import (
    masked_local_softmax,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    load_a1_parent,
    load_v2_checkpoint,
    parse_batch,
    seed_worker,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import (
    read_list,
)

from .losses import (
    damage_semantic_reconstruction_loss,
    occurrence_boundary_affinity_loss,
    stage_losses,
)
from .model import DamageFormerRGBAnchoredChangeSuperpixelV3


def _event(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


class ScalarAccumulator:
    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []

    def add(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        value = value.detach().float().reshape(-1).cpu()
        if mask is not None:
            mask = mask.detach().bool().reshape(-1).cpu()
            value = value[mask]
        if value.numel():
            self.values.append(value)

    def tensor(self) -> torch.Tensor:
        if not self.values:
            return torch.empty(0)
        return torch.cat(self.values)

    def summary(self) -> Dict[str, float | int | None]:
        value = self.tensor()
        if not value.numel():
            return {"count": 0, "mean": None, "std": None, "p05": None,
                    "p25": None, "p50": None, "p75": None, "p95": None}
        q = torch.quantile(value, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
        return {
            "count": int(value.numel()),
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]),
        }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 2 or right.numel() != left.numel():
        return None
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 1.0e-12:
        return None
    return float((left * right).sum() / denominator)


def _masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.float(), value.shape[-2:], mode="nearest") > 0.5
    return value.detach().float()[mask.expand_as(value)].cpu()


def _matching_stats(model, rgb: torch.Tensor, sar: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Recompute the exact LCA scores/weights without changing its forward."""
    lca = model.biscp.lca
    if hasattr(lca, "base"):
        lca = lca.base
    batch, _, height, width = rgb.shape
    neighbours = int(lca.window_size) ** 2
    locations = height * width
    query = lca._matching_key(lca.query(rgb)).flatten(2)
    key = lca._matching_key(lca.key(sar))
    patches = F.unfold(
        key, int(lca.window_size), padding=int(lca.window_size) // 2
    ).view(batch, key.shape[1], neighbours, locations)
    scores = torch.einsum("bcl,bckl->bkl", query, patches) / float(lca.temperature)
    valid = F.unfold(
        torch.ones((batch, 1, height, width), device=rgb.device, dtype=scores.dtype),
        int(lca.window_size), padding=int(lca.window_size) // 2,
    ).view(batch, neighbours, locations)
    masked_scores = scores.masked_fill(valid < 0.5, -1.0e4)
    weights = torch.softmax(masked_scores, dim=1)
    top2_score = masked_scores.topk(k=2, dim=1).values
    top2_weight = weights.topk(k=2, dim=1).values
    score_margin = top2_score[:, 0] - top2_score[:, 1]
    weight_margin = top2_weight[:, 0] - top2_weight[:, 1]
    entropy = -(weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()).sum(1)
    entropy = entropy / valid.sum(1).clamp_min(2.0).log()
    return {
        "score_margin": score_margin.view(batch, 1, height, width),
        "weight_margin": weight_margin.view(batch, 1, height, width),
        "top1_weight": top2_weight[:, 0].view(batch, 1, height, width),
        "normalized_entropy": entropy.view(batch, 1, height, width),
        "winning_index": weights.argmax(1).view(batch, 1, height, width),
        "center_is_winner": (
            weights.argmax(1) == neighbours // 2
        ).float().view(batch, 1, height, width),
        "weights": weights.view(batch, neighbours, height, width),
    }


def _translate(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Zero-padded integer translation; positive dy/dx move content down/right."""
    output = torch.zeros_like(value)
    height, width = value.shape[-2:]
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    if src_y1 > src_y0 and src_x1 > src_x0:
        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = value[..., src_y0:src_y1, src_x0:src_x1]
    return output


def _shift_equivariance(
    model,
    rgb: torch.Tensor,
    sar: torch.Tensor,
    natural_winner: torch.Tensor,
    building_mask: torch.Tensor,
    dy: int,
    dx: int,
) -> tuple[int, int]:
    """Check whether a known SAR shift produces the corresponding winner shift."""
    shifted = _matching_stats(model, rgb, _translate(sar, dy, dx))["winning_index"]
    lca = model.biscp.lca.base if hasattr(model.biscp.lca, "base") else model.biscp.lca
    window = int(lca.window_size)
    radius = window // 2
    natural_y = natural_winner // window - radius
    natural_x = natural_winner % window - radius
    expected_y = natural_y + int(dy)
    expected_x = natural_x + int(dx)
    valid = (
        building_mask.bool()
        & (expected_y >= -radius) & (expected_y <= radius)
        & (expected_x >= -radius) & (expected_x <= radius)
    )
    # Exclude image borders touched by zero padding.
    border = radius + max(abs(int(dy)), abs(int(dx)))
    if border > 0:
        valid[..., :border, :] = False
        valid[..., -border:, :] = False
        valid[..., :, :border] = False
        valid[..., :, -border:] = False
    expected = (expected_y + radius) * window + (expected_x + radius)
    correct = (shifted == expected) & valid
    return int(correct.sum()), int(valid.sum())


def _counterfactual(
    aux: Mapping[str, torch.Tensor], scale: float
) -> Dict[str, torch.Tensor | float]:
    structural = aux["structural_logits"].float()
    applied = aux["change_alpha"].float() * aux["change_delta_logits"].float()
    logits = structural + float(scale) * applied
    probabilities = masked_local_softmax(
        logits, aux["candidate_valid"]
    )
    base = masked_local_softmax(structural, aux["candidate_valid"])
    eps = 1.0e-8
    kl = (probabilities * (
        probabilities.clamp_min(eps).log() - base.clamp_min(eps).log()
    )).sum(1)
    return {
        "probabilities": probabilities,
        "mean_l1_from_rgb": float((probabilities - base).abs().sum(1).mean()),
        "mean_kl_from_rgb": float(kl.mean()),
        "hard_assignment_change_fraction": float(
            (probabilities.argmax(1) != base.argmax(1)).float().mean()
        ),
    }


def _quantile_gate_table(confidence: torch.Tensor, gate: torch.Tensor) -> list[dict]:
    confidence = confidence.float().cpu()
    gate = gate.float().cpu()
    if confidence.numel() == 0:
        return []
    edges = torch.quantile(confidence, torch.linspace(0, 1, 6))
    rows = []
    for index in range(5):
        lo, hi = edges[index], edges[index + 1]
        selected = (confidence >= lo) & (
            confidence <= hi if index == 4 else confidence < hi
        )
        rows.append({
            "bin": index + 1,
            "confidence_low": float(lo),
            "confidence_high": float(hi),
            "count": int(selected.sum()),
            "gate_mean": float(gate[selected].mean()) if bool(selected.any()) else None,
        })
    return rows


def _gradient(value: torch.Tensor, parameter: torch.Tensor) -> float:
    grad = torch.autograd.grad(
        value, parameter, retain_graph=True, allow_unused=True
    )[0]
    return 0.0 if grad is None else float(grad.detach())


def main() -> None:
    parser = argparse.ArgumentParser("V3 aligned-change branch diagnostics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--a1_ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--variant", default="p5_anchored_change")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--gradient_batches", type=int, default=8)
    parser.add_argument("--shift_batches", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", args.variant))
    model = DamageFormerRGBAnchoredChangeSuperpixelV3(variant).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    checkpoint_report = load_v2_checkpoint(model, args.checkpoint)
    model.configure_stage("change")
    model.eval()

    dataset = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), args.crop_size, type="val"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0,
    )

    accumulators = defaultdict(ScalarAccumulator)
    event_accumulators: dict[str, dict[str, ScalarAccumulator]] = defaultdict(
        lambda: defaultdict(ScalarAccumulator)
    )
    counterfactual = {
        scale: defaultdict(list) for scale in (0.0, 1.0, 5.0, 10.0)
    }
    gradient_rows: list[dict] = []
    shift_counts = {(0, 1): [0, 0], (1, 0): [0, 0]}
    identity_error = 0.0
    images = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        pre, post, _, damage, names = parse_batch(batch, device)
        gradient_enabled = batch_index < int(args.gradient_batches)
        with torch.set_grad_enabled(gradient_enabled):
            _, _, aux = model(pre, post, return_aux=True)
            mask = (aux["building_ids"] > 0).unsqueeze(1)
            matching = _matching_stats(model, aux["rgb_feature"], aux["sar_feature"])
            tensors = {
                "confidence": aux["region_confidence"],
                "alignment_entropy": aux["region_alignment_entropy"],
                "gate": aux["change_gate_probability"],
                "a1_uncertainty": aux["a1_uncertainty"],
                "local_variation": aux["local_relation_variation"],
                "score_margin": matching["score_margin"],
                "weight_margin": matching["weight_margin"],
                "top1_weight": matching["top1_weight"],
                "matching_entropy_recomputed": matching["normalized_entropy"],
                "center_is_winner": matching["center_is_winner"],
            }
            for key, value in tensors.items():
                accumulators[key].add(value, mask)
            if batch_index < int(args.shift_batches):
                for shift, counts in shift_counts.items():
                    correct, valid_count = _shift_equivariance(
                        model, aux["rgb_feature"], aux["sar_feature"],
                        matching["winning_index"], mask, *shift,
                    )
                    counts[0] += correct
                    counts[1] += valid_count
            applied = aux["change_alpha"].float() * aux["change_delta_logits"].float()
            accumulators["structural_logit_abs"].add(
                aux["structural_logits"].float().abs().mean(1, keepdim=True), mask
            )
            accumulators["delta_logit_abs"].add(
                aux["change_delta_logits"].float().abs().mean(1, keepdim=True), mask
            )
            accumulators["applied_residual_abs"].add(
                applied.abs().mean(1, keepdim=True), mask
            )
            identity_error = max(
                identity_error,
                float(((aux["association_logits"].float() - aux["structural_logits"].float()) - applied).abs().max()),
            )

            for scale in counterfactual:
                row = _counterfactual(aux, scale)
                for key in ("mean_l1_from_rgb", "mean_kl_from_rgb", "hard_assignment_change_fraction"):
                    counterfactual[scale][key].append(float(row[key]))
                # This is the decisive alpha test: if a larger residual makes
                # the GT-aligned region objectives worse, alpha is small for a
                # good reason rather than merely because of regularization.
                cf_aux = dict(aux)
                cf_aux["association_probabilities"] = row["probabilities"]
                counterfactual[scale]["damage_reconstruction_loss"].append(float(
                    damage_semantic_reconstruction_loss(cf_aux, damage).detach()
                ))
                counterfactual[scale]["occurrence_boundary_loss"].append(float(
                    occurrence_boundary_affinity_loss(cf_aux, damage).detach()
                ))

            if gradient_enabled:
                losses = stage_losses(
                    aux, damage, stage="change", use_bag_supervision=True,
                    ignore_index=255, bag_aggregator="logmeanexp",
                    bag_temperature=0.50, bag_topk=3, bag_label_purity=0.70,
                )
                gain = model.biscp.change_gain
                weights = {
                    "bag": 1.0, "change_reg": 0.01,
                    "damage_reconstruction": 0.05, "occurrence_boundary": 0.02,
                }
                grad_row = {"batch": batch_index, "total": _gradient(losses["total"], gain)}
                for key, weight in weights.items():
                    if key in losses and torch.is_tensor(losses[key]):
                        grad_row[key] = _gradient(float(weight) * losses[key], gain)
                gradient_rows.append(grad_row)

        # Per-event summaries use detached values and the same building mask.
        for bi, name in enumerate(names):
            event = _event(str(name))
            for key in ("confidence", "gate", "score_margin", "weight_margin"):
                event_accumulators[event][key].add(tensors[key][bi:bi + 1], mask[bi:bi + 1])
        images += int(pre.shape[0])
        if images % 32 < int(pre.shape[0]):
            print(f"change diagnostic {images}/{min(len(dataset), args.max_batches * args.batch_size if args.max_batches else len(dataset))}", flush=True)

    confidence = accumulators["confidence"].tensor()
    gate = accumulators["gate"].tensor()
    entropy = accumulators["alignment_entropy"].tensor()
    uncertainty = accumulators["a1_uncertainty"].tensor()
    variation = accumulators["local_variation"].tensor()
    raw_gain = float(model.biscp.change_gain.detach())
    alignment_summary = {
        key: accumulators[key].summary() for key in (
            "confidence", "alignment_entropy", "score_margin",
            "weight_margin", "top1_weight", "matching_entropy_recomputed",
            "center_is_winner",
        )
    }
    alignment_summary["synthetic_shift_equivariance"] = {
        f"dy{dy}_dx{dx}": {
            "correct": counts[0],
            "total": counts[1],
            "accuracy": counts[0] / max(1, counts[1]),
        }
        for (dy, dx), counts in shift_counts.items()
    }
    result = {
        "protocol": "DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3_CHANGE_DIAGNOSTIC",
        "variant": variant,
        "processed_images": images,
        "checkpoint": checkpoint_report,
        "parent": parent_report,
        "alignment": alignment_summary,
        "gate": {
            "distribution": accumulators["gate"].summary(),
            "pearson_with_confidence": _pearson(confidence, gate),
            "pearson_with_entropy": _pearson(entropy, gate),
            "pearson_with_a1_uncertainty": _pearson(uncertainty, gate),
            "pearson_with_local_variation": _pearson(variation, gate),
            "by_confidence_quintile": _quantile_gate_table(confidence, gate),
        },
        "residual_path": {
            "equation_max_abs_error": identity_error,
            "structural_logit_abs": accumulators["structural_logit_abs"].summary(),
            "delta_logit_abs": accumulators["delta_logit_abs"].summary(),
            "applied_residual_abs": accumulators["applied_residual_abs"].summary(),
            "raw_change_gain": raw_gain,
            "tanh_change_gain": math.tanh(raw_gain),
            "change_alpha_max": float(model.biscp.change_alpha_max),
            "counterfactual_alpha_scale": {
                str(scale): {
                    key: float(np.mean(values)) if values else None
                    for key, values in rows.items()
                } for scale, rows in counterfactual.items()
            },
        },
        "gain_gradient": {
            "batches": gradient_rows,
            "mean": {
                key: float(np.mean([row.get(key, 0.0) for row in gradient_rows]))
                for key in ("total", "bag", "change_reg", "damage_reconstruction", "occurrence_boundary")
            } if gradient_rows else {},
            "interpretation": (
                "gradient descent subtracts the reported gradient; a gradient with "
                "the same sign as raw_change_gain pushes |gain| toward zero"
            ),
        },
        "event_diagnostics": {
            event: {key: value.summary() for key, value in rows.items()}
            for event, rows in event_accumulators.items()
        },
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
