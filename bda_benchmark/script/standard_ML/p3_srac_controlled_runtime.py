# -*- coding: utf-8 -*-
"""Loss and audit utilities for the controlled P3 structure classifier."""
from __future__ import annotations

from typing import Dict
import math

import torch
import torch.nn.functional as F

from bda_benchmark.model.DamageFormer_RGBSAR_H13_SPER import (
    pool_superpixel_statistics,
    unpool_superpixel_values,
)
from bda_benchmark.model.DamageFormer_P3_SRAC_Controlled import (
    AFFINITY_OFFSETS,
)


def _shift_labels(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift BHW labels without circular wrap-around."""
    if value.ndim != 3:
        raise ValueError("affinity labels must be BHW")
    height, width = value.shape[-2:]
    shifted = torch.full_like(value, 255)
    source_y = slice(max(0, dy), min(height, height + dy))
    target_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, dx), min(width, width + dx))
    target_x = slice(max(0, -dx), min(width, width - dx))
    shifted[:, target_y, target_x] = value[:, source_y, source_x]
    return shifted


def damage_affinity_supervision(
    outputs: Dict[str, object],
    labels: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    """Balanced eight-neighbour supervision inside labelled buildings."""
    logits = outputs.get("srac_affinity_logits")
    geometry_valid = outputs.get("srac_affinity_valid")
    if not isinstance(logits, torch.Tensor) or not isinstance(
        geometry_valid, torch.Tensor
    ):
        raise TypeError("SR1PAA affinity tensors are missing")
    if logits.shape[1] != len(AFFINITY_OFFSETS):
        raise ValueError("SR1PAA affinity channel count does not match offsets")
    resized = F.interpolate(
        labels[:, None].float(), size=logits.shape[-2:], mode="nearest"
    )[:, 0].long()
    centre_valid = resized != int(ignore_index)
    centre_building = centre_valid & (resized >= 1) & (resized <= 3)
    targets, masks, damaged_boundaries = [], [], []
    for index, (dy, dx) in enumerate(AFFINITY_OFFSETS):
        neighbour = _shift_labels(resized, dy, dx)
        neighbour_valid = neighbour != int(ignore_index)
        neighbour_building = (
            neighbour_valid & (neighbour >= 1) & (neighbour <= 3)
        )
        pair_valid = (
            geometry_valid[:, index].bool()
            & centre_building
            & neighbour_building
        )
        same = resized == neighbour
        damaged_boundary = pair_valid & ~same & (
            (resized == 2) | (neighbour == 2)
        )
        targets.append(same.float())
        masks.append(pair_valid)
        damaged_boundaries.append(damaged_boundary)
    target = torch.stack(targets, dim=1)
    valid = torch.stack(masks, dim=1)
    damaged_boundary = torch.stack(damaged_boundaries, dim=1)
    positive = valid & (target > 0.5)
    negative = valid & ~positive

    zero = logits.float().sum() * 0.0
    positive_loss = (
        F.binary_cross_entropy_with_logits(
            logits.float()[positive], target[positive], reduction="mean"
        ) if positive.any() else zero
    )
    negative_loss = (
        F.binary_cross_entropy_with_logits(
            logits.float()[negative], target[negative], reduction="mean"
        ) if negative.any() else zero
    )
    if positive.any() and negative.any():
        loss = 0.5 * (positive_loss + negative_loss)
    elif positive.any():
        loss = positive_loss
    elif negative.any():
        loss = negative_loss
    else:
        loss = zero

    probability = logits.float().sigmoid()
    same_mean = probability[positive].mean() if positive.any() else zero.detach()
    different_mean = (
        probability[negative].mean() if negative.any() else zero.detach()
    )
    same_accuracy = (
        (probability[positive] >= 0.5).float().mean()
        if positive.any() else zero.detach()
    )
    different_recall = (
        (probability[negative] < 0.5).float().mean()
        if negative.any() else zero.detach()
    )
    if positive.any() and negative.any():
        balanced_accuracy = 0.5 * (same_accuracy + different_recall)
    else:
        balanced_accuracy = zero.detach()
    damage_boundary_recall = (
        (probability[damaged_boundary] < 0.5).float().mean()
        if damaged_boundary.any() else zero.detach()
    )
    valid_count = valid.float().sum()
    same_count = positive.float().sum()
    different_count = negative.float().sum()
    damage_boundary_count = damaged_boundary.float().sum()
    same_correct = (
        (probability[positive] >= 0.5).float().sum()
        if positive.any() else zero.detach()
    )
    different_correct = (
        (probability[negative] < 0.5).float().sum()
        if negative.any() else zero.detach()
    )
    damage_boundary_correct = (
        (probability[damaged_boundary] < 0.5).float().sum()
        if damaged_boundary.any() else zero.detach()
    )
    return {
        "representation_loss": loss,
        "srac_affinity_loss": loss.detach(),
        "srac_affinity_positive_loss": positive_loss.detach(),
        "srac_affinity_negative_loss": negative_loss.detach(),
        "srac_affinity_same_mean": same_mean.detach(),
        "srac_affinity_different_mean": different_mean.detach(),
        "srac_affinity_margin": (same_mean - different_mean).detach(),
        "srac_affinity_same_accuracy": same_accuracy.detach(),
        "srac_affinity_different_recall": different_recall.detach(),
        "srac_affinity_balanced_accuracy": balanced_accuracy.detach(),
        "srac_affinity_damage_boundary_recall": damage_boundary_recall.detach(),
        "srac_affinity_valid_pairs": valid_count.detach(),
        "srac_affinity_same_count": same_count.detach(),
        "srac_affinity_same_correct": same_correct.detach(),
        "srac_affinity_different_count": different_count.detach(),
        "srac_affinity_different_correct": different_correct.detach(),
        "srac_affinity_damage_boundary_count": damage_boundary_count.detach(),
        "srac_affinity_damage_boundary_correct": (
            damage_boundary_correct.detach()
        ),
        "srac_affinity_negative_fraction": (
            negative.float().sum() / valid_count.clamp_min(1.0)
        ).detach(),
    }


def node_class_weights(target: torch.Tensor, classes: int = 3) -> torch.Tensor:
    """Inverse-square-root node weights, normalised to unit mean."""
    count = torch.bincount(target.long(), minlength=classes).float()
    weight = count.clamp_min(1.0).rsqrt()
    return weight / weight.mean().clamp_min(1.0e-6)


def structure_region_supervision(
    outputs: Dict[str, object],
    labels: torch.Tensor,
    ignore_index: int,
    model=None,
    minimum_building_fraction: float = 0.50,
    minimum_purity: float = 0.70,
    reconstruction_weight: float = 0.25,
    building_weight: float = 1.0,
    boundary_weight: float = 0.5,
    entropy_weight: float = 0.02,
    entropy_target: float = 0.60,
) -> Dict[str, torch.Tensor]:
    """Train the region classifier and its differentiable partition.

    Targets are constructed exclusively from training labels.  They are never
    consumed by ``forward`` and therefore cannot affect validation/test region
    generation.  The semantic reconstruction term is the SSN-style signal
    that encourages learned regions to respect damage boundaries.
    """
    reference = outputs["output_dam"]
    assert isinstance(reference, torch.Tensor)
    if "srac_affinity_logits" in outputs:
        return damage_affinity_supervision(outputs, labels, ignore_index)
    if "srac_assignment" not in outputs:
        zero = reference.float().sum() * 0.0
        return {
            "representation_loss": zero,
            "srac_region_ce": zero.detach(),
            "srac_reconstruction_loss": zero.detach(),
            "srac_supervised_nodes": zero.detach(),
        }

    assignment = outputs["srac_assignment"]
    feature_size = assignment.entropy.shape[-2:]
    resized = F.interpolate(
        labels[:, None].float(), size=feature_size, mode="nearest"
    )[:, 0].long()
    valid_pixel = resized != int(ignore_index)
    building = valid_pixel & (resized >= 1) & (resized <= 3)
    class_map = (resized - 1).clamp(0, 2)
    one_hot = F.one_hot(class_map, num_classes=3).permute(0, 3, 1, 2).float()
    one_hot = one_hot * building[:, None].float()
    class_fraction, _std, _maximum, _mass = pool_superpixel_statistics(
        one_hot, assignment
    )
    building_fraction = class_fraction.sum(-1)
    conditional = class_fraction / building_fraction.unsqueeze(-1).clamp_min(1.0e-6)
    purity, target = conditional.max(-1)
    keep = (
        (building_fraction >= float(minimum_building_fraction))
        & (purity >= float(minimum_purity))
    )

    candidate = building_fraction >= 0.05
    if candidate.any():
        candidate_count = candidate.float().sum()
        candidate_building_sum = building_fraction[candidate].sum()
        background_leakage_sum = (1.0 - building_fraction[candidate]).sum()
        all_node_purity = purity[candidate].mean()
        mixed_fraction = (purity[candidate] < float(minimum_purity)).float().mean()
        kept_fraction = keep.float().sum() / candidate.float().sum().clamp_min(1.0)
        candidate_building_fraction = building_fraction[candidate].mean()
        background_leakage = (1.0 - building_fraction[candidate]).mean()
    else:
        candidate_count = purity.new_zeros(())
        candidate_building_sum = purity.new_zeros(())
        background_leakage_sum = purity.new_zeros(())
        all_node_purity = purity.new_zeros(())
        mixed_fraction = purity.new_zeros(())
        kept_fraction = purity.new_zeros(())
        candidate_building_fraction = purity.new_zeros(())
        background_leakage = purity.new_zeros(())
    partition_entropy = outputs.get("srac_partition_entropy", purity.new_zeros(()))
    if not isinstance(partition_entropy, torch.Tensor):
        partition_entropy = purity.new_tensor(float(partition_entropy))
    effective_candidates = torch.exp(
        partition_entropy.float().clamp(0.0, 1.0) * math.log(9.0)
    )
    node_mass = outputs.get("srac_node_mass")
    if isinstance(node_mass, torch.Tensor):
        flat_mass = node_mass.float().reshape(-1)
        mean_node_mass = flat_mass.mean().detach()
        node_mass_std = flat_mass.std(unbiased=False).detach()
        node_mass_cv = (
            node_mass_std / mean_node_mass.clamp_min(1.0e-6)
        ).detach()
        node_mass_p10 = torch.quantile(flat_mass, 0.10).detach()
        node_mass_p90 = torch.quantile(flat_mass, 0.90).detach()
        empty_node_fraction = (flat_mass < 0.1).float().mean().detach()
    else:
        mean_node_mass = purity.new_zeros(())
        node_mass_std = purity.new_zeros(())
        node_mass_cv = purity.new_zeros(())
        node_mass_p10 = purity.new_zeros(())
        node_mass_p90 = purity.new_zeros(())
        empty_node_fraction = purity.new_zeros(())

    common_diagnostics = {
        "srac_assignment_iterations": labels.new_tensor(
            float(
                getattr(
                    getattr(model, "partition", None),
                    "assignment_iterations",
                    1,
                )
            )
        ).float(),
        "srac_supervised_nodes": keep.float().sum().detach(),
        "srac_region_purity": purity[keep].mean().detach()
        if keep.any()
        else purity.new_zeros(()),
        "srac_all_node_purity": all_node_purity.detach(),
        "srac_mixed_node_fraction": mixed_fraction.detach(),
        "srac_kept_node_fraction": kept_fraction.detach(),
        "srac_candidate_building_fraction": candidate_building_fraction.detach(),
        "srac_background_leakage_fraction": background_leakage.detach(),
        # Sums/counts let full-validation aggregation ignore images/crops with
        # no building candidates instead of treating their leakage as zero.
        "srac_candidate_node_count": candidate_count.detach(),
        "srac_candidate_building_sum": candidate_building_sum.detach(),
        "srac_background_leakage_sum": background_leakage_sum.detach(),
        "srac_has_candidate_nodes": (candidate_count > 0).float().detach(),
        "srac_partition_entropy": partition_entropy.detach(),
        "srac_effective_candidates": effective_candidates.detach(),
        "srac_mean_node_mass": mean_node_mass,
        "srac_node_mass_std": node_mass_std,
        "srac_node_mass_cv": node_mass_cv,
        "srac_node_mass_p10": node_mass_p10,
        "srac_node_mass_p90": node_mass_p90,
        "srac_empty_node_fraction": empty_node_fraction,
    }
    for key in (
        "srac_assignment_top1_probability",
        "srac_assignment_top12_margin",
        "srac_q0_q1_l1",
        "srac_q1_q2_l1",
        "srac_seed_move_0_1",
        "srac_seed_move_1_2",
    ):
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            common_diagnostics[key] = value.detach()

    boundary_flag = outputs.get("srac_boundary_partition")
    boundary_aware = (
        isinstance(boundary_flag, torch.Tensor)
        and float(boundary_flag.detach().item()) > 0.5
    )
    if boundary_aware:
        # The partition is a structural component: supervise only the stable
        # Background/Building geometry, not post-disaster I/D/X severity.
        building_target = building[:, None].float()
        node_building, _bs, _bm, _mass = pool_superpixel_statistics(
            building_target, assignment
        )
        reconstructed_building = unpool_superpixel_values(
            node_building, assignment, feature_size
        ).clamp(1.0e-5, 1.0 - 1.0e-5)
        valid_mask = valid_pixel[:, None]
        if valid_mask.any():
            # BCE(probability, target) is intentionally rejected by PyTorch
            # autocast because its backward can underflow in fp16/bf16.  The
            # logit form is mathematically equivalent after clamping and is
            # explicitly autocast-safe.
            building_logits = torch.logit(
                reconstructed_building.float(), eps=1.0e-5
            )
            building_bce = F.binary_cross_entropy_with_logits(
                building_logits[valid_mask],
                building_target.float()[valid_mask],
            )
            prediction_valid = reconstructed_building * valid_mask.float()
            target_valid = building_target * valid_mask.float()
            intersection = (prediction_valid * target_valid).sum()
            building_dice = 1.0 - (
                2.0 * intersection + 1.0
            ) / (prediction_valid.sum() + target_valid.sum() + 1.0)
        else:
            building_bce = reconstructed_building.sum() * 0.0
            building_dice = reconstructed_building.sum() * 0.0
        building_loss = building_bce + building_dice

        def morphological_edge(value: torch.Tensor) -> torch.Tensor:
            dilated = F.max_pool2d(value, 3, stride=1, padding=1)
            eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
            return (dilated - eroded).clamp(0.0, 1.0)

        predicted_edge = morphological_edge(reconstructed_building)
        target_edge = morphological_edge(building_target)
        if valid_mask.any():
            edge_logits = torch.logit(
                predicted_edge.float().clamp(1.0e-5, 1.0 - 1.0e-5)
            )
            edge_bce = F.binary_cross_entropy_with_logits(
                edge_logits[valid_mask],
                target_edge.float()[valid_mask],
            )
            pred_edge_valid = predicted_edge * valid_mask.float()
            target_edge_valid = target_edge * valid_mask.float()
            edge_intersection = (pred_edge_valid * target_edge_valid).sum()
            edge_dice = 1.0 - (
                2.0 * edge_intersection + 1.0
            ) / (pred_edge_valid.sum() + target_edge_valid.sum() + 1.0)
        else:
            edge_bce = predicted_edge.sum() * 0.0
            edge_dice = predicted_edge.sum() * 0.0
        boundary_loss = edge_bce + edge_dice

        entropy_map = assignment.entropy.float()
        valid_entropy = entropy_map[valid_mask]
        entropy_mean = (
            valid_entropy.mean() if valid_entropy.numel() else entropy_map.mean()
        )
        entropy_loss = F.relu(entropy_mean - float(entropy_target))
        boundary_band = target_edge >= 0.5
        building_interior = building_target.bool() & ~boundary_band
        background_interior = ~building_target.bool() & ~boundary_band & valid_mask
        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            selected = value[mask]
            return selected.mean() if selected.numel() else value.new_zeros(())
        common_diagnostics.update(
            {
                "srac_entropy_building_interior": masked_mean(
                    entropy_map, building_interior
                ).detach(),
                "srac_entropy_boundary": masked_mean(
                    entropy_map, boundary_band & valid_mask
                ).detach(),
                "srac_entropy_background": masked_mean(
                    entropy_map, background_interior
                ).detach(),
            }
        )
        total_partition = (
            float(building_weight) * building_loss
            + float(boundary_weight) * boundary_loss
            + float(entropy_weight) * entropy_loss
        )

        pred_building = reconstructed_building >= 0.5
        gt_building = building_target >= 0.5
        valid_bool = valid_mask.bool()
        intersection = (pred_building & gt_building & valid_bool).sum().float()
        union = ((pred_building | gt_building) & valid_bool).sum().float()
        building_iou = intersection / union.clamp_min(1.0)
        pred_edge_binary = predicted_edge >= 0.5
        target_edge_binary = target_edge >= 0.5
        tp = (pred_edge_binary & target_edge_binary & valid_bool).sum().float()
        fp = (pred_edge_binary & ~target_edge_binary & valid_bool).sum().float()
        fn = (~pred_edge_binary & target_edge_binary & valid_bool).sum().float()
        boundary_f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)
        temperature = outputs.get("srac_partition_temperature", purity.new_zeros(()))
        return {
            "representation_loss": total_partition,
            "srac_region_ce": purity.new_zeros(()),
            "srac_reconstruction_loss": building_loss.detach(),
            "srac_building_loss": building_loss.detach(),
            "srac_boundary_loss": boundary_loss.detach(),
            "srac_entropy_loss": entropy_loss.detach(),
            "srac_building_iou": building_iou.detach(),
            "srac_boundary_f1": boundary_f1.detach(),
            "srac_partition_temperature": (
                temperature.detach()
                if isinstance(temperature, torch.Tensor)
                else purity.new_tensor(float(temperature))
            ),
            **common_diagnostics,
        }

    # SR1P is trained only by the ordinary four-class pixel objective.  This
    # keeps E1 causal: the sole change is feature-space pool/unpool before
    # P3's original classifier.  Boundary-aware supervision belongs to the
    # later E2 variant and must not be silently mixed into this screen.
    if "srac_feature_projection" in outputs:
        zero = reference.float().sum() * 0.0
        return {
            "representation_loss": zero,
            "srac_region_ce": zero.detach(),
            "srac_reconstruction_loss": zero.detach(),
            **common_diagnostics,
        }

    node_logits = outputs.get("srac_node_logits")
    node_embedding = outputs.get("srac_node_embedding")
    if not isinstance(node_logits, torch.Tensor) or not isinstance(
        node_embedding, torch.Tensor
    ):
        raise TypeError("P3-SRAC node tensors are missing")
    if keep.any():
        weights = node_class_weights(target[keep])
        region_ce = F.cross_entropy(
            node_logits.float()[keep], target[keep], weight=weights
        )
    else:
        region_ce = node_logits.float().sum() * 0.0

    reconstructed = unpool_superpixel_values(
        conditional, assignment, feature_size
    )
    reconstruction_error = (reconstructed - one_hot).square().sum(1)
    reconstruction_mask = building & valid_pixel
    if reconstruction_mask.any():
        reconstruction = reconstruction_error[reconstruction_mask].mean()
    else:
        reconstruction = reconstructed.sum() * 0.0
    total = region_ce + float(reconstruction_weight) * reconstruction
    # Memory mutation is deliberately outside forward and only happens while
    # training.  Validation/evaluation never invokes this loss callback.
    if (
        model is not None
        and bool(getattr(model.config, "retrieval_memory", False))
        and model.training
        and keep.any()
    ):
        model.region_classifier.memory.enqueue(
            node_embedding.reshape(-1, node_embedding.shape[-1]),
            target.reshape(-1),
            keep.reshape(-1),
        )
    return {
        "representation_loss": total,
        "srac_region_ce": region_ce.detach(),
        "srac_reconstruction_loss": reconstruction.detach(),
        **common_diagnostics,
    }


__all__ = [
    "node_class_weights",
    "damage_affinity_supervision",
    "structure_region_supervision",
]
