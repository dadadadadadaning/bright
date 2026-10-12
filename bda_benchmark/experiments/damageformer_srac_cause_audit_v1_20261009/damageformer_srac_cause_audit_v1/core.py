"""Fixed-evidence routing interventions. No GT enters prediction functions."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import project_region_probabilities


MODES = (
    "a1", "a1_resize", "a1_q", "a1_q_sharp", "a1_q_split", "region",
    "region_read_sharp", "region_pool_sharp", "region_both_sharp", "region_split",
)


def sharpen(q, power=2.0):
    if not math.isfinite(power) or power <= 1:
        raise ValueError("Predeclared sharpening power must be finite and > 1")
    # Subtract the maximum before exponentiation, keeping impossible routes zero.
    valid = q > 0
    score = q.clamp_min(torch.finfo(q.dtype).tiny).log() * power
    score = score.masked_fill(~valid, -torch.inf)
    result = score.softmax(1)
    if not torch.isfinite(result).all():
        raise ValueError("Every position needs at least one valid route")
    return result


def split_ids(ids, building_grid, count, threshold=0.5):
    """Keep Q weights; reindex each route into predicted BG/building slots.

    Batch-one diagnostic, 9 supports remain 9, ceiling K -> 2K. This is not
    instance segmentation, and neither group is forced to output a class.
    """
    if building_grid.shape[0:2] != (1, 1) or not 0 < threshold < 1:
        raise ValueError("Expected one B1hw predicted building map and threshold in (0,1)")
    group = (building_grid.flatten() >= threshold).long()
    if ids.shape[1] != group.numel():
        raise ValueError("Building grid and Q have different resolutions")
    return ids + count * group[None], 2 * count


def probability_read(tokens, q, ids, shape, output_size):
    """Tokens here are probabilities, NOT logits: never softmax them twice."""
    b, _, classes = tokens.shape
    output = tokens.new_zeros(b, q.shape[-1], classes)
    for j in range(q.shape[1]):
        index = ids[j][None, :, None].expand(b, -1, classes)
        output = output + tokens.gather(1, index) * q[:, j, :, None]
    output = output.transpose(1, 2).reshape(b, classes, *shape)
    output = F.interpolate(output, output_size, mode="bilinear", align_corners=False)
    return output / output.sum(1, keepdim=True).clamp_min(1e-8)


def probability_roundtrip(probability_grid, q, ids, count, output_size):
    pooled = soft_pool_unpool(probability_grid.float(), q, ids, count)
    return probability_read(pooled["tokens"], q, ids, probability_grid.shape[-2:], output_size)


def capture(model, pre, post):
    """One native parent forward; temporary hook only exposes existing aux data."""
    captured = {}
    def take_aux(_module, _args, output):
        captured.update(output[2])
    handle = model.base.register_forward_hook(take_aux)
    try:
        result = model(pre, post)
    finally:
        handle.remove()
    for key in ("p_building", "fused_roi"):
        if key not in captured:
            raise ValueError("Parent auxiliary output missing: " + key)
    return result, captured


def routing_stats(q, ids, count):
    ones = q.new_ones(q.shape[0], 1, 1, q.shape[-1])
    mass = soft_pool_unpool(ones, q, ids, count)["mass"]
    hard = ids[None].expand(q.shape[0], -1, -1).gather(1, q.argmax(1, keepdim=True))[:, 0]
    return dict(
        allocated=count,
        soft_occupied=int((mass > 1e-6).sum()),
        hard_occupied=sum(int(torch.unique(row).numel()) for row in hard),
        tiny_nonempty_mass=int(((mass > 1e-6) & (mass < 1)).sum()),
        entropy_mean=float((-(q * q.clamp_min(1e-8).log()).sum(1) / math.log(9)).mean()),
        max_assignment_mean=float(q.max(1).values.mean()),
    )


@torch.no_grad()
def interventions(model, result, auxiliary, output_size, power=2.0, threshold=0.5):
    if result["q"].shape[0] != 1:
        raise ValueError("Audit interventions intentionally use batch size 1")
    q, ids, count, shape = result["q"], result["ids"], result["count"], result["shape"]
    original_q = q.clone()
    q_sharp = sharpen(q, power)
    building = F.interpolate(auxiliary["p_building"].float(), shape, mode="bilinear", align_corners=False)
    ids_split, split_count = split_ids(ids, building, count, threshold)
    a1 = result["old_pixel"].float().softmax(1)
    grid = F.interpolate(a1, shape, mode="area")
    resized = F.interpolate(grid, output_size, mode="bilinear", align_corners=False)
    probabilities = dict(
        a1=a1,
        a1_resize=resized / resized.sum(1, keepdim=True).clamp_min(1e-8),
        a1_q=probability_roundtrip(grid, q, ids, count, output_size),
        a1_q_sharp=probability_roundtrip(grid, q_sharp, ids, count, output_size),
        a1_q_split=probability_roundtrip(grid, q, ids_split, split_count, output_size),
        region=result["prob"],
        region_read_sharp=project_region_probabilities(result["region_logits"], q_sharp, ids, shape, output_size),
    )
    # Keep feature adapter output AND its original boundary/entropy inputs fixed.
    # Rebuilding tokens includes attention pooling, mean/std and Q geometry.
    def logits_for(pool_q, pool_ids, slots):
        tokens, _, _, _ = model.embedding(
            result["adapted_content"], auxiliary["fused_roi"], pool_q, pool_ids, shape, slots,
        )
        return model.region_classifier(tokens)
    sharp_logits = logits_for(q_sharp, ids, count)
    probabilities["region_pool_sharp"] = project_region_probabilities(sharp_logits, q, ids, shape, output_size)
    probabilities["region_both_sharp"] = project_region_probabilities(sharp_logits, q_sharp, ids, shape, output_size)
    split_logits = logits_for(q, ids_split, split_count)
    probabilities["region_split"] = project_region_probabilities(split_logits, q, ids_split, shape, output_size)
    if not torch.equal(q, original_q):
        raise RuntimeError("Native Q was modified during the intervention")
    for name, prob in probabilities.items():
        if not torch.isfinite(prob).all() or prob.min() < 0:
            raise ValueError("Invalid probabilities in " + name)
        torch.testing.assert_close(prob.sum(1), torch.ones_like(prob[:, 0]), atol=2e-5, rtol=2e-5)
    return probabilities, building, dict(
        native=routing_stats(q, ids, count),
        sharp=routing_stats(q_sharp, ids, count),
        predicted_split=routing_stats(q, ids_split, split_count),
    )


@torch.no_grad()
def gt_mixing_diagnostics(gt, q, ids, count, shape):
    """GT is consumed only AFTER predictions, for offline association statistics."""
    valid = (gt >= 0) & (gt < 4)
    fractions = F.interpolate(
        F.one_hot(gt.clamp(0, 3).long(), 4).permute(0, 3, 1, 2).float() * valid[:, None],
        shape, mode="area",
    )
    pooled = soft_pool_unpool(fractions, q, ids, count)
    class_mass = pooled["tokens"] * pooled["mass"][..., None]
    labelled_mass = class_mass.sum(-1)
    distribution = class_mass / labelled_mass.clamp_min(1e-8)[..., None]
    purity = distribution.max(-1).values
    foreground_fraction = distribution[..., 1:].sum(-1)
    binary_impurity = 2 * foreground_fraction * distribution[..., 0]
    entropy = -(q * q.clamp_min(1e-8).log()).sum(1) / math.log(9)
    per_class = {}
    for c, label in enumerate(("Background", "Intact", "Damaged", "Destroyed")):
        weight = class_mass[..., c]
        area = float(weight.sum())
        pixel_weight = fractions[:, c].flatten(1)
        per_class[label] = dict(
            label_mass_feature_pixels=area,
            purity_weighted_sum=float((weight * purity).sum()),
            binary_impurity_weighted_sum=float((weight * binary_impurity).sum()),
            entropy_weighted_sum=float((pixel_weight * entropy).sum()),
            entropy_weight=float(pixel_weight.sum()),
        )
    hard = ids[None].expand(q.shape[0], -1, -1).gather(1, q.argmax(1, keepdim=True))[:, 0]
    purity_map = purity.gather(1, hard).reshape(1, 1, *shape)
    buckets = torch.where(purity_map >= 0.95, 2, torch.where(purity_map >= 0.8, 1, 0))
    return dict(per_class=per_class, labelled_regions=int((labelled_mass > 1e-6).sum())), buckets
