"""Train-only label-read supervision. Does not construct or change routes.

Targets belong in the loss, never the classifier/query/retrieval interface.
Reduction is an equal mean over valid query slots, not a pixel or class mean.
The loss is an auxiliary objective; it does not change inference probabilities.
"""
from __future__ import annotations

import torch


def read_supervision(probability, target_distribution, valid, target_mass,
                     candidate_class_max=None):
    """Return (scalar loss, detached JSON-compatible diagnostic statistics).

    probability/target_distribution: [B,R,2,4]; valid/target_mass: [B,R,2].
    ``valid`` says the query and at least one legal candidate are valid.
    Missing candidate classes are reported, NOT silently removed from targets.
    ``candidate_class_max`` is a per-class componentwise bound, NOT a jointly
    attainable oracle distribution. It is never used to choose candidates.
    """
    if probability.ndim != 4 or probability.shape[-2:] != (2, 4):
        raise ValueError("read probabilities must have shape [B,R,2,4]")
    if target_distribution.shape != probability.shape:
        raise ValueError("target distribution shape differs")
    if valid.shape != probability.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,R,2]")
    if target_mass.shape != valid.shape:
        raise ValueError("target mass shape differs")
    tensors = (probability, target_distribution, valid, target_mass)
    if any(x.device != probability.device for x in tensors):
        raise ValueError("read supervision tensors must share a device")
    for name, x in (("probability", probability), ("target", target_distribution),
                    ("target_mass", target_mass)):
        if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
            raise ValueError(name + " must be finite floating point")
        if bool((x < 0).any()):
            raise ValueError(name + " must be nonnegative")
    target = target_distribution.detach()
    mass = target_mass.detach()
    has_gt = mass > 0
    if has_gt.any() and not torch.allclose(target[has_gt].sum(-1),
                                          torch.ones_like(mass[has_gt]),
                                          atol=2e-5, rtol=2e-5):
        raise ValueError("positive-mass targets must sum to one")
    if valid.any() and not torch.allclose(probability[valid].sum(-1),
                                         torch.ones_like(mass[valid]),
                                         atol=2e-5, rtol=2e-5):
        raise ValueError("valid read probabilities must sum to one")
    mask = valid & has_gt
    # The floor protects the log only. No route or returned probability changes.
    per_slot = -(target * probability.clamp_min(1e-8).log()).sum(-1)
    loss = per_slot[mask].mean() if bool(mask.any()) else probability.sum() * 0.0
    with torch.no_grad():
        n = int(mask.sum())
        stats = dict(
            reduction="equal_mean_over_valid_query_slots",
            valid_query_slots=n,
            valid_gt_slots=int(has_gt.sum()),
            gt_slots_without_support=int((has_gt & ~valid).sum()),
            read_loss=float(loss.detach()),
            label_gt_used_in_model_forward=False,
            zero_valid_slots=n == 0,
        )
        if n:
            stats.update(
                mean_target_distribution=target[mask].mean(0).cpu().tolist(),
                mean_read_distribution=probability.detach()[mask].mean(0).cpu().tolist(),
                mean_target_read_l1=float((target[mask] - probability.detach()[mask]).abs().sum(-1).mean()),
            )
        if candidate_class_max is not None:
            cmax = candidate_class_max.detach()
            if cmax.shape != probability.shape or cmax.device != probability.device:
                raise ValueError("candidate_class_max shape/device differs")
            if not bool(torch.isfinite(cmax).all()) or bool(((cmax < 0) | (cmax > 1 + 2e-5)).any()):
                raise ValueError("invalid candidate class bound")
            stats.update(
                candidate_max_is_componentwise_not_joint_oracle=True,
                mean_target_mass_on_absent_candidate_classes=(
                    float((target[mask] * (cmax[mask] <= 1e-8)).sum(-1).mean()) if n else None
                ),
                mean_candidate_class_max=cmax[mask].mean(0).cpu().tolist() if n else None,
            )
        for role in range(2):
            rm = mask[..., role]
            stats["role_%d_valid_slots" % role] = int(rm.sum())
            stats["role_%d_read_loss" % role] = float(per_slot[..., role][rm].mean()) if rm.any() else None
    return loss, stats


def query_slot_targets(slots_module, example, support, *, split):
    """Use the original area/ignore-aware pooling strictly outside forward.

    The training driver additionally checks image membership in Train. This
    helper refuses Val/Test optimization even if a caller supplies their GT.
    """
    if split != "train":
        raise ValueError("query targets for auxiliary optimization are Train-only")
    with torch.no_grad():
        shape = tuple(example.common["pixel_features"].shape[-2:])
        count = 2 * int(example.common["region_f"].shape[1])
        target, mass = slots_module.slot_labels(
            example.target, support["route_q"].detach(), support["route_ids"],
            shape=shape, count=count,
        )
        return target.detach(), mass.detach()
