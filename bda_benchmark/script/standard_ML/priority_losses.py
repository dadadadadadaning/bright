from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def ramp_weight(step: int, target: float, warmup: int, ramp: int) -> float:
    if step < warmup:
        return 0.0
    if ramp > 0 and step < warmup + ramp:
        return float(target) * float(step - warmup) / float(ramp)
    return float(target)


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    if p == 0:
        return gt_sorted
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        # Out-of-place: compatible with PyTorch 2.11 overlapping-slice checks.
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]], dim=0)
    return jaccard


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore: int = 255) -> torch.Tensor:
    if labels.shape[-2:] != probas.shape[-2:]:
        labels = F.interpolate(labels[:, None].float(), size=probas.shape[-2:], mode="nearest")[:, 0].long()
    c = probas.shape[1]
    p = probas.permute(0, 2, 3, 1).reshape(-1, c)
    y = labels.reshape(-1)
    valid = y != ignore
    p, y = p[valid], y[valid]
    losses: List[torch.Tensor] = []
    for cls in range(c):
        fg = (y == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - p[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else probas.sum() * 0.0


def segmentation_loss(logits, labels, ignore=255):
    ce = F.cross_entropy(logits.float(), labels, ignore_index=ignore)
    lv = lovasz_softmax(torch.softmax(logits.float(), 1), labels, ignore)
    return ce + lv, ce.detach(), lv.detach()


def intact_alignment_loss(outputs: Dict[str, torch.Tensor], damage_gt: torch.Tensor, ignore=255):
    losses, stats = [], {}
    for stage in (3, 4):
        r, s = outputs[f"ia_rgb_s{stage}"].float(), outputs[f"ia_sar_s{stage}"].float()
        gt = F.interpolate(damage_gt[:, None].float(), r.shape[-2:], mode="nearest")[:, 0].long()
        dist = 1.0 - (F.normalize(r, dim=1) * F.normalize(s, dim=1)).sum(1)
        m = (gt == 1) & (gt != ignore)
        if m.any():
            losses.append(dist[m].mean())
            stats[f"intact_dist_s{stage}"] = dist[m].detach().mean()
    if not losses:
        return outputs["output_dam"].sum() * 0.0, stats
    return torch.stack(losses).mean(), stats


def flat_instance_loss(outputs: Dict[str, torch.Tensor]):
    labels = outputs["instance_labels"]
    valid = (labels >= 1) & (labels <= 3)
    if not valid.any():
        return outputs["output_dam"].sum() * 0.0, {"instance_count": 0, "destroyed_gt": 0, "destroyed_pred": 0}
    logits = outputs["building_logits3"][valid]
    y = labels[valid] - 1
    loss = F.cross_entropy(logits, y)
    pred = logits.argmax(-1)
    return loss, {"instance_count": int(y.numel()), "destroyed_gt": int((y == 2).sum()), "destroyed_pred": int((pred == 2).sum())}


def hierarchical_instance_loss(outputs: Dict[str, torch.Tensor]):
    labels = outputs["instance_labels"]
    valid = (labels >= 1) & (labels <= 3)
    if not valid.any():
        return outputs["output_dam"].sum() * 0.0, {"instance_count": 0, "destroyed_gt": 0, "destroyed_pred": 0}
    y = labels[valid]
    exist_target = (y >= 2).long()
    exist_loss = F.cross_entropy(outputs["exist_logits"][valid], exist_target)
    damage_mask = y >= 2
    if damage_mask.any():
        severity_target = (y[damage_mask] == 3).long()
        severity_loss = F.cross_entropy(outputs["severity_logits"][valid][damage_mask], severity_target)
    else:
        severity_loss = exist_loss * 0.0
    pred = outputs["building_logits3"][valid].argmax(-1)
    return exist_loss + severity_loss, {
        "instance_count": int(y.numel()), "destroyed_gt": int((y == 3).sum()),
        "destroyed_pred": int((pred == 2).sum()), "exist_loss": float(exist_loss.detach()),
        "severity_loss": float(severity_loss.detach()),
    }


def class_conditional_consistency_loss(outputs: Dict[str, torch.Tensor], damage_gt: torch.Tensor):
    """Weak instance-to-pixel consistency on instance-average probabilities.

    The teacher instance distribution is detached. Class weights are:
    intact=1.0, damaged=0.2, destroyed=0.5.
    """
    records = outputs.get("instance_records", [])
    if not records:
        return outputs["output_dam"].sum() * 0.0
    p_pix = torch.softmax(outputs["output_dam"].float(), 1)
    p_inst = torch.softmax(outputs["building_logits3"].float(), -1).detach()
    weights = {1: 1.0, 2: 0.2, 3: 0.5}
    terms = []
    for i, rec in enumerate(records):
        # records are generated from a lower-resolution map; use GT building support
        # at output resolution to obtain a stable per-image class-conditioned mean.
        m = damage_gt[rec.batch_index] == rec.label
        if not m.any() or rec.label not in weights:
            continue
        q = p_pix[rec.batch_index, 1:4, m].mean(dim=1).clamp_min(1e-8)
        q = q / q.sum().clamp_min(1e-8)
        target = p_inst[i].clamp_min(1e-8)
        terms.append(weights[rec.label] * F.kl_div(q.log(), target, reduction="sum"))
    return torch.stack(terms).mean() if terms else outputs["output_dam"].sum() * 0.0
