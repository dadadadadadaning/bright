from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def ramp_weight(step: int, base: float, warmup: int, ramp: int) -> float:
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return float(base)
    return float(base) * min(1.0, max(0.0, (step - warmup) / float(ramp)))


def hierarchical_instance_loss(
    outputs: Dict[str, torch.Tensor],
    class_weights_3: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    labels = outputs["instance_labels"]
    if labels.numel() == 0:
        zero = outputs["output_dam"].sum() * 0.0
        return zero, {"inst_count": 0.0, "exist_loss": 0.0, "severity_loss": 0.0}
    exist_target = (labels >= 2).long()
    exist_loss = F.cross_entropy(outputs["exist_logits"].float(), exist_target)
    sev_mask = labels >= 2
    severity_loss = exist_loss * 0.0
    if sev_mask.any():
        sev_target = (labels[sev_mask] == 3).long()
        severity_loss = F.cross_entropy(outputs["severity_logits"][sev_mask].float(), sev_target)
    direct_target = labels - 1
    direct_loss = F.cross_entropy(outputs["building_logits3"].float(), direct_target, weight=class_weights_3)
    total = exist_loss + severity_loss + 0.5 * direct_loss
    with torch.no_grad():
        pred = outputs["building_logits3"].argmax(-1) + 1
        destroyed_pred = float((pred == 3).sum().item())
        destroyed_gt = float((labels == 3).sum().item())
    return total, {
        "inst_count": float(labels.numel()),
        "exist_loss": float(exist_loss.detach()),
        "severity_loss": float(severity_loss.detach()),
        "direct_loss": float(direct_loss.detach()),
        "destroyed_pred_instances": destroyed_pred,
        "destroyed_gt_instances": destroyed_gt,
    }


def intact_alignment_loss(outputs: Dict[str, torch.Tensor], damage_gt: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    losses = []
    for suffix in ("s3", "s4"):
        rgb = outputs[f"ia_rgb_{suffix}"]
        sar = outputs[f"ia_sar_{suffix}"]
        gt = F.interpolate(damage_gt[:,None].float(), size=rgb.shape[-2:], mode="nearest")[:,0].long()
        mask = gt == 1
        if mask.any():
            r = F.normalize(rgb, dim=1)
            s = F.normalize(sar, dim=1)
            dist = 1.0 - (r * s).sum(1)
            losses.append(dist[mask].mean())
    return torch.stack(losses).mean() if losses else outputs["output_dam"].sum() * 0.0
