from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist, metrics_from_hist
from .core import augment_metrics, event_name, parse_batch


@torch.no_grad()
def evaluate(model, loader, device, ignore_index: int = 255, max_batches: int = 0):
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    new_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    base_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hists: Dict[str, torch.Tensor] = {}
    counts = defaultdict(float)
    max_probability_error = 0.0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        pre, post, location, damage, names = parse_batch(batch, device)
        loc_logits, new_logits, aux = model(pre, post, return_aux=True)
        base_logits = torch.nn.functional.interpolate(
            aux["parent_damage_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False)
        pred_loc, new_pred, base_pred = loc_logits.argmax(1), new_logits.argmax(1), base_logits.argmax(1)
        loc_hist += fast_hist(pred_loc, location, 2, ignore_index)
        new_hist += fast_hist(new_pred, damage, 4, ignore_index)
        base_hist += fast_hist(base_pred, damage, 4, ignore_index)
        valid = damage != ignore_index
        base_ok, new_ok = (base_pred == damage) & valid, (new_pred == damage) & valid
        counts["base_wrong"] += float((valid & ~base_ok).sum())
        counts["base_correct"] += float(base_ok.sum())
        counts["rescue"] += float((~base_ok & new_ok & valid).sum())
        counts["harm"] += float((base_ok & ~new_ok).sum())
        counts["changed"] += float(((new_pred != base_pred) & valid).sum())
        counts["valid"] += float(valid.sum())
        candidate = aux["candidate_gate"] > 0
        accepted = candidate & (aux["selective_gate"] > 0)
        counts["candidate_pixels"] += float(candidate.sum())
        counts["accepted_pixels"] += float(accepted.sum())
        for bi in range(aux["building_ids"].shape[0]):
            ids = aux["building_ids"][bi]
            for building_id in torch.unique(ids).tolist():
                if int(building_id) <= 0:
                    continue
                region = (ids == int(building_id)).unsqueeze(0)
                if bool((region & candidate[bi]).any()):
                    counts["candidate_buildings"] += 1.0
                    if bool((region & accepted[bi]).any()):
                        counts["accepted_buildings"] += 1.0
        new_prob = torch.softmax(new_logits.float(), 1)
        base_prob = torch.softmax(base_logits.float(), 1)
        max_probability_error = max(max_probability_error, float((new_prob - base_prob).abs().max()))
        for i, name in enumerate(names):
            key = event_name(str(name))
            hist = fast_hist(new_pred[i:i+1], damage[i:i+1], 4, ignore_index)
            event_hists[key] = event_hists.get(key, torch.zeros_like(hist)) + hist
    metrics = augment_metrics(metrics_from_hist(loc_hist, new_hist), loc_hist, new_hist)
    base_metrics = augment_metrics(metrics_from_hist(loc_hist, base_hist), loc_hist, base_hist)
    per_event = {}
    for key, hist in event_hists.items():
        inter = torch.diag(hist); union = hist.sum(1) + hist.sum(0) - inter
        per_event[key] = float((inter / union.clamp_min(1.0)).mean())
    metrics.update({
        "event_miou4_by_name": per_event,
        "event_mean_miou4": sum(per_event.values()) / max(1, len(per_event)),
        "event_worst_miou4": min(per_event.values()) if per_event else 0.0,
        "rescue_rate": counts["rescue"] / max(1.0, counts["base_wrong"]),
        "harm_rate": counts["harm"] / max(1.0, counts["base_correct"]),
        "rescue_count": int(counts["rescue"]), "harm_count": int(counts["harm"]),
        "net_correction_count": int(counts["rescue"] - counts["harm"]),
        "changed_pixel_fraction": counts["changed"] / max(1.0, counts["valid"]),
        "accepted_candidate_pixel_fraction": counts["accepted_pixels"] / max(1.0, counts["candidate_pixels"]),
        "accepted_candidate_building_fraction": counts["accepted_buildings"] / max(1.0, counts["candidate_buildings"]),
        "candidate_building_count": int(counts["candidate_buildings"]),
        "max_probability_difference_vs_base": max_probability_error,
    })
    return {"metrics": metrics, "base_metrics": base_metrics,
            "gain_valid_miou_pp": 100.0 * (metrics["valid_miou"] - base_metrics["valid_miou"])}


def format_result(result) -> str:
    m = result["metrics"]
    return (f"valid_mIoU={100*m['valid_miou']:.4f} D-F1={100*m['damaged_f1']:.4f} "
            f"IoU(I/D/X)={100*m['intact_iou']:.2f}/{100*m['damaged_iou']:.2f}/{100*m['destroyed_iou']:.2f} "
            f"D->I={100*m['damaged_to_intact']:.2f} gain={result['gain_valid_miou_pp']:+.4f}pp "
            f"rescue/harm={100*m['rescue_rate']:.2f}/{100*m['harm_rate']:.2f}%")
