from __future__ import annotations
import math
import torch

from .config import PIX_INTACT, PIX_DAMAGED

def iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    cm = cm.float()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    return tp / union.clamp_min(1.0)

def metrics_from_cm(cm: torch.Tensor) -> dict:
    cm = cm.float()
    iou = iou_from_cm(cm)
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1.0)
    recall = tp / cm.sum(1).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    d2i = cm[2,1] / cm[2].sum().clamp_min(1.0)
    return {
        "miou4": float(iou.mean()),
        "valid_miou": float(iou[1:4].mean()),
        "background_iou": float(iou[0]),
        "intact_iou": float(iou[1]),
        "damaged_iou": float(iou[2]),
        "destroyed_iou": float(iou[3]),
        "intact_f1": float(f1[1]),
        "damaged_f1": float(f1[2]),
        "destroyed_f1": float(f1[3]),
        "damaged_to_intact": float(d2i),
    }

def intact_only_override_cm(building_cm: torch.Tensor) -> torch.Tensor:
    out = building_cm.clone()
    out[:, PIX_DAMAGED] += out[:, PIX_INTACT]
    out[:, PIX_INTACT] = 0
    return out

def apply_selection(base_cm: torch.Tensor, building_cms: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    out = base_cm.clone()
    for i in rows.tolist():
        old = building_cms[i]
        out = out - old + intact_only_override_cm(old)
    return out

def exact_single_utility(base_cm: torch.Tensor, building_cm: torch.Tensor) -> float:
    before = metrics_from_cm(base_cm)["valid_miou"]
    after = metrics_from_cm(base_cm - building_cm + intact_only_override_cm(building_cm))["valid_miou"]
    return after - before

def fixed_coverage_select(score: torch.Tensor, candidate: torch.Tensor, coverage: float = 0.02) -> torch.Tensor:
    rows = torch.where(candidate)[0]
    n = max(1, int(round(len(rows) * coverage)))
    order = torch.argsort(score[rows], descending=True)
    return rows[order[:n]]

def audit_selection(cache: dict, selected: torch.Tensor) -> dict:
    labels = cache["labels"].long()[selected]
    purity = cache["label_purity"].float()[selected]
    clean = purity >= 0.70
    clean_d = clean & (labels == 1)
    clean_i = clean & (labels == 0)
    clean_x = clean & (labels == 2)
    unknown = (~clean) | (labels < 0)

    cm = cache["building_pixel_cm"][selected].long()
    # Correct D pixels that are currently predicted I and will be moved to D.
    rescue_px = int(cm[:, 2, 1].sum())
    # Correct I pixels currently predicted I that will be harmed.
    harm_px = int(cm[:, 1, 1].sum())
    # True X pixels predicted I that will become D.
    x_px = int(cm[:, 3, 1].sum())

    return {
        "selected": int(len(selected)),
        "clean_rescue_buildings": int(clean_d.sum()),
        "clean_harm_buildings": int(clean_i.sum()),
        "clean_destroyed_buildings": int(clean_x.sum()),
        "unknown_or_impure": int(unknown.sum()),
        "rescue_pixels_D_to_I_reversed": rescue_px,
        "harm_pixels_I_correct_to_D": harm_px,
        "destroyed_pixels_X_pred_I_to_D": x_px,
    }

def evaluate_score(cache: dict, score: torch.Tensor, candidate: torch.Tensor, coverage: float, name: str) -> dict:
    selected = fixed_coverage_select(score, candidate, coverage)
    base = metrics_from_cm(cache["base_pixel_cm"])
    new_cm = apply_selection(cache["base_pixel_cm"], cache["building_pixel_cm"], selected)
    after = metrics_from_cm(new_cm)
    out = {
        "name": name,
        "coverage": coverage,
        "candidate_count": int(candidate.sum()),
        "base_metrics": base,
        "metrics": after,
        "delta_valid_miou_pp": 100.0*(after["valid_miou"] - base["valid_miou"]),
        "delta_damaged_iou_pp": 100.0*(after["damaged_iou"] - base["damaged_iou"]),
        "delta_damaged_f1_pp": 100.0*(after["damaged_f1"] - base["damaged_f1"]),
        **audit_selection(cache, selected),
        "selected_indices": selected.tolist(),
    }
    return out

def binary_auroc(y: torch.Tensor, s: torch.Tensor) -> float:
    y = y.long()
    sp = s[y == 1].float()
    sn = s[y == 0].float()
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    wins = 0.0
    total = 0
    for i in range(0, len(sp), 512):
        d = sp[i:i+512, None] - sn[None, :]
        wins += float((d > 0).sum()) + 0.5 * float((d == 0).sum())
        total += d.numel()
    return wins / max(1, total)

def average_precision(y: torch.Tensor, s: torch.Tensor) -> float:
    y = y.long()
    order = torch.argsort(s, descending=True)
    yy = y[order].float()
    p = int(yy.sum())
    if p == 0: return float("nan")
    ctp = torch.cumsum(yy, 0)
    precision = ctp / torch.arange(1, len(yy)+1, device=yy.device)
    return float((precision * yy).sum() / p)
