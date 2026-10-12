# -*- coding: utf-8 -*-
"""
S3V5 K=[1,3,3] role-aware gated prototype evaluator.

Supports:
- best_model.pth (plain state_dict)
- best_training_state.pth (dict with model/optimizer/args)
- active vs main-only evaluation from one forward pass
- clean lambda sweep reconstructed from main logits + gated evidence
- global metrics, confusion, underestimation errors
- prototype role usage/cosine, gate/evidence, rescue/harm transitions
- correct per-event aggregation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
BDA_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = BDA_ROOT.parent
for root in (PROJECT_ROOT, BDA_ROOT):
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BGMP_K3Gated_DDP import DamageFormer_BGMP_K3Gated_DDP

CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]
DAMAGE_NAMES = ["Intact", "Damaged", "Destroyed"]
ROLE_NAMES = ["core", "hard_to_intact", "diverse"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_data_list_path", required=True)
    p.add_argument("--test_data_list_path", required=True)
    p.add_argument("--split", choices=["val", "test", "both"], default="both")
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")

    # None => read from training-state args, else use S3V5 defaults.
    p.add_argument("--roi_alpha", type=float, default=None)
    p.add_argument("--proto_dim", type=int, default=None)
    p.add_argument("--proto_tau", type=float, default=None)
    p.add_argument("--proto_warmup_iters", type=int, default=None)
    p.add_argument("--core_confidence", type=float, default=None)
    p.add_argument("--lambda_damaged", type=float, default=None)
    p.add_argument("--lambda_destroyed", type=float, default=None)
    p.add_argument("--proto_evidence_clip", type=float, default=None)
    p.add_argument("--proto_momentum", type=float, default=0.99)

    p.add_argument(
        "--lambda_sweep",
        default="0,0.02,0.05,0.08",
        help="Clean shared-lambda sweep. Empty string disables it.",
    )
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def read_list(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = [x.strip() for x in p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    if not data:
        raise RuntimeError(f"Empty data list: {p}")
    return data


def unpack_checkpoint(path: str):
    obj = torch.load(path, map_location="cpu")
    meta: Dict[str, Any] = {}
    ckpt_args: Dict[str, Any] = {}

    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
        meta = {
            "format": "training_state",
            "optimizer_step": obj.get("optimizer_step"),
            "best_miou": obj.get("best_miou"),
        }
        raw_args = obj.get("args", {})
        if isinstance(raw_args, argparse.Namespace):
            ckpt_args = vars(raw_args)
        elif isinstance(raw_args, dict):
            ckpt_args = dict(raw_args)
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        meta = {"format": "state_dict_wrapper"}
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
        meta = {"format": "model_state_dict_wrapper"}
    elif isinstance(obj, dict):
        state = obj
        meta = {"format": "plain_state_dict"}
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj).__name__}")

    state = {k.removeprefix("module."): v for k, v in state.items()}
    return state, ckpt_args, meta


def pick(cli_value: Any, ckpt_args: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    value = ckpt_args.get(key, None)
    return default if value is None else value


def disable_unused_swin_heads(model: nn.Module) -> None:
    for encoder_name in ("encoder_1", "encoder_2"):
        encoder = getattr(model, encoder_name, None)
        swin = getattr(encoder, "swin_transformer", None)
        if swin is not None:
            if hasattr(swin, "norm"):
                swin.norm = nn.Identity()
            if hasattr(swin, "head"):
                swin.head = nn.Identity()


def build_model(args, state, ckpt_args, device):
    proto_key = "damage_proto_memory.prototypes"
    inferred_dim = int(state[proto_key].shape[-1]) if proto_key in state else 256
    cfg = {
        "roi_alpha": float(pick(args.roi_alpha, ckpt_args, "roi_alpha", 0.1)),
        "proto_dim": int(pick(args.proto_dim, ckpt_args, "proto_dim", inferred_dim)),
        "proto_tau": float(pick(args.proto_tau, ckpt_args, "proto_tau", 0.1)),
        "proto_warmup_iters": int(pick(args.proto_warmup_iters, ckpt_args, "proto_warmup_iters", 5000)),
        "core_confidence": float(pick(args.core_confidence, ckpt_args, "core_confidence", 0.70)),
        "lambda_damaged": float(pick(args.lambda_damaged, ckpt_args, "lambda_damaged", 0.05)),
        "lambda_destroyed": float(pick(args.lambda_destroyed, ckpt_args, "lambda_destroyed", 0.05)),
        "evidence_clip": float(pick(args.proto_evidence_clip, ckpt_args, "proto_evidence_clip", 4.0)),
    }

    model = DamageFormer_BGMP_K3Gated_DDP(
        roi_alpha=cfg["roi_alpha"],
        detach_building_prior=True,
        use_pretrained_swin=False,
        proto_dim=cfg["proto_dim"],
        proto_momentum=args.proto_momentum,
        proto_tau=cfg["proto_tau"],
        proto_warmup_iters=cfg["proto_warmup_iters"],
        core_confidence=cfg["core_confidence"],
        lambda_damaged=cfg["lambda_damaged"],
        lambda_destroyed=cfg["lambda_destroyed"],
        evidence_clip=cfg["evidence_clip"],
    )
    disable_unused_swin_heads(model)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg


def cmatrix(gt: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    gt = np.asarray(gt).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    valid = (gt >= 0) & (gt < n) & (pred >= 0) & (pred < n)
    code = gt[valid].astype(np.int64) * n + pred[valid].astype(np.int64)
    return np.bincount(code, minlength=n * n).reshape(n, n)


def div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def full_metrics(cm: np.ndarray, absent_nan: bool = False) -> Dict[str, Any]:
    x = cm.astype(np.float64)
    tp = np.diag(x)
    row = x.sum(1)
    col = x.sum(0)
    union = row + col - tp
    precision = div(tp, col)
    recall = div(tp, row)
    f1 = div(2 * precision * recall, precision + recall)
    iou = div(tp, union)
    if absent_nan:
        iou[union == 0] = np.nan
        miou = float(np.nanmean(iou))
    else:
        miou = float(iou.mean())
    return {
        "oa": float(tp.sum() / max(x.sum(), 1.0) * 100),
        "class_precision": (precision * 100).tolist(),
        "class_recall": (recall * 100).tolist(),
        "class_f1": (f1 * 100).tolist(),
        "class_iou": [None if np.isnan(v) else float(v * 100) for v in iou],
        "miou": miou * 100,
        "_iou": iou,
    }


def damage_f1(building_cm: np.ndarray) -> Dict[str, Any]:
    x = building_cm.astype(np.float64)
    tp = np.diag(x)
    precision = div(tp, x.sum(0))
    recall = div(tp, x.sum(1))
    f1 = div(2 * precision * recall, precision + recall)[1:4]
    harmonic = 0.0 if np.any(f1 <= 0) else float(3.0 / np.sum(1.0 / f1))
    return {
        "damage_class_precision": (precision[1:4] * 100).tolist(),
        "damage_class_recall": (recall[1:4] * 100).tolist(),
        "damage_class_f1": (f1 * 100).tolist(),
        "damage_harmonic_f1": harmonic * 100,
    }


def loc_metrics(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {"loc_precision": p * 100, "loc_recall": r * 100, "loc_f1": f1 * 100, "loc_iou": iou * 100}


def error_rates(cm: np.ndarray) -> Dict[str, float]:
    x = cm.astype(np.float64)
    d = max(x[2].sum(), 1.0)
    z = max(x[3].sum(), 1.0)
    i = max(x[1].sum(), 1.0)
    return {
        "damaged_to_background": x[2, 0] / d * 100,
        "damaged_to_intact": x[2, 1] / d * 100,
        "damaged_to_destroyed": x[2, 3] / d * 100,
        "destroyed_to_background": x[3, 0] / z * 100,
        "destroyed_to_intact": x[3, 1] / z * 100,
        "destroyed_to_damaged": x[3, 2] / z * 100,
        "intact_to_damaged": x[1, 2] / i * 100,
        "intact_to_destroyed": x[1, 3] / i * 100,
    }


def row_percent(cm: np.ndarray) -> List[List[float]]:
    x = cm.astype(np.float64)
    return (x / np.maximum(x.sum(1, keepdims=True), 1.0) * 100).tolist()


def event_name(sample: str) -> str:
    token = str(sample).strip().split()[0]
    stem = Path(token.replace("\\", "/")).stem.lower()
    stem = re.sub(r"_(pre|post)_disaster$", "", stem)
    stem = re.sub(r"_(pre|post)$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem or "unknown"


def sample_ref(data_idx: Any, pos: int, names: List[str]) -> str:
    if torch.is_tensor(data_idx):
        item = data_idx[pos]
        item = int(item.item()) if item.numel() == 1 else item.detach().cpu().tolist()
    elif isinstance(data_idx, (list, tuple)):
        item = data_idx[pos]
    else:
        item = data_idx
    if isinstance(item, (int, np.integer)) and 0 <= int(item) < len(names):
        return names[int(item)]
    return str(item)


class Running:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None):
        if value is None:
            return
        x = value.detach().float()
        if mask is not None:
            m = mask.detach().bool()
            if m.shape != x.shape:
                m = m.expand_as(x)
            x = x[m]
        else:
            x = x.reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel():
            self.total += float(x.sum().item())
            self.count += int(x.numel())

    def mean(self):
        return None if self.count == 0 else self.total / self.count


def amp_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def prototype_diagnostics(model: nn.Module) -> Dict[str, Any]:
    mem = model.damage_proto_memory
    p = F.normalize(mem.prototypes.detach().float().cpu(), dim=-1)
    valid = mem.prototype_valid.detach().cpu().bool()
    init = mem.prototype_initialized.detach().cpu().bool()
    usage = mem.prototype_usage.detach().float().cpu()
    roles = {}
    centres = []
    for c, name in enumerate(DAMAGE_NAMES):
        idx = torch.where(valid[c])[0]
        q = p[c, idx]
        u = usage[c] * valid[c].float()
        ratio = u / u.sum().clamp_min(1e-6)
        roles[name] = {
            "valid_slots": idx.tolist(),
            "initialized": init[c].int().tolist(),
            "usage_count": usage[c].tolist(),
            "usage_ratio": ratio.tolist(),
            "within_class_cosine": (q @ q.t()).tolist(),
        }
        active = valid[c] & init[c]
        if not bool(active.any()):
            active = valid[c]
        centres.append(F.normalize(p[c, active].mean(0), dim=0))
    centres = torch.stack(centres)
    norms = mem.prototypes.detach().float().cpu()[valid].norm(dim=-1)
    return {
        "prototype_shape": list(p.shape),
        "valid_mask": valid.int().tolist(),
        "initialized_mask": init.int().tolist(),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "roles": roles,
        "interclass_centre_cosine": (centres @ centres.t()).tolist(),
    }


def build_dataset(dataset_path: str, names: List[str]):
    return MultimodalDamageAssessmentDatset(dataset_path, names, 1024, None, "test")


@torch.no_grad()
def evaluate(model, dataset_path, names, split_name, batch_size, workers, device, amp, lambdas):
    dataset = build_dataset(dataset_path, names)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )

    modes = ["active", "main_only"] + [f"lambda_{x:g}" for x in lambdas if x != 0]
    full_cm = {m: np.zeros((4, 4), dtype=np.int64) for m in modes}
    building_cm = {m: np.zeros((4, 4), dtype=np.int64) for m in modes}
    event_cm = {m: defaultdict(lambda: np.zeros((4, 4), dtype=np.int64)) for m in modes}
    event_images = defaultdict(int)
    lcm = np.zeros((2, 2), dtype=np.int64)

    trans = defaultdict(int)
    proto_correct = proto_total = proto_active_batches = 0
    role_wins = np.zeros((3, 3), dtype=np.int64)
    stats = {k: Running() for k in [
        "p_build_building", "p_build_background",
        "gate_d_all", "gate_x_all", "gate_d_build", "gate_x_build",
        "evidence_d_positive", "evidence_x_positive",
        "correction_d", "correction_x",
    ]}

    for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
        pre, post, label_loc, label_clf, data_idx = batch[:5]
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        label_loc = label_loc.to(device, non_blocking=True).long()
        label_clf = label_clf.to(device, non_blocking=True).long()

        with amp_context(device, amp):
            out_loc, out_active, aux = model(
                pre, post, damage_target=None, current_iter=None,
                return_aux=True, force_proto_off=False,
            )
        proto_active_batches += int(bool(aux["proto_active"]))

        main_feat = aux["main_damage_logits_feat"].float()
        out_main = F.interpolate(main_feat, size=post.shape[-2:], mode="bilinear", align_corners=False)
        logits = {"active": out_active.float(), "main_only": out_main}

        gd = aux.get("proto_gate_damaged")
        gx = aux.get("proto_gate_destroyed")
        ed = aux.get("proto_evidence_damaged")
        ex = aux.get("proto_evidence_destroyed")
        if all(v is not None for v in (gd, gx, ed, ex)):
            for lam in lambdas:
                if lam == 0:
                    continue
                y = main_feat.clone()
                y[:, 2:3] += float(lam) * gd.float() * ed.float()
                y[:, 3:4] += float(lam) * gx.float() * ex.float()
                logits[f"lambda_{lam:g}"] = F.interpolate(y, size=post.shape[-2:], mode="bilinear", align_corners=False)

        pred_loc = out_loc.float().argmax(1).cpu().numpy()
        gt_loc = label_loc.cpu().numpy()
        valid_loc = gt_loc != 255
        lcm += cmatrix((gt_loc > 0).astype(np.int64)[valid_loc], (pred_loc > 0).astype(np.int64)[valid_loc], 2)

        gt = label_clf.cpu().numpy()
        main_pred = out_main.argmax(1).cpu().numpy()
        active_pred = out_active.float().argmax(1).cpu().numpy()

        def t(g, before, after, name):
            trans[name] += int(((gt == g) & (main_pred == before) & (active_pred == after)).sum())
        t(2, 1, 2, "damaged_intact_to_damaged_rescued")
        t(2, 2, 1, "damaged_damaged_to_intact_harmed")
        t(3, 1, 3, "destroyed_intact_to_destroyed_rescued")
        t(3, 3, 1, "destroyed_destroyed_to_intact_harmed")
        t(1, 1, 2, "intact_to_damaged_new_false_alarm")
        t(1, 1, 3, "intact_to_destroyed_new_false_alarm")
        trans["gt_damaged_total"] += int((gt == 2).sum())
        trans["gt_destroyed_total"] += int((gt == 3).sum())
        trans["gt_intact_total"] += int((gt == 1).sum())

        feat_size = main_feat.shape[-2:]
        target_feat = F.interpolate(label_clf[:, None].float(), size=feat_size, mode="nearest")[:, 0].long()
        loc_feat = F.interpolate(label_loc[:, None].float(), size=feat_size, mode="nearest")[:, 0].long()
        build_mask = ((loc_feat > 0) & (loc_feat != 255))[:, None]
        bg_mask = (loc_feat == 0)[:, None]
        pb = F.interpolate(aux["p_building"].float(), size=feat_size, mode="bilinear", align_corners=False)
        stats["p_build_building"].add(pb, build_mask)
        stats["p_build_background"].add(pb, bg_mask)
        stats["gate_d_all"].add(gd)
        stats["gate_x_all"].add(gx)
        stats["gate_d_build"].add(gd, build_mask)
        stats["gate_x_build"].add(gx, build_mask)
        if ed is not None:
            stats["evidence_d_positive"].add((ed > 0).float())
        if ex is not None:
            stats["evidence_x_positive"].add((ex > 0).float())
        if gd is not None and ed is not None:
            stats["correction_d"].add(gd.float() * ed.float())
        if gx is not None and ex is not None:
            stats["correction_x"].add(gx.float() * ex.float())

        scores = aux.get("proto_class_scores")
        wins = aux.get("proto_winning_slot")
        if scores is not None:
            pp = scores.argmax(1) + 1
            valid = (target_feat >= 1) & (target_feat <= 3)
            proto_correct += int((pp[valid] == target_feat[valid]).sum())
            proto_total += int(valid.sum())
        if wins is not None:
            for c, lab in enumerate((1, 2, 3)):
                selected = wins[:, c][target_feat == lab]
                if selected.numel():
                    role_wins[c] += torch.bincount(selected.long(), minlength=3)[:3].cpu().numpy()

        for mode, lg in logits.items():
            pred = lg.argmax(1).cpu().numpy()
            full_cm[mode] += cmatrix(gt, pred, 4)
            bmask = (gt >= 1) & (gt <= 3)
            building_cm[mode] += cmatrix(gt[bmask], pred[bmask], 4)
            for b in range(gt.shape[0]):
                ev = event_name(sample_ref(data_idx, b, names))
                event_cm[mode][ev] += cmatrix(gt[b], pred[b], 4)
        for b in range(gt.shape[0]):
            event_images[event_name(sample_ref(data_idx, b, names))] += 1

    mode_results = {}
    for mode in modes:
        fm = full_metrics(full_cm[mode])
        fm.pop("_iou")
        target3 = float(np.mean([v for v in fm["class_iou"][1:4] if v is not None]))
        events = []
        for ev, cm in event_cm[mode].items():
            em = full_metrics(cm, absent_nan=True)
            raw = em.pop("_iou")
            present = raw[1:4][~np.isnan(raw[1:4])]
            events.append({
                "event": ev,
                "num_images": int(event_images[ev]),
                **em,
                "target3_miou_present": None if present.size == 0 else float(present.mean() * 100),
                **error_rates(cm),
            })
        events.sort(key=lambda x: float("inf") if x["target3_miou_present"] is None else x["target3_miou_present"])
        mode_results[mode] = {
            "global": {**loc_metrics(lcm), **fm, "target_3class_miou": target3, **damage_f1(building_cm[mode]), **error_rates(full_cm[mode])},
            "full_confusion_matrix": full_cm[mode].tolist(),
            "full_confusion_row_percent": row_percent(full_cm[mode]),
            "gt_building_confusion_matrix": building_cm[mode].tolist(),
            "events": events,
            "worst_event_by_target3_present": events[0] if events else None,
            "event_mean_target3_present": None if not events else float(np.mean([x["target3_miou_present"] for x in events if x["target3_miou_present"] is not None])),
        }

    def pct(num, den):
        return None if trans[den] <= 0 else trans[num] / trans[den] * 100
    transitions = {k: int(v) for k, v in trans.items()}
    transitions.update({
        "damaged_rescue_rate_over_gt": pct("damaged_intact_to_damaged_rescued", "gt_damaged_total"),
        "destroyed_rescue_rate_over_gt": pct("destroyed_intact_to_destroyed_rescued", "gt_destroyed_total"),
        "intact_new_damaged_false_alarm_rate": pct("intact_to_damaged_new_false_alarm", "gt_intact_total"),
        "intact_new_destroyed_false_alarm_rate": pct("intact_to_destroyed_new_false_alarm", "gt_intact_total"),
    })

    role_report = {}
    for c, name in enumerate(DAMAGE_NAMES):
        count = role_wins[c]
        total = int(count.sum())
        role_report[name] = {
            "role_names": ROLE_NAMES,
            "count": count.tolist(),
            "ratio": (count / total if total else np.zeros(3)).tolist(),
        }

    return {
        "split": split_name,
        "modes": mode_results,
        "diagnostics": {
            "samples": len(dataset),
            "proto_active_batches": int(proto_active_batches),
            "total_batches": len(loader),
            "prototype_class_accuracy": None if proto_total == 0 else proto_correct / proto_total * 100,
            "prototype_role_win_on_gt_class": role_report,
            "mean_p_building_on_gt_building": stats["p_build_building"].mean(),
            "mean_p_building_on_gt_background": stats["p_build_background"].mean(),
            "mean_gate_damaged_all": stats["gate_d_all"].mean(),
            "mean_gate_destroyed_all": stats["gate_x_all"].mean(),
            "mean_gate_damaged_gt_building": stats["gate_d_build"].mean(),
            "mean_gate_destroyed_gt_building": stats["gate_x_build"].mean(),
            "positive_evidence_ratio_damaged": stats["evidence_d_positive"].mean(),
            "positive_evidence_ratio_destroyed": stats["evidence_x_positive"].mean(),
            "mean_unscaled_correction_damaged": stats["correction_d"].mean(),
            "mean_unscaled_correction_destroyed": stats["correction_x"].mean(),
            "active_vs_main_transitions": transitions,
        },
    }


def print_summary(result: Dict[str, Any], mode: str):
    g = result["modes"][mode]["global"]
    print(f"\n---------------- {result['split']} | {mode} ----------------")
    print(f"Loc F1               : {g['loc_f1']:.4f}")
    print(f"Damage Harmonic F1   : {g['damage_harmonic_f1']:.4f}")
    print("Damage F1 [I/D/Dest] : " + " / ".join(f"{x:.4f}" for x in g["damage_class_f1"]))
    print(f"OA                   : {g['oa']:.4f}")
    print(f"mIoU                 : {g['miou']:.4f}")
    print(f"target3 mIoU         : {g['target_3class_miou']:.4f}")
    for name, value in zip(CLASS_NAMES, g["class_iou"]):
        print(f"  IoU {name:<10s}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"Damaged   -> Intact  : {g['damaged_to_intact']:.4f}%")
    print(f"Destroyed -> Intact  : {g['destroyed_to_intact']:.4f}%")
    print(f"Intact -> Damaged    : {g['intact_to_damaged']:.4f}%")
    print(f"Intact -> Destroyed  : {g['intact_to_destroyed']:.4f}%")


def print_delta(result: Dict[str, Any]):
    a = result["modes"]["active"]["global"]
    m = result["modes"]["main_only"]["global"]
    print(f"\n================ {result['split']} ACTIVE - MAIN_ONLY ================")
    for key in ["miou", "target_3class_miou", "damage_harmonic_f1", "damaged_to_intact", "destroyed_to_intact", "intact_to_damaged", "intact_to_destroyed"]:
        print(f"{key:<26s}: {a[key] - m[key]:+.4f}")
    print("Class IoU delta:")
    for name, av, mv in zip(CLASS_NAMES, a["class_iou"], m["class_iou"]):
        print(f"  {name:<10s}: {av - mv:+.4f}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    state, ckpt_args, meta = unpack_checkpoint(args.checkpoint)
    print("Loading checkpoint:", args.checkpoint)
    print("Checkpoint metadata:", meta)
    model, cfg = build_model(args, state, ckpt_args, device)
    print("STRICT CHECKPOINT LOAD: PASS")
    print("Resolved configuration:", cfg)

    pdiag = prototype_diagnostics(model)
    print("\n================ PROTOTYPE DIAGNOSTICS ================")
    print("shape:", pdiag["prototype_shape"])
    print("valid:", pdiag["valid_mask"])
    print("initialized:", pdiag["initialized_mask"])
    print("norm range:", pdiag["norm_min"], pdiag["norm_max"])
    for name, d in pdiag["roles"].items():
        print(name, "usage ratio:", d["usage_ratio"])
        print(name, "within-class cosine:", d["within_class_cosine"])
    print("interclass centre cosine:", pdiag["interclass_centre_cosine"])

    lambdas = sorted(set(float(x.strip()) for x in args.lambda_sweep.split(",") if x.strip()))
    if any(x < 0 for x in lambdas):
        raise ValueError("S3V5 rescue lambdas must be non-negative.")

    output = {
        "checkpoint": args.checkpoint,
        "checkpoint_meta": meta,
        "resolved_model_configuration": cfg,
        "prototype_diagnostics": pdiag,
        "splits": {},
    }

    if args.split in ("val", "both"):
        names = read_list(args.val_data_list_path)
        print("\nValidation samples:", len(names))
        r = evaluate(model, args.dataset_path, names, "validation", args.eval_batch_size, args.num_workers, device, args.amp, lambdas)
        output["splits"]["validation"] = r
        print_summary(r, "active")
        print_summary(r, "main_only")
        print_delta(r)
        sweep = [m for m in r["modes"] if m.startswith("lambda_")]
        if sweep:
            best = max(sweep, key=lambda m: r["modes"][m]["global"]["miou"])
            print("\nBest validation clean lambda mode:", best)
            print_summary(r, best)

    if args.split in ("test", "both"):
        names = read_list(args.test_data_list_path)
        print("\nTest samples:", len(names))
        r = evaluate(model, args.dataset_path, names, "testing", args.eval_batch_size, args.num_workers, device, args.amp, lambdas)
        output["splits"]["testing"] = r
        print_summary(r, "active")
        print_summary(r, "main_only")
        print_delta(r)
        print("\nTest lambda sweep is diagnostic only:")
        for mode in [m for m in r["modes"] if m.startswith("lambda_")]:
            g = r["modes"][mode]["global"]
            print(f"  {mode:<12s} mIoU={g['miou']:.4f}, DmgIoU={g['class_iou'][2]:.4f}, DestIoU={g['class_iou'][3]:.4f}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nEvaluation JSON saved to:", out)


if __name__ == "__main__":
    main()
