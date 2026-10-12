#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified evaluator for DamageFormer-family variants."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
BDA_ROOT = os.path.join(ROOT, "bda_benchmark")
if BDA_ROOT not in sys.path:
    sys.path.insert(0, BDA_ROOT)


EVENT_NAMES = [
    "turkey-earthquake",
    "hawaii-wildfire",
    "morocco-earthquake",
    "haiti-earthquake",
    "la_palma-volcano",
    "congo-volcano",
    "beirut-explosion",
    "bata-explosion",
    "libya-flood",
    "noto-earthquake",
    "marshall-wildfire",
    "ukraine-conflict",
    "myanmar-hurricane",
    "mexico-hurricane",
]

DISASTER_TYPES = [
    "earthquake",
    "wildfire",
    "volcano",
    "explosion",
    "flood",
    "conflict",
    "hurricane",
]

DAMAGE_CLASSES = OrderedDict(
    [
        (0, "background"),
        (1, "intact"),
        (2, "damaged"),
        (3, "destroyed"),
    ]
)

IGNORE_INDEX = 255
EVENT_DIAGNOSIS_GAP = 0.05
LOW_DESTROYED_SUPPORT_THRESHOLD = 10000


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def str2bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no_{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def import_dataset_class():
    try:
        from dataset.make_data_loader import MultimodalDamageAssessmentDatset

        return MultimodalDamageAssessmentDatset
    except Exception:
        from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset

        return MultimodalDamageAssessmentDatset


def import_model_class(model_type: str):
    if model_type == "DamageFormer":
        try:
            from model.DamageFormer import DamageFormer
        except Exception:
            from bda_benchmark.model.DamageFormer import DamageFormer

        return DamageFormer
    if model_type == "DamageFormer_BGCA":
        try:
            from model.DamageFormer_BGCA import DamageFormer_BGCA
        except Exception:
            from bda_benchmark.model.DamageFormer_BGCA import DamageFormer_BGCA

        return DamageFormer_BGCA
    if model_type == "DamageFormer_S8_IA_MLA":
        try:
            from model.DamageFormer_S8_IA_MLA import DamageFormer_S8_IA_MLA
        except Exception:
            from bda_benchmark.model.DamageFormer_S8_IA_MLA import DamageFormer_S8_IA_MLA

        return DamageFormer_S8_IA_MLA
    if model_type == "DamageFormer_S8_MICB_DSMIL":
        try:
            from model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
        except Exception:
            from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL

        return DamageFormer_S8_MICB_DSMIL
    raise ValueError(f"Unsupported model_type: {model_type}")


def filter_init_kwargs(model_cls, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls.__init__)
    allowed = {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model_cls = import_model_class(args.model_type)
    candidate_kwargs = {
        "roi_mode": args.roi_mode,
        "roi_alpha": args.roi_alpha,
        "roi_dilate": args.roi_dilate,
        "roi_fusion": args.roi_fusion,
        "roi_beta": args.roi_beta,
        "detach_mask": args.detach_mask,
        "use_align": args.use_align,
        "align_type": args.align_type,
        "use_shared_diff": args.use_shared_diff,
        "shared_diff_dim": args.shared_diff_dim,
        "use_residual_diff": args.use_residual_diff,
        "use_modality_prompt": args.use_modality_prompt,
        "use_reliability_gate": args.use_reliability_gate,
        "gate_use_diff": args.gate_use_diff,
        "use_diff_branch": args.use_diff_branch,
        "gamma_diff": args.gamma_diff,
        "use_mddm": args.use_mddm,
        "mddm_variant": args.mddm_variant,
        "mddm_gamma_init": args.mddm_gamma_init,
        "mddm_use_roi": args.mddm_use_roi,
        "use_inner_ring_context": args.use_inner_ring_context,
        "ring_kernel_size": args.ring_kernel_size,
        "irc_gamma_init": args.irc_gamma_init,
        "dg_gate_reduction": args.dg_gate_reduction,
        "dg_gamma_init": args.dg_gamma_init,
        "dg_gamma_mode": args.dg_gamma_mode,
        "dg_use_roi": args.dg_use_roi,
        "dg_large_beta": args.dg_large_beta,
        "dg_warmup_iters": args.dg_warmup_iters,
        "dg_gate_bias_init": args.dg_gate_bias_init,
        "use_damage_exist_head": args.use_damage_exist_head,
        "lambda_exist": args.lambda_exist,
        "exist_pos_weight": args.exist_pos_weight,
        "exist_loss_type": args.exist_loss_type,
        "exist_focal_gamma": args.exist_focal_gamma,
        "use_exist_logit_calib": args.use_exist_logit_calib,
        "exist_calib_alpha": args.exist_calib_alpha,
        "exist_calib_beta": args.exist_calib_beta,
        "exist_calib_warmup": args.exist_calib_warmup,
        "use_exist_feature_modulation": args.use_exist_feature_modulation,
        "exist_fm_gamma": args.exist_fm_gamma,
        "exist_fm_warmup": args.exist_fm_warmup,
        "use_hdc": args.use_hdc,
        "hdc_mode": args.hdc_mode,
        "hdc_alpha": args.hdc_alpha,
        "lambda_hdc": args.lambda_hdc,
        "lambda_severity": args.lambda_severity,
        "use_severity_head": args.use_severity_head,
        "use_severity_proto": args.use_severity_proto,
        "proto_k": args.proto_k,
        "proto_tau": args.proto_tau,
        "use_intact_consistency": args.use_intact_consistency,
        "lambda_ic": args.lambda_ic,
        "ic_dim": args.ic_dim,
        "ic_normalize": args.ic_normalize,
        "use_damage_margin": args.use_damage_margin,
        "lambda_margin": args.lambda_margin,
        "margin_damaged": args.margin_damaged,
        "margin_destroyed": args.margin_destroyed,
        "use_bdpcde": args.use_bdpcde,
        "bdpcde_use_damage_prior": args.bdpcde_use_damage_prior,
        "bdpcde_beta": args.bdpcde_beta,
        "bdpcde_gamma": args.bdpcde_gamma,
        "bdpcde_detach_prior": args.bdpcde_detach_prior,
        "bdpcde_dropout": args.bdpcde_dropout,
        "use_sfdg": args.use_sfdg,
        "sfdg_gamma": args.sfdg_gamma,
        "sfdg_detach_gate": args.sfdg_detach_gate,
        "return_dict": False,
        "use_pretrained_swin": False,
    }
    kwargs = filter_init_kwargs(model_cls, candidate_kwargs)
    return model_cls(**kwargs)


def extract_state_dict(checkpoint: Any) -> MutableMapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint does not contain a state_dict-like mapping: {type(checkpoint)}")

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        clean_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def migrate_legacy_state_dict_keys(
    model: torch.nn.Module,
    state_dict: MutableMapping[str, torch.Tensor],
) -> Tuple[MutableMapping[str, torch.Tensor], List[Tuple[str, str]]]:
    """Apply narrow, architecture-checked key migrations before strict loading.

    The first IA-MLA checkpoints named each StageDifferenceBlock fusion module
    ``relation``.  The current source calls the same module ``relation_fuse``.
    Only keys whose renamed destination exists in the current model are
    migrated; tensor shape compatibility is still enforced by strict loading.
    """

    if model.__class__.__name__ != "DamageFormer_S8_IA_MLA":
        return state_dict, []

    model_keys = set(model.state_dict().keys())
    migrated: MutableMapping[str, torch.Tensor] = OrderedDict()
    migrations: List[Tuple[str, str]] = []
    pattern = re.compile(r"^(hcdm\.stage_blocks\.\d+)\.relation\.(.+)$")

    for key, value in state_dict.items():
        match = pattern.match(str(key))
        new_key = (
            f"{match.group(1)}.relation_fuse.{match.group(2)}"
            if match is not None
            else str(key)
        )
        if (
            new_key != key
            and new_key in model_keys
            and key not in model_keys
        ):
            if new_key in state_dict or new_key in migrated:
                raise RuntimeError(
                    "Ambiguous IA-MLA checkpoint migration: both legacy and "
                    f"current keys resolve to {new_key!r}."
                )
            migrated[new_key] = value
            migrations.append((str(key), new_key))
        else:
            migrated[str(key)] = value
    return migrated, migrations


def load_checkpoint(
    model: torch.nn.Module,
    model_path: str,
    device: torch.device,
    strict: bool = False,
) -> Dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    state_dict, migrations = migrate_legacy_state_dict_keys(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded checkpoint: {model_path}")
    print(f"Strict checkpoint load: {strict}")
    print(f"Legacy key migrations: {len(migrations)}")
    if migrations:
        print("Migration examples:", migrations[:4])
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print("Missing key examples:", list(missing)[:10])
    if unexpected:
        print("Unexpected key examples:", list(unexpected)[:10])
    return {
        "load_missing_keys_count": len(missing),
        "load_unexpected_keys_count": len(unexpected),
        "load_missing_key_examples": list(missing)[:10],
        "load_unexpected_key_examples": list(unexpected)[:10],
        "load_key_migration_count": len(migrations),
        "load_key_migration_examples": migrations[:10],
    }


def normalize_name_item(name: Any) -> str:
    if isinstance(name, (list, tuple)):
        if not name:
            return ""
        return normalize_name_item(name[0])
    if torch.is_tensor(name):
        if name.numel() == 1:
            return str(name.item())
        return ""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="ignore")
    return str(name)


def names_to_list(names: Any, batch_size: int) -> List[str]:
    if isinstance(names, (list, tuple)):
        values = [normalize_name_item(item) for item in names]
        if len(values) == batch_size:
            return values
        if len(values) == 1 and batch_size > 1:
            return values * batch_size
        return values[:batch_size] + [""] * max(0, batch_size - len(values))
    value = normalize_name_item(names)
    return [value] * batch_size


def pick_from_dict(batch: Mapping[str, Any], keys: Sequence[str], required: bool = True) -> Any:
    lower_map = {str(key).lower(): key for key in batch.keys()}
    for key in keys:
        if key in batch:
            return batch[key]
        mapped = lower_map.get(key.lower())
        if mapped is not None:
            return batch[mapped]
    if required:
        raise KeyError(f"Cannot find any of {keys}. Available keys: {list(batch.keys())}")
    return None


def parse_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    if isinstance(batch, Mapping):
        pre = pick_from_dict(batch, ["pre_img", "pre_image", "pre_data", "pre", "image_pre", "img_pre"])
        post = pick_from_dict(batch, ["post_img", "post_image", "post_data", "post", "image_post", "img_post"])
        loc = pick_from_dict(batch, ["loc_label", "label_loc", "labels_loc", "building_label", "building_mask", "loc"])
        dam = pick_from_dict(batch, ["damage_label", "dam_label", "label_dam", "labels_clf", "label_cls", "damage"])
        names = pick_from_dict(batch, ["name", "names", "file_name", "filename", "id", "data_idx"], required=False)
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 5:
            pre, post, loc, dam, names = batch
        elif len(batch) == 6:
            pre, post, _unused, loc, dam, names = batch
        else:
            raise ValueError(f"Unsupported tuple/list batch length: {len(batch)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    pre = pre.to(device, non_blocking=True).float()
    post = post.to(device, non_blocking=True).float()
    loc = loc.to(device, non_blocking=True).long()
    dam = dam.to(device, non_blocking=True).long()
    if loc.dim() == 4 and loc.size(1) == 1:
        loc = loc.squeeze(1)
    if dam.dim() == 4 and dam.size(1) == 1:
        dam = dam.squeeze(1)
    return pre, post, loc, dam, names_to_list(names, pre.shape[0])


def parse_outputs(outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(outputs, Mapping):
        loc = outputs.get("output_loc")
        dam = outputs.get("output_dam")
        if loc is None or dam is None:
            raise KeyError(f"Dict output must contain output_loc and output_dam. Keys: {list(outputs.keys())}")
        return loc, dam
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        return outputs[0], outputs[1]
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def resize_logits(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def update_confusion(conf: np.ndarray, target: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    mask = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if not np.any(mask):
        return
    encoded = num_classes * target[mask].astype(np.int64) + pred[mask].astype(np.int64)
    conf += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def nan_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def update_binary_conf_from_damage(
    conf: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
) -> None:
    valid = (target != IGNORE_INDEX) & (target >= 0) & (target < 4)
    valid &= (pred >= 0) & (pred < 4)
    if not np.any(valid):
        return
    gt_building = np.isin(target[valid], [1, 2, 3]).astype(np.int64)
    pred_building = np.isin(pred[valid], [1, 2, 3]).astype(np.int64)
    update_confusion(conf, gt_building, pred_building, 2)


def update_loc_head_conf(
    conf: np.ndarray,
    damage_target: np.ndarray,
    loc_pred: np.ndarray,
) -> None:
    valid = (damage_target != IGNORE_INDEX) & (damage_target >= 0) & (damage_target < 4)
    valid &= (loc_pred >= 0) & (loc_pred < 2)
    if not np.any(valid):
        return
    gt_building = np.isin(damage_target[valid], [1, 2, 3]).astype(np.int64)
    pred_building = (loc_pred[valid] > 0).astype(np.int64)
    update_confusion(conf, gt_building, pred_building, 2)


def class_metrics_from_conf(conf: np.ndarray) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    for cls in range(conf.shape[0]):
        tp = float(conf[cls, cls])
        fp = float(conf[:, cls].sum() - tp)
        fn = float(conf[cls, :].sum() - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        iou = safe_div(tp, tp + fp + fn)
        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
    return metrics


def compute_loc_metrics(conf: np.ndarray) -> Dict[str, float]:
    m = class_metrics_from_conf(conf)[1]
    acc = safe_div(float(np.trace(conf)), float(conf.sum()))
    return {
        "loc_acc": acc,
        "loc_precision": m["precision"],
        "loc_recall": m["recall"],
        "loc_f1": m["f1"],
        "loc_iou": m["iou"],
    }


def compute_damage_metrics(conf: np.ndarray) -> Dict[str, Any]:
    per_class = class_metrics_from_conf(conf)
    valid = [1, 2, 3]
    result: Dict[str, Any] = {"per_class": per_class}
    for key in ("precision", "recall", "f1", "iou"):
        result[f"valid_mean_{key}"] = float(np.mean([per_class[cls][key] for cls in valid]))
    return result


def confusion_analysis(conf: np.ndarray) -> Dict[str, float]:
    gt_damaged = float(conf[2, :].sum())
    gt_destroyed = float(conf[3, :].sum())
    gt_building = float(conf[1:4, :].sum())

    valid_mask = conf[1:4, 1:4]
    total_valid_pred = float(valid_mask.sum())
    ordinal_sum = 0.0
    severe = 0.0
    for gt_cls in (1, 2, 3):
        for pred_cls in (1, 2, 3):
            count = float(conf[gt_cls, pred_cls])
            err = abs(gt_cls - pred_cls)
            ordinal_sum += err * count
            if err >= 2:
                severe += count

    return {
        "damaged_to_intact_rate": safe_div(float(conf[2, 1]), gt_damaged),
        "damaged_to_destroyed_rate": safe_div(float(conf[2, 3]), gt_damaged),
        "destroyed_to_intact_rate": safe_div(float(conf[3, 1]), gt_destroyed),
        "destroyed_to_damaged_rate": safe_div(float(conf[3, 2]), gt_destroyed),
        "pred_background_on_gt_building_rate": safe_div(float(conf[1:4, 0].sum()), gt_building),
        "ordinal_mae_on_valid_predictions": safe_div(ordinal_sum, total_valid_pred),
        "severe_ordinal_error_rate": safe_div(severe, total_valid_pred),
    }


def empty_stats() -> Dict[str, np.ndarray]:
    return {
        "loc_conf": np.zeros((2, 2), dtype=np.int64),
        "damage_conf": np.zeros((4, 4), dtype=np.int64),
    }


def empty_official_event_stats() -> Dict[str, Any]:
    return {
        "damage_conf": np.zeros((4, 4), dtype=np.int64),
        "loc_head_conf": np.zeros((2, 2), dtype=np.int64),
        "damage_building_conf": np.zeros((2, 2), dtype=np.int64),
        "num_images": 0,
        "num_pixels": 0,
        "num_valid_pixels": 0,
    }


def canonical_event_id(text: Any) -> str:
    value = normalize_name_item(text).strip()
    if not value:
        return ""
    fields = [field for field in re.split(r"[\\/]", value) if field]
    stem = fields[-1] if fields else value
    stem = re.sub(r"\.(tif|tiff|png|jpg|jpeg|npy|npz)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(pre_disaster|post_disaster|building_damage)$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


def event_from_name(name: str, data_list_entry: Optional[str], event_index: Optional[int]) -> str:
    if event_index is not None and data_list_entry:
        fields = [field for field in re.split(r"[\\/]", data_list_entry) if field]
        if 0 <= event_index < len(fields):
            return canonical_event_id(fields[event_index]) or "unknown"
        if -len(fields) <= event_index < 0:
            return canonical_event_id(fields[event_index]) or "unknown"

    for text in (data_list_entry or "", name):
        event = canonical_event_id(text)
        if event:
            return event
    return "unknown"


def fixed_match(texts: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    joined = " ".join(text or "" for text in texts)
    for candidate in candidates:
        if candidate in joined:
            return candidate
    return None


def summarize_stats(stats: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    loc = compute_loc_metrics(stats["loc_conf"])
    damage = compute_damage_metrics(stats["damage_conf"])
    analysis = confusion_analysis(stats["damage_conf"])
    valid_miou = damage["valid_mean_iou"]
    final_miou = 0.5 * loc["loc_iou"] + 0.5 * valid_miou
    return {
        **loc,
        "valid_mean_precision": damage["valid_mean_precision"],
        "valid_mean_recall": damage["valid_mean_recall"],
        "valid_mean_f1": damage["valid_mean_f1"],
        "valid_miou": valid_miou,
        "final_miou": final_miou,
        **analysis,
        "per_class": damage["per_class"],
    }


def summarize_paper_conf(conf: np.ndarray) -> Dict[str, Any]:
    per_class = class_metrics_from_conf(conf)
    ious = [per_class[cls]["iou"] for cls in range(4)]
    return {
        "pixel_acc": safe_div(float(np.trace(conf)), float(conf.sum())),
        "miou_4class": float(np.mean(ious)),
        "iou_background": per_class[0]["iou"],
        "iou_intact": per_class[1]["iou"],
        "iou_damaged": per_class[2]["iou"],
        "iou_destroyed": per_class[3]["iou"],
        "per_class": per_class,
    }


def official_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def official_per_class_from_conf(conf: np.ndarray) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    for cls in range(conf.shape[0]):
        tp = float(conf[cls, cls])
        fp = float(conf[:, cls].sum() - tp)
        fn = float(conf[cls, :].sum() - tp)
        support = float(conf[cls, :].sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, support)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        denom_iou = tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else float("nan")
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }
    return metrics


def summarize_official_conf(
    damage_conf: np.ndarray,
    loc_conf: Optional[np.ndarray] = None,
    damage_building_conf: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    per_class = official_per_class_from_conf(damage_conf)
    class_names = list(DAMAGE_CLASSES.values())
    ious = [per_class[cls]["iou"] for cls in range(4)]
    summary: Dict[str, Any] = {
        "background_iou": per_class[0]["iou"],
        "intact_iou": per_class[1]["iou"],
        "damaged_iou": per_class[2]["iou"],
        "destroyed_iou": per_class[3]["iou"],
        "official_style_4class_miou": official_nanmean(ious),
        "target_3class_miou": official_nanmean(ious[1:4]),
        "num_classes": 4,
        "class_names": class_names,
        "eval_protocol": "official",
        "per_class": per_class,
    }

    for cls, name in DAMAGE_CLASSES.items():
        summary[f"{name}_precision"] = per_class[cls]["precision"]
        summary[f"{name}_recall"] = per_class[cls]["recall"]
        summary[f"{name}_f1"] = per_class[cls]["f1"]

    if loc_conf is not None:
        loc_metrics = compute_loc_metrics(loc_conf)
        summary.update({
            "loc_iou": loc_metrics["loc_iou"],
            "loc_f1": loc_metrics["loc_f1"],
            "loc_precision": loc_metrics["loc_precision"],
            "loc_recall": loc_metrics["loc_recall"],
            "loc_head_iou": loc_metrics["loc_iou"],
            "loc_head_f1": loc_metrics["loc_f1"],
            "loc_head_precision": loc_metrics["loc_precision"],
            "loc_head_recall": loc_metrics["loc_recall"],
        })
    else:
        summary.update({
            "loc_iou": None,
            "loc_f1": None,
            "loc_precision": None,
            "loc_recall": None,
            "loc_head_iou": None,
            "loc_head_f1": None,
            "loc_head_precision": None,
            "loc_head_recall": None,
        })

    if damage_building_conf is not None:
        building_metrics = compute_loc_metrics(damage_building_conf)
        summary.update({
            "damage_building_iou": building_metrics["loc_iou"],
            "damage_building_f1": building_metrics["loc_f1"],
            "damage_building_precision": building_metrics["loc_precision"],
            "damage_building_recall": building_metrics["loc_recall"],
        })
    else:
        summary.update({
            "damage_building_iou": None,
            "damage_building_f1": None,
            "damage_building_precision": None,
            "damage_building_recall": None,
        })

    gt_damaged = float(damage_conf[2, :].sum())
    gt_destroyed = float(damage_conf[3, :].sum())
    summary.update({
        "damaged_to_background": safe_div(float(damage_conf[2, 0]), gt_damaged),
        "damaged_to_intact": safe_div(float(damage_conf[2, 1]), gt_damaged),
        "damaged_to_destroyed": safe_div(float(damage_conf[2, 3]), gt_damaged),
        "destroyed_to_background": safe_div(float(damage_conf[3, 0]), gt_destroyed),
        "destroyed_to_intact": safe_div(float(damage_conf[3, 1]), gt_destroyed),
        "destroyed_to_damaged": safe_div(float(damage_conf[3, 2]), gt_destroyed),
    })
    return summary


def summarize_official_event(
    event: str,
    stats: Mapping[str, Any],
    overall_summary: Mapping[str, Any],
    overall_num_building_pixels: float,
    overall_num_destroyed_pixels: float,
) -> Dict[str, Any]:
    damage_conf = np.asarray(stats["damage_conf"])
    summary = summarize_official_conf(
        damage_conf,
        np.asarray(stats["loc_head_conf"]),
        np.asarray(stats["damage_building_conf"]),
    )
    supports = damage_conf.sum(axis=1).astype(np.int64)
    num_background, num_intact, num_damaged, num_destroyed = supports.tolist()
    num_building = int(num_intact + num_damaged + num_destroyed)

    if num_damaged == 0:
        summary["damaged_iou"] = float("nan")
        summary["damaged_recall"] = float("nan")
        summary["damaged_to_background"] = float("nan")
        summary["damaged_to_intact"] = float("nan")
        summary["damaged_to_destroyed"] = float("nan")
    if num_destroyed == 0:
        summary["destroyed_iou"] = float("nan")
        summary["destroyed_recall"] = float("nan")
        summary["destroyed_to_background"] = float("nan")
        summary["destroyed_to_intact"] = float("nan")
        summary["destroyed_to_damaged"] = float("nan")

    event_ious = [
        summary["background_iou"],
        summary["intact_iou"],
        summary["damaged_iou"],
        summary["destroyed_iou"],
    ]
    summary["official_style_4class_miou"] = official_nanmean(event_ious)
    summary["target_3class_miou"] = official_nanmean(event_ious[1:4])

    destroyed_fp_counts = [
        int(damage_conf[0, 3]),
        int(damage_conf[1, 3]),
        int(damage_conf[2, 3]),
    ]
    destroyed_fp_total = sum(destroyed_fp_counts)
    summary.update({
        "destroyed_fp_from_background_count": destroyed_fp_counts[0],
        "destroyed_fp_from_intact_count": destroyed_fp_counts[1],
        "destroyed_fp_from_damaged_count": destroyed_fp_counts[2],
        "destroyed_fp_from_background_share": nan_div(destroyed_fp_counts[0], destroyed_fp_total),
        "destroyed_fp_from_intact_share": nan_div(destroyed_fp_counts[1], destroyed_fp_total),
        "destroyed_fp_from_damaged_share": nan_div(destroyed_fp_counts[2], destroyed_fp_total),
        "intact_to_destroyed": nan_div(float(damage_conf[1, 3]), float(damage_conf[1, :].sum())),
    })

    building_ratio = safe_div(float(num_building), overall_num_building_pixels)
    destroyed_ratio = safe_div(float(num_destroyed), overall_num_destroyed_pixels)
    event_target_miou = summary["target_3class_miou"]
    overall_target_miou = float(overall_summary["target_3class_miou"])
    impact_score = (
        building_ratio * max(0.0, overall_target_miou - float(event_target_miou))
        if event_target_miou == event_target_miou
        else 0.0
    )
    event_destroyed_iou = summary["destroyed_iou"]
    overall_destroyed_iou = float(overall_summary["destroyed_iou"])
    destroyed_impact_score = 0.0
    if num_destroyed > 0 and event_destroyed_iou == event_destroyed_iou:
        destroyed_impact_score = destroyed_ratio * max(
            0.0,
            overall_destroyed_iou - float(event_destroyed_iou),
        )

    row: Dict[str, Any] = {
        "event": event,
        "num_images": int(stats["num_images"]),
        "num_pixels": int(stats["num_pixels"]),
        "num_valid_pixels": int(stats["num_valid_pixels"]),
        "num_building_pixels": num_building,
        "num_background_pixels": int(num_background),
        "num_intact_pixels": int(num_intact),
        "num_damaged_pixels": int(num_damaged),
        "num_destroyed_pixels": int(num_destroyed),
        **{key: value for key, value in summary.items() if key not in {"per_class", "class_names"}},
        "event_building_pixel_ratio": building_ratio,
        "event_destroyed_pixel_ratio": destroyed_ratio,
        "impact_score": impact_score,
        "destroyed_impact_score": destroyed_impact_score,
    }
    row.update(diagnose_destroyed_event(row, overall_summary))
    return row


def diagnose_destroyed_event(
    row: Mapping[str, Any],
    overall: Mapping[str, Any],
) -> Dict[str, str]:
    num_destroyed = int(row["num_destroyed_pixels"])
    support_warning = (
        "LOW_DESTROYED_SUPPORT"
        if 0 < num_destroyed < LOW_DESTROYED_SUPPORT_THRESHOLD
        else ""
    )
    if num_destroyed == 0:
        return {
            "diagnosis_code": "NO_DESTROYED_GT",
            "diagnosis_text": "该 event 没有 GT destroyed 像素，不对 destroyed 漏检方向作结论。",
            "recommended_direction": "Collect/inspect destroyed support before model diagnosis",
            "support_warning": "NO_DESTROYED_SUPPORT",
        }

    gap = EVENT_DIAGNOSIS_GAP
    to_intact_high = float(row["destroyed_to_intact"]) >= float(overall["destroyed_to_intact"]) + gap
    to_background_high = float(row["destroyed_to_background"]) >= float(overall["destroyed_to_background"]) + gap
    to_damaged_high = float(row["destroyed_to_damaged"]) >= float(overall["destroyed_to_damaged"]) + gap
    recall_low = float(row["destroyed_recall"]) <= float(overall["destroyed_recall"]) - gap
    recall_not_low = float(row["destroyed_recall"]) >= float(overall["destroyed_recall"]) - gap
    precision_low = float(row["destroyed_precision"]) <= float(overall["destroyed_precision"]) - gap
    building_not_low = float(row["damage_building_iou"]) >= float(overall["damage_building_iou"]) - gap
    building_low = float(row["damage_building_iou"]) <= float(overall["damage_building_iou"]) - gap
    to_background_low = float(row["destroyed_to_background"]) <= float(overall["destroyed_to_background"]) - gap

    if to_background_high and building_low:
        code = "DESTROYED_TO_BACKGROUND_LOC_FAIL"
        text = "destroyed 建筑在最终四分类中被漏成 background，属于建筑定位或 ROI 覆盖问题。"
        direction = "ROI strategy, loc head, building-aware fusion"
    elif to_intact_high and recall_low and building_not_low:
        code = "DESTROYED_TO_INTACT"
        text = "建筑区域基本找到，但损毁头将 destroyed 判成 intact，属于完好/损毁混淆。"
        direction = "A4-DG-beta, DamageExistHead, A4-HDC"
    elif to_damaged_high and to_background_low and building_not_low:
        code = "DESTROYED_TO_DAMAGED_SEVERITY_CONFUSION"
        text = "模型知道该区域是损毁建筑，但分不清 damaged 和 destroyed。"
        direction = "Severity Head, HDC, large-scale MDDM, ordinal loss"
    elif precision_low and recall_not_low:
        code = "DESTROYED_OVER_PREDICTION"
        text = "模型过度预测 destroyed，false positive 较多。"
        direction = "decision boundary calibration, class weight, focal loss, intact consistency"
    else:
        code = "MIXED_OR_NO_DOMINANT_ERROR"
        text = "未发现单一 destroyed 错误方向显著偏离 overall，需结合 confusion 和 support 检查。"
        direction = "Inspect event confusion, support, precision/recall and qualitative samples"

    return {
        "diagnosis_code": code,
        "diagnosis_text": text,
        "recommended_direction": direction,
        "support_warning": support_warning,
    }


def write_confusion_csv(path: str, conf: np.ndarray, labels: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *conf[idx].tolist()])


def write_per_class_csv(path: str, per_class: Mapping[int, Mapping[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "iou"])
        for cls, name in DAMAGE_CLASSES.items():
            row = per_class[cls]
            writer.writerow([cls, name, row["precision"], row["recall"], row["f1"], row["iou"]])


def write_official_per_class_csv(path: str, per_class: Mapping[int, Mapping[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "iou", "tp", "fp", "fn", "support"])
        for cls, name in DAMAGE_CLASSES.items():
            row = per_class[cls]
            writer.writerow([
                cls,
                name,
                row["precision"],
                row["recall"],
                row["f1"],
                row["iou"],
                row["tp"],
                row["fp"],
                row["fn"],
                row["support"],
            ])


OFFICIAL_EVENT_FIELDS = [
    "event",
    "num_images",
    "num_pixels",
    "num_valid_pixels",
    "num_building_pixels",
    "num_background_pixels",
    "num_intact_pixels",
    "num_damaged_pixels",
    "num_destroyed_pixels",
    "background_iou",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_precision",
    "damaged_recall",
    "destroyed_precision",
    "destroyed_recall",
    "official_style_4class_miou",
    "target_3class_miou",
    "damaged_to_background",
    "damaged_to_intact",
    "damaged_to_destroyed",
    "destroyed_to_background",
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "loc_iou",
    "loc_f1",
    "loc_precision",
    "loc_recall",
    "loc_head_iou",
    "loc_head_f1",
    "loc_head_precision",
    "loc_head_recall",
    "damage_building_iou",
    "damage_building_f1",
    "damage_building_precision",
    "damage_building_recall",
    "event_building_pixel_ratio",
    "event_destroyed_pixel_ratio",
    "impact_score",
    "destroyed_impact_score",
    "diagnosis_code",
    "diagnosis_text",
    "recommended_direction",
    "support_warning",
]

DESTROYED_DIAGNOSIS_FIELDS = [
    "event",
    "num_destroyed_pixels",
    "destroyed_iou",
    "destroyed_precision",
    "destroyed_recall",
    "destroyed_to_background",
    "destroyed_to_intact",
    "destroyed_to_damaged",
    "destroyed_fp_from_background_count",
    "destroyed_fp_from_intact_count",
    "destroyed_fp_from_damaged_count",
    "destroyed_fp_from_background_share",
    "destroyed_fp_from_intact_share",
    "destroyed_fp_from_damaged_share",
    "intact_to_destroyed",
    "damaged_to_destroyed",
    "damage_building_iou",
    "destroyed_impact_score",
    "diagnosis_code",
    "diagnosis_text",
    "recommended_direction",
    "support_warning",
]


def write_official_event_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OFFICIAL_EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OFFICIAL_EVENT_FIELDS})

    miou_values = [
        float(row["official_style_4class_miou"])
        for row in rows
        if row["official_style_4class_miou"] == row["official_style_4class_miou"]
    ]
    valid_rows = [
        row
        for row in rows
        if row["official_style_4class_miou"] == row["official_style_4class_miou"]
    ]
    worst = min(valid_rows, key=lambda r: r["official_style_4class_miou"]) if valid_rows else None
    return {
        "event_mean_official_style_4class_miou": float(np.mean(miou_values)) if miou_values else float("nan"),
        "event_worst": worst,
    }


def write_event_confusion_csv(
    path: str,
    event_stats: Mapping[str, Mapping[str, Any]],
) -> None:
    conf_fields = [f"conf_{gt}{pred}" for gt in range(4) for pred in range(4)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event", *conf_fields])
        writer.writeheader()
        for event in sorted(event_stats.keys()):
            conf = np.asarray(event_stats[event]["damage_conf"])
            row: Dict[str, Any] = {"event": event}
            for gt in range(4):
                for pred in range(4):
                    row[f"conf_{gt}{pred}"] = int(conf[gt, pred])
            writer.writerow(row)


def write_event_destroyed_diagnosis_csv(
    path: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DESTROYED_DIAGNOSIS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DESTROYED_DIAGNOSIS_FIELDS})


def analysis_config() -> Dict[str, Any]:
    return {
        "event_id_rule": {
            "source_priority": ["data_list_entry", "sample_name"],
            "strip_suffixes": ["_pre_disaster", "_post_disaster", "_building_damage"],
            "strip_final_image_id_regex": r"_\d+$",
            "does_not_use_parent_event_directory": True,
        },
        "ignore_index": IGNORE_INDEX,
        "building_class_ids": [1, 2, 3],
        "confusion_direction": "rows=GT, columns=prediction",
        "diagnosis": {
            "comparison_method": "overall_gap",
            "gap": EVENT_DIAGNOSIS_GAP,
            "high_rule": "event_metric >= overall_metric + gap",
            "low_rule": "event_metric <= overall_metric - gap",
            "not_low_rule": "event_metric >= overall_metric - gap",
            "low_destroyed_support_threshold": LOW_DESTROYED_SUPPORT_THRESHOLD,
            "priority": [
                "DESTROYED_TO_BACKGROUND_LOC_FAIL",
                "DESTROYED_TO_INTACT",
                "DESTROYED_TO_DAMAGED_SEVERITY_CONFUSION",
                "DESTROYED_OVER_PREDICTION",
                "MIXED_OR_NO_DOMINANT_ERROR",
            ],
        },
        "impact_score": (
            "event_building_pixel_ratio * "
            "max(0, overall_target_3class_miou - event_target_3class_miou)"
        ),
        "destroyed_impact_score": (
            "event_destroyed_pixel_ratio * "
            "max(0, overall_destroyed_iou - event_destroyed_iou)"
        ),
        "empty_gt_class_rule": {
            "damaged": "GT support 0 -> damaged IoU/recall/error-flow metrics are NaN",
            "destroyed": "GT support 0 -> destroyed IoU/recall/error-flow metrics are NaN",
            "destroyed_precision": "retained because it is conditioned on predicted destroyed",
        },
    }


def write_official_summary_text(path: str, summary: Mapping[str, Any], event_info: Mapping[str, Any]) -> None:
    lines = [
        "Official-Style 4-Class Evaluation",
        "",
        f"background_iou: {summary['background_iou']:.6f}",
        f"intact_iou: {summary['intact_iou']:.6f}",
        f"damaged_iou: {summary['damaged_iou']:.6f}",
        f"destroyed_iou: {summary['destroyed_iou']:.6f}",
        f"official_style_4class_miou: {summary['official_style_4class_miou']:.6f}",
        f"target_3class_miou: {summary['target_3class_miou']:.6f}",
        "",
        f"background_precision: {summary['background_precision']:.6f}",
        f"background_recall: {summary['background_recall']:.6f}",
        f"background_f1: {summary['background_f1']:.6f}",
        f"intact_precision: {summary['intact_precision']:.6f}",
        f"intact_recall: {summary['intact_recall']:.6f}",
        f"intact_f1: {summary['intact_f1']:.6f}",
        f"damaged_precision: {summary['damaged_precision']:.6f}",
        f"damaged_recall: {summary['damaged_recall']:.6f}",
        f"damaged_f1: {summary['damaged_f1']:.6f}",
        f"destroyed_precision: {summary['destroyed_precision']:.6f}",
        f"destroyed_recall: {summary['destroyed_recall']:.6f}",
        f"destroyed_f1: {summary['destroyed_f1']:.6f}",
        "",
        f"loc_iou: {summary.get('loc_iou')}",
        f"loc_f1: {summary.get('loc_f1')}",
        f"loc_precision: {summary.get('loc_precision')}",
        f"loc_recall: {summary.get('loc_recall')}",
        f"loc_head_iou: {summary.get('loc_head_iou')}",
        f"loc_head_f1: {summary.get('loc_head_f1')}",
        f"damage_building_iou: {summary.get('damage_building_iou')}",
        f"damage_building_f1: {summary.get('damage_building_f1')}",
        "",
        f"damaged_to_background: {summary['damaged_to_background']:.6f}",
        f"damaged_to_intact: {summary['damaged_to_intact']:.6f}",
        f"damaged_to_destroyed: {summary['damaged_to_destroyed']:.6f}",
        f"destroyed_to_background: {summary['destroyed_to_background']:.6f}",
        f"destroyed_to_intact: {summary['destroyed_to_intact']:.6f}",
        f"destroyed_to_damaged: {summary['destroyed_to_damaged']:.6f}",
        "",
        f"event_mean_official_style_4class_miou: {event_info.get('event_mean_official_style_4class_miou')}",
        f"event_worst: {event_info.get('event_worst')}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_paper_summary_text(path: str, args: argparse.Namespace, summary: Mapping[str, Any]) -> None:
    per_class = summary["per_class"]
    lines = [
        "Paper-Style Evaluation Summary",
        "",
        f"model_type: {args.model_type}",
        f"model_path: {args.model_path}",
        f"dataset_path: {args.dataset_path}",
        f"data_list_path: {args.data_list_path}",
        f"image_size: {args.image_size}",
        "",
        f"Pixel Accuracy: {summary['pixel_acc']:.6f}",
        f"Mean IoU over 4 classes: {summary['miou_4class']:.6f}",
        f"IoU background: {summary['iou_background']:.6f}",
        f"IoU intact: {summary['iou_intact']:.6f}",
        f"IoU damaged: {summary['iou_damaged']:.6f}",
        f"IoU destroyed: {summary['iou_destroyed']:.6f}",
        "",
        "Per-class metrics:",
    ]
    for cls, name in DAMAGE_CLASSES.items():
        row = per_class[cls]
        lines.append(
            f"{cls} {name}: Precision={row['precision']:.6f}, Recall={row['recall']:.6f}, "
            f"F1={row['f1']:.6f}, IoU={row['iou']:.6f}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_paper_event_csv(path: str, named_stats: Mapping[str, np.ndarray], name_field: str = "event") -> None:
    fields = [
        name_field,
        "pixel_acc",
        "miou_4class",
        "iou_background",
        "iou_intact",
        "iou_damaged",
        "iou_destroyed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in named_stats.keys():
            summary = summarize_paper_conf(named_stats[name])
            writer.writerow(
                {
                    name_field: name,
                    "pixel_acc": summary["pixel_acc"],
                    "miou_4class": summary["miou_4class"],
                    "iou_background": summary["iou_background"],
                    "iou_intact": summary["iou_intact"],
                    "iou_damaged": summary["iou_damaged"],
                    "iou_destroyed": summary["iou_destroyed"],
                }
            )


def write_event_csv(path: str, event_stats: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    fields = [
        "event",
        "loc_f1",
        "loc_iou",
        "valid_mean_f1",
        "valid_miou",
        "final_miou",
        "intact_f1",
        "intact_iou",
        "damaged_f1",
        "damaged_iou",
        "destroyed_f1",
        "destroyed_iou",
        "damaged_to_intact_rate",
        "damaged_to_destroyed_rate",
        "destroyed_to_damaged_rate",
        "pred_background_on_gt_building_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in sorted(event_stats.keys()):
            summary = summarize_stats(event_stats[event])
            per_class = summary["per_class"]
            writer.writerow(
                {
                    "event": event,
                    "loc_f1": summary["loc_f1"],
                    "loc_iou": summary["loc_iou"],
                    "valid_mean_f1": summary["valid_mean_f1"],
                    "valid_miou": summary["valid_miou"],
                    "final_miou": summary["final_miou"],
                    "intact_f1": per_class[1]["f1"],
                    "intact_iou": per_class[1]["iou"],
                    "damaged_f1": per_class[2]["f1"],
                    "damaged_iou": per_class[2]["iou"],
                    "destroyed_f1": per_class[3]["f1"],
                    "destroyed_iou": per_class[3]["iou"],
                    "damaged_to_intact_rate": summary["damaged_to_intact_rate"],
                    "damaged_to_destroyed_rate": summary["damaged_to_destroyed_rate"],
                    "destroyed_to_damaged_rate": summary["destroyed_to_damaged_rate"],
                    "pred_background_on_gt_building_rate": summary["pred_background_on_gt_building_rate"],
                }
            )


def write_summary_text(path: str, args: argparse.Namespace, summary: Mapping[str, Any]) -> None:
    per_class = summary["per_class"]
    lines = [
        "Unified Evaluation Summary",
        "",
        f"model_type: {args.model_type}",
        f"model_path: {args.model_path}",
        f"dataset_path: {args.dataset_path}",
        f"data_list_path: {args.data_list_path}",
        "",
        "Loc metrics:",
        f"Loc Acc: {summary['loc_acc']:.6f}",
        f"Loc Precision: {summary['loc_precision']:.6f}",
        f"Loc Recall: {summary['loc_recall']:.6f}",
        f"Loc F1: {summary['loc_f1']:.6f}",
        f"Loc IoU: {summary['loc_iou']:.6f}",
        "",
        "Damage metrics on GT building region:",
    ]
    for cls, name in DAMAGE_CLASSES.items():
        row = per_class[cls]
        lines.append(
            f"{cls} {name}: Precision={row['precision']:.6f}, Recall={row['recall']:.6f}, "
            f"F1={row['f1']:.6f}, IoU={row['iou']:.6f}"
        )
    lines.extend(
        [
            f"Valid Mean Precision: {summary['valid_mean_precision']:.6f}",
            f"Valid Mean Recall: {summary['valid_mean_recall']:.6f}",
            f"Valid Mean F1: {summary['valid_mean_f1']:.6f}",
            f"Valid mIoU: {summary['valid_miou']:.6f}",
            f"Final mIoU: {summary['final_miou']:.6f}",
            "",
            "Confusion analysis:",
            f"damaged -> intact rate: {summary['damaged_to_intact_rate']:.6f}",
            f"damaged -> destroyed rate: {summary['damaged_to_destroyed_rate']:.6f}",
            f"destroyed -> intact rate: {summary['destroyed_to_intact_rate']:.6f}",
            f"destroyed -> damaged rate: {summary['destroyed_to_damaged_rate']:.6f}",
            f"pred background on GT building rate: {summary['pred_background_on_gt_building_rate']:.6f}",
            f"ordinal MAE on valid predictions: {summary['ordinal_mae_on_valid_predictions']:.6f}",
            f"severe ordinal error rate: {summary['severe_ordinal_error_rate']:.6f}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def evaluate_research(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list = read_list(args.data_list_path)
    DatasetClass = import_dataset_class()
    dataset = DatasetClass(args.dataset_path, data_list, args.image_size, None, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(args)
    load_checkpoint(model, args.model_path, device, strict=args.strict_checkpoint)
    model.to(device)
    model.eval()

    overall = empty_stats()
    event_stats: Dict[str, Dict[str, np.ndarray]] = {}
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if args.num_batches is not None and batch_idx >= args.num_batches:
                break

            pre, post, loc_label, dam_label, names = parse_batch(batch, device)
            loc_logits, dam_logits = parse_outputs(model(pre, post))
            loc_logits = resize_logits(loc_logits, loc_label)
            dam_logits = resize_logits(dam_logits, dam_label)

            loc_pred = torch.argmax(loc_logits, dim=1)
            dam_pred = torch.argmax(dam_logits, dim=1)

            loc_np = loc_label.detach().cpu().numpy()
            dam_np = dam_label.detach().cpu().numpy()
            loc_pred_np = loc_pred.detach().cpu().numpy()
            dam_pred_np = dam_pred.detach().cpu().numpy()

            update_loc_head_conf(overall["loc_conf"], dam_np.reshape(-1), loc_pred_np.reshape(-1))
            building_mask = np.isin(dam_np, [1, 2, 3])
            update_confusion(overall["damage_conf"], dam_np[building_mask], dam_pred_np[building_mask], 4)

            batch_size = pre.shape[0]
            for i in range(batch_size):
                list_index = sample_offset + i
                data_entry = data_list[list_index] if list_index < len(data_list) else None
                event = event_from_name(names[i] if i < len(names) else "", data_entry, args.event_index)
                if event not in event_stats:
                    event_stats[event] = empty_stats()
                update_loc_head_conf(
                    event_stats[event]["loc_conf"],
                    dam_np[i].reshape(-1),
                    loc_pred_np[i].reshape(-1),
                )
                mask_i = building_mask[i]
                update_confusion(event_stats[event]["damage_conf"], dam_np[i][mask_i], dam_pred_np[i][mask_i], 4)
            sample_offset += batch_size

    summary = summarize_stats(overall)
    summary_for_json = {
        key: value for key, value in summary.items() if key != "per_class"
    }
    summary_for_json["per_class"] = summary["per_class"]
    summary_for_json["loc_confusion"] = overall["loc_conf"]
    summary_for_json["damage_confusion_valid_gt_building"] = overall["damage_conf"]

    write_summary_text(os.path.join(args.save_dir, "summary.txt"), args, summary)
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready(summary_for_json), f, indent=2, ensure_ascii=False)
    write_per_class_csv(os.path.join(args.save_dir, "per_class_metrics.csv"), summary["per_class"])
    write_event_csv(os.path.join(args.save_dir, "event_metrics.csv"), event_stats)
    write_confusion_csv(os.path.join(args.save_dir, "confusion_loc.csv"), overall["loc_conf"], ["background", "building"])
    write_confusion_csv(
        os.path.join(args.save_dir, "confusion_damage_valid_gt_building.csv"),
        overall["damage_conf"],
        list(DAMAGE_CLASSES.values()),
    )

    print(f"Saved research evaluation files to: {args.save_dir}")
    print(f"Final mIoU: {summary['final_miou']:.6f}")
    return summary


def evaluate_paper(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list = read_list(args.data_list_path)
    DatasetClass = import_dataset_class()
    dataset = DatasetClass(args.dataset_path, data_list, args.image_size, None, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(args)
    load_checkpoint(model, args.model_path, device, strict=args.strict_checkpoint)
    model.to(device)
    model.eval()

    overall_conf = np.zeros((4, 4), dtype=np.int64)
    event_confs = OrderedDict((event, np.zeros((4, 4), dtype=np.int64)) for event in EVENT_NAMES)
    type_confs = OrderedDict((disaster_type, np.zeros((4, 4), dtype=np.int64)) for disaster_type in DISASTER_TYPES)
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating paper protocol")):
            if args.num_batches is not None and batch_idx >= args.num_batches:
                break

            pre, post, _loc_label, dam_label, names = parse_batch(batch, device)
            _loc_logits, dam_logits = parse_outputs(model(pre, post))
            dam_logits = resize_logits(dam_logits, dam_label)
            dam_pred = torch.argmax(dam_logits, dim=1)

            dam_np = dam_label.detach().cpu().numpy()
            dam_pred_np = dam_pred.detach().cpu().numpy()
            update_confusion(overall_conf, dam_np.reshape(-1), dam_pred_np.reshape(-1), 4)

            batch_size = pre.shape[0]
            for i in range(batch_size):
                list_index = sample_offset + i
                data_entry = data_list[list_index] if list_index < len(data_list) else ""
                file_name = names[i] if i < len(names) else ""
                texts = [file_name, data_entry]

                event = fixed_match(texts, EVENT_NAMES)
                if event is not None:
                    update_confusion(event_confs[event], dam_np[i].reshape(-1), dam_pred_np[i].reshape(-1), 4)

                disaster_type = fixed_match(texts, DISASTER_TYPES)
                if disaster_type is not None:
                    update_confusion(type_confs[disaster_type], dam_np[i].reshape(-1), dam_pred_np[i].reshape(-1), 4)

            sample_offset += batch_size

    summary = summarize_paper_conf(overall_conf)
    summary_for_json = {key: value for key, value in summary.items() if key != "per_class"}
    summary_for_json["per_class"] = summary["per_class"]
    summary_for_json["damage_confusion_all_pixels"] = overall_conf

    write_paper_summary_text(os.path.join(args.save_dir, "paper_summary.txt"), args, summary)
    with open(os.path.join(args.save_dir, "paper_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready(summary_for_json), f, indent=2, ensure_ascii=False)
    write_per_class_csv(os.path.join(args.save_dir, "paper_per_class_metrics.csv"), summary["per_class"])
    write_confusion_csv(
        os.path.join(args.save_dir, "paper_confusion_damage_all_pixels.csv"),
        overall_conf,
        list(DAMAGE_CLASSES.values()),
    )
    write_paper_event_csv(os.path.join(args.save_dir, "paper_event_metrics.csv"), event_confs, "event")
    write_paper_event_csv(os.path.join(args.save_dir, "paper_disaster_type_metrics.csv"), type_confs, "disaster_type")

    print(f"Saved paper evaluation files to: {args.save_dir}")
    print(f"Paper Pixel Accuracy: {summary['pixel_acc']:.6f}")
    print(f"Paper 4-class mIoU: {summary['miou_4class']:.6f}")
    return summary


def evaluate_official(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list = read_list(args.data_list_path)
    DatasetClass = import_dataset_class()
    dataset = DatasetClass(args.dataset_path, data_list, args.image_size, None, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(args)
    load_info = load_checkpoint(
        model,
        args.model_path,
        device,
        strict=args.strict_checkpoint,
    )
    model.to(device)
    model.eval()

    loc_conf = np.zeros((2, 2), dtype=np.int64)
    damage_building_conf = np.zeros((2, 2), dtype=np.int64)
    damage_conf = np.zeros((4, 4), dtype=np.int64)
    event_stats: Dict[str, Dict[str, Any]] = {}
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating official protocol")):
            if args.num_batches is not None and batch_idx >= args.num_batches:
                break

            pre, post, loc_label, dam_label, names = parse_batch(batch, device)
            loc_logits, dam_logits = parse_outputs(model(pre, post))
            loc_logits = resize_logits(loc_logits, loc_label)
            dam_logits = resize_logits(dam_logits, dam_label)

            loc_pred = torch.argmax(loc_logits, dim=1)
            dam_pred = torch.argmax(dam_logits, dim=1)

            dam_np = dam_label.detach().cpu().numpy()
            loc_pred_np = loc_pred.detach().cpu().numpy()
            dam_pred_np = dam_pred.detach().cpu().numpy()

            update_loc_head_conf(loc_conf, dam_np.reshape(-1), loc_pred_np.reshape(-1))
            update_binary_conf_from_damage(
                damage_building_conf,
                dam_np.reshape(-1),
                dam_pred_np.reshape(-1),
            )
            update_confusion(damage_conf, dam_np.reshape(-1), dam_pred_np.reshape(-1), 4)

            batch_size = pre.shape[0]
            for i in range(batch_size):
                list_index = sample_offset + i
                data_entry = data_list[list_index] if list_index < len(data_list) else None
                event = event_from_name(names[i] if i < len(names) else "", data_entry, args.event_index)
                if event not in event_stats:
                    event_stats[event] = empty_official_event_stats()
                stats = event_stats[event]
                target_i = dam_np[i].reshape(-1)
                damage_pred_i = dam_pred_np[i].reshape(-1)
                loc_pred_i = loc_pred_np[i].reshape(-1)
                valid_i = (target_i != IGNORE_INDEX) & (target_i >= 0) & (target_i < 4)
                stats["num_images"] += 1
                stats["num_pixels"] += int(target_i.size)
                stats["num_valid_pixels"] += int(valid_i.sum())
                update_confusion(stats["damage_conf"], target_i, damage_pred_i, 4)
                update_loc_head_conf(stats["loc_head_conf"], target_i, loc_pred_i)
                update_binary_conf_from_damage(
                    stats["damage_building_conf"],
                    target_i,
                    damage_pred_i,
                )
            sample_offset += batch_size

    summary = summarize_official_conf(damage_conf, loc_conf, damage_building_conf)
    overall_num_building_pixels = float(damage_conf[1:4, :].sum())
    overall_num_destroyed_pixels = float(damage_conf[3, :].sum())
    summary.update({
        "num_valid_pixels": int(damage_conf.sum()),
        "num_background_pixels": int(damage_conf[0, :].sum()),
        "num_intact_pixels": int(damage_conf[1, :].sum()),
        "num_damaged_pixels": int(damage_conf[2, :].sum()),
        "num_destroyed_pixels": int(overall_num_destroyed_pixels),
        "num_building_pixels": int(overall_num_building_pixels),
        **load_info,
    })
    event_rows = [
        summarize_official_event(
            event,
            event_stats[event],
            summary,
            overall_num_building_pixels,
            overall_num_destroyed_pixels,
        )
        for event in sorted(event_stats.keys())
    ]
    event_info = write_official_event_csv(
        os.path.join(args.save_dir, "event_metrics_official.csv"),
        event_rows,
    )
    summary["event_mean_official_style_4class_miou"] = event_info.get("event_mean_official_style_4class_miou")
    summary["event_wise_miou"] = event_info.get("event_mean_official_style_4class_miou")
    summary["event_worst"] = event_info.get("event_worst")
    summary["confusion_4class"] = damage_conf
    summary["loc_head_confusion"] = loc_conf
    summary["damage_building_confusion"] = damage_building_conf

    with open(os.path.join(args.save_dir, "summary_official.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready({k: v for k, v in summary.items() if k != "per_class"}), f, indent=2, ensure_ascii=False)
    write_official_per_class_csv(os.path.join(args.save_dir, "per_class_metrics_official.csv"), summary["per_class"])
    write_confusion_csv(
        os.path.join(args.save_dir, "confusion_4class_official.csv"),
        damage_conf,
        list(DAMAGE_CLASSES.values()),
    )
    write_event_confusion_csv(
        os.path.join(args.save_dir, "event_confusion_4class.csv"),
        event_stats,
    )
    write_event_destroyed_diagnosis_csv(
        os.path.join(args.save_dir, "event_destroyed_diagnosis.csv"),
        event_rows,
    )
    with open(os.path.join(args.save_dir, "analysis_config.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_config(), f, indent=2, ensure_ascii=False)
    write_official_summary_text(os.path.join(args.save_dir, "official_metrics.txt"), summary, event_info)

    print(f"Saved official evaluation files to: {args.save_dir}")
    print(f"Official 4-class mIoU: {summary['official_style_4class_miou']:.6f}")
    return summary


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.eval_protocol == "paper":
        return evaluate_paper(args)
    if args.eval_protocol == "official":
        return evaluate_official(args)
    return evaluate_research(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified evaluator for DamageFormer family models.")
    parser.add_argument(
        "--eval_protocol",
        type=str,
        default="official",
        choices=["paper", "research", "official"],
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "DamageFormer",
            "DamageFormer_BGCA",
            "DamageFormer_S8_IA_MLA",
            "DamageFormer_S8_MICB_DSMIL",
        ],
        required=True,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--strict_checkpoint",
        action="store_true",
        help=(
            "Require an exact state-dict match. Recommended for final model "
            "comparison so an incompatible architecture cannot be evaluated silently."
        ),
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--use_lovasz", action="store_true", help="Training-only compatibility flag.")
    parser.add_argument("--roi_mode", default="none")
    parser.add_argument("--roi_alpha", type=float, default=0.1)
    parser.add_argument("--roi_dilate", type=int, default=0)
    parser.add_argument("--roi_fusion", choices=["multiplicative", "residual"], default="multiplicative")
    parser.add_argument("--roi_beta", type=float, default=1.0)
    str2bool_flag(parser, "detach_mask", True)
    parser.add_argument("--use_align", action="store_true")
    parser.add_argument("--align_type", default="standard")
    parser.add_argument("--use_shared_diff", action="store_true")
    parser.add_argument("--shared_diff_dim", type=int, default=256)
    str2bool_flag(parser, "use_residual_diff", True)
    parser.add_argument("--use_modality_prompt", action="store_true")
    parser.add_argument("--use_reliability_gate", action="store_true")
    str2bool_flag(parser, "gate_use_diff", True)
    parser.add_argument("--use_diff_branch", action="store_true")
    parser.add_argument("--gamma_diff", type=float, default=0.1)
    parser.add_argument("--use_mddm", action="store_true")
    parser.add_argument(
        "--mddm_variant",
        type=str,
        default="base",
        choices=["base", "large_scale", "dual_scale_gate"],
    )
    parser.add_argument("--mddm_gamma_init", type=float, default=0.0)
    str2bool_flag(parser, "mddm_use_roi", False)
    parser.add_argument("--dg_gate_reduction", type=int, default=4)
    parser.add_argument("--dg_gamma_init", type=float, default=0.0)
    parser.add_argument(
        "--dg_gamma_mode",
        type=str,
        default="sigmoid01",
        choices=["raw", "sigmoid01", "softplus"],
    )
    str2bool_flag(parser, "dg_use_roi", True)
    parser.add_argument("--dg_large_beta", type=float, default=1.0)
    parser.add_argument("--dg_warmup_iters", type=int, default=0)
    parser.add_argument("--dg_gate_bias_init", type=float, default=0.0)
    parser.add_argument("--use_inner_ring_context", action="store_true")
    parser.add_argument("--ring_kernel_size", type=int, default=15)
    parser.add_argument("--irc_gamma_init", type=float, default=0.0)
    parser.add_argument("--use_damage_exist_head", action="store_true")
    parser.add_argument("--lambda_exist", type=float, default=0.1)
    parser.add_argument("--exist_pos_weight", type=float, default=1.0)
    parser.add_argument("--exist_loss_type", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--exist_focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_exist_consistency", action="store_true", help="Training-only compatibility flag.")
    parser.add_argument("--lambda_exist_cons", type=float, default=0.0)
    parser.add_argument("--exist_cons_warmup", type=int, default=10000)
    parser.add_argument("--exist_cons_mode", type=str, default="mse_detach", choices=["mse_detach"])
    parser.add_argument("--use_exist_logit_calib", action="store_true")
    parser.add_argument("--exist_calib_alpha", type=float, default=0.1)
    parser.add_argument("--exist_calib_beta", type=float, default=0.1)
    parser.add_argument("--exist_calib_warmup", type=int, default=10000)
    parser.add_argument("--use_exist_feature_modulation", action="store_true")
    parser.add_argument("--exist_fm_gamma", type=float, default=0.1)
    parser.add_argument("--exist_fm_warmup", type=int, default=10000)
    parser.add_argument("--use_hdc", action="store_true")
    parser.add_argument("--hdc_mode", type=str, default="blend", choices=["aux", "blend", "full"])
    parser.add_argument("--hdc_alpha", type=float, default=0.3)
    parser.add_argument("--lambda_hdc", type=float, default=0.5)
    parser.add_argument("--lambda_severity", type=float, default=0.2)
    parser.add_argument("--use_severity_head", action="store_true")
    parser.add_argument("--use_severity_proto", action="store_true")
    parser.add_argument("--proto_k", type=int, default=3)
    parser.add_argument("--proto_tau", type=float, default=0.1)
    parser.add_argument("--use_intact_consistency", action="store_true")
    parser.add_argument("--lambda_ic", type=float, default=0.01)
    parser.add_argument("--ic_dim", type=int, default=256)
    parser.add_argument("--ic_normalize", action="store_true")
    parser.add_argument("--use_damage_margin", action="store_true")
    parser.add_argument("--lambda_margin", type=float, default=0.0)
    parser.add_argument("--margin_damaged", type=float, default=0.3)
    parser.add_argument("--margin_destroyed", type=float, default=0.6)
    parser.add_argument("--use_damage_under_loss", action="store_true", help="Training-only compatibility flag.")
    parser.add_argument("--lambda_under", type=float, default=0.005)
    parser.add_argument("--under_warmup", type=int, default=20000)
    parser.add_argument("--under_margin", type=float, default=0.05)
    parser.add_argument("--use_bdpcde", action="store_true")
    parser.add_argument("--bdpcde_use_damage_prior", action="store_true")
    parser.add_argument("--bdpcde_beta", type=float, default=0.5)
    parser.add_argument("--bdpcde_gamma", type=float, default=1.0)
    str2bool_flag(parser, "bdpcde_detach_prior", True)
    parser.add_argument("--bdpcde_dropout", type=float, default=0.0)
    parser.add_argument("--use_sfdg", action="store_true")
    parser.add_argument("--sfdg_gamma", type=float, default=0.5)
    parser.add_argument("--sfdg_detach_gate", action="store_true")
    parser.add_argument("--event_index", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
