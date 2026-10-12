# -*- coding: utf-8 -*-

import argparse
import json
import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BDA_ROOT = PROJECT_ROOT / "bda_benchmark"

for p in (PROJECT_ROOT, BDA_ROOT, SCRIPT_DIR):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_BGMP_Aux import DamageFormer_BGMP_Aux


CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_data_list_path", default="dataset/splitname/standard_ML/val_set.txt")
    p.add_argument("--test_data_list_path", default="dataset/splitname/standard_ML/test_set.txt")
    p.add_argument("--split", default="both", choices=["val", "test", "both"])
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--roi_alpha", type=float, default=0.1)
    p.add_argument("--proto_dim", type=int, default=256)
    p.add_argument("--proto_k_list", type=str, default="1,3,3")
    p.add_argument("--proto_tau", type=float, default=0.1)
    p.add_argument("--proto_warmup_iters", type=int, default=5000)
    p.add_argument("--proto_assign_mode", type=str, default="sinkhorn", choices=["hard", "soft", "sinkhorn"])

    p.add_argument("--proto_logit_weight", type=float, default=0.0, help="Eval-only: final damage logits[1:4] += weight * prototype logits")

    p.add_argument("--output_json", required=True)

    return p.parse_args()


def strip_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj:
            state = obj["state_dict"]
        elif "model" in obj:
            state = obj["model"]
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        else:
            state = obj
    else:
        state = obj

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


def confusion_matrix(gt, pred, num_classes=4):
    valid = (gt >= 0) & (gt < num_classes)
    gt = gt[valid].astype(np.int64)
    pred = pred[valid].astype(np.int64)
    idx = gt * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    precision = tp / np.maximum(col_sum, 1.0)
    recall = tp / np.maximum(row_sum, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    iou = tp / np.maximum(row_sum + col_sum - tp, 1.0)

    oa = tp.sum() / np.maximum(cm.sum(), 1.0)

    result = {
        "oa": float(oa * 100.0),
        "class_iou": [float(x * 100.0) for x in iou],
        "class_precision": [float(x * 100.0) for x in precision],
        "class_recall": [float(x * 100.0) for x in recall],
        "class_f1": [float(x * 100.0) for x in f1],
        "miou": float(iou.mean() * 100.0),
        "target_3class_miou": float(iou[1:4].mean() * 100.0),
    }

    damage_f1 = f1[1:4]
    result["damage_f1_intact"] = float(damage_f1[0] * 100.0)
    result["damage_f1_damaged"] = float(damage_f1[1] * 100.0)
    result["damage_f1_destroyed"] = float(damage_f1[2] * 100.0)
    result["damage_harmonic_f1"] = float(
        3.0 / np.maximum(np.sum(1.0 / np.maximum(damage_f1, 1e-12)), 1e-12) * 100.0
    )

    return result


def binary_metrics(gt_build, pred_build):
    gt = gt_build.astype(bool)
    pr = pred_build.astype(bool)

    tp = np.logical_and(gt, pr).sum()
    fp = np.logical_and(~gt, pr).sum()
    fn = np.logical_and(gt, ~pr).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "loc_iou": float(iou * 100.0),
        "loc_precision": float(precision * 100.0),
        "loc_recall": float(recall * 100.0),
        "loc_f1": float(f1 * 100.0),
    }


def damage_error_rates(cm):
    cm = cm.astype(np.float64)
    out = {}

    damaged_total = max(cm[2].sum(), 1.0)
    destroyed_total = max(cm[3].sum(), 1.0)

    out["damaged_to_background"] = float(cm[2, 0] / damaged_total * 100.0)
    out["damaged_to_intact"] = float(cm[2, 1] / damaged_total * 100.0)
    out["damaged_to_destroyed"] = float(cm[2, 3] / damaged_total * 100.0)

    out["destroyed_to_background"] = float(cm[3, 0] / destroyed_total * 100.0)
    out["destroyed_to_intact"] = float(cm[3, 1] / destroyed_total * 100.0)
    out["destroyed_to_damaged"] = float(cm[3, 2] / destroyed_total * 100.0)

    return out


def row_percent(cm):
    cm = cm.astype(np.float64)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) * 100.0


def infer_event_name(data_idx):
    if isinstance(data_idx, (list, tuple)):
        data_idx = data_idx[0]
    if torch.is_tensor(data_idx):
        data_idx = str(int(data_idx.item()))

    s = str(data_idx).replace("\\", "/").lower()
    stem = Path(s).stem.lower()

    # remove common image suffix such as _00000018
    stem = re.sub(r"_\d{6,}$", "", stem)

    # remove modality / label suffixes if they appear
    stem = re.sub(r"_(pre|post|pre_disaster|post_disaster|label|target|mask).*", "", stem)

    # if path contains folders, prefer event-like folder name
    parts = [Path(x).stem.lower() for x in s.split("/") if x]
    candidates = []
    for item in parts[::-1]:
        item = re.sub(r"_\d{6,}$", "", item)
        item = re.sub(r"_(pre|post|pre_disaster|post_disaster|label|target|mask).*", "", item)
        candidates.append(item)

    keywords = [
        "earthquake", "explosion", "wildfire", "flood", "hurricane",
        "storm", "conflict", "volcano", "volcanic", "typhoon",
        "bata", "beirut", "haiti", "turkey", "libya", "hawaii",
        "ukraine", "morocco", "noto", "myanmar", "mexico",
    ]

    for item in candidates:
        if any(k in item for k in keywords):
            return item

    return stem if stem else "unknown"


def load_dataset(dataset_path, list_path):
    # BRIGHT dataset class expects a Python list of sample ids, not the txt path string.
    list_file = Path(list_path)
    if not list_file.is_absolute():
        list_file = Path.cwd() / list_file

    if not list_file.exists():
        raise FileNotFoundError(f"Data list not found: {list_file}")

    with open(list_file, "r", encoding="utf-8") as f:
        data_list = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"Loaded data list: {list_file} | samples={len(data_list)}")

    return MultimodalDamageAssessmentDatset(
        dataset_path=dataset_path,
        data_list=data_list,
        crop_size=None,
        max_iters=None,
        type="val",
    )


@torch.no_grad()
def evaluate(model, loader, device, split_name, proto_logit_weight=0.0):
    model.eval()

    global_cm = np.zeros((4, 4), dtype=np.int64)
    global_loc_gt = []
    global_loc_pred = []

    event_cm = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    event_loc_gt = defaultdict(list)
    event_loc_pred = defaultdict(list)
    event_count = defaultdict(int)

    proto_correct = 0
    proto_total = 0

    for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
        if len(batch) == 5:
            pre, post, labels_loc, labels_clf, data_idx = batch
        else:
            pre, post, labels_loc, labels_clf, data_idx = batch[:5]

        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc = labels_loc.to(device, non_blocking=True).long()
        labels_clf = labels_clf.to(device, non_blocking=True).long()

        output_loc, output_dam, aux = model(
            pre,
            post,
            damage_target=labels_clf,
            current_iter=None,
            return_aux=True,
        )

        if abs(float(proto_logit_weight)) > 1e-12:
            proto_logits = aux.get("proto_class_logits", None)
            if proto_logits is not None:
                proto_logits_up = F.interpolate(
                    proto_logits,
                    size=output_dam.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                output_dam = output_dam.clone()
                # prototype logits only affect Intact/Damaged/Destroyed.
                # Background logit remains from the main classifier.
                output_dam[:, 1:4] = output_dam[:, 1:4] + float(proto_logit_weight) * proto_logits_up

        pred_loc = output_loc.argmax(dim=1)
        pred_dam = output_dam.argmax(dim=1)

        gt_np = labels_clf.detach().cpu().numpy()
        pred_np = pred_dam.detach().cpu().numpy()

        loc_gt_np = (labels_loc.detach().cpu().numpy() > 0).astype(np.uint8)
        loc_pred_np = (pred_loc.detach().cpu().numpy() > 0).astype(np.uint8)

        for i in range(gt_np.shape[0]):
            event = infer_event_name(data_idx[i] if isinstance(data_idx, (list, tuple)) else data_idx)

            cm = confusion_matrix(gt_np[i], pred_np[i], num_classes=4)
            global_cm += cm
            event_cm[event] += cm

            global_loc_gt.append(loc_gt_np[i].reshape(-1))
            global_loc_pred.append(loc_pred_np[i].reshape(-1))

            event_loc_gt[event].append(loc_gt_np[i].reshape(-1))
            event_loc_pred[event].append(loc_pred_np[i].reshape(-1))
            event_count[event] += 1

        proto_logits = aux.get("proto_class_logits", None)
        if proto_logits is not None:
            proto_logits_up = F.interpolate(
                proto_logits,
                size=labels_clf.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            proto_pred = proto_logits_up.argmax(dim=1)
            valid = (labels_clf >= 1) & (labels_clf <= 3)
            if bool(valid.any().item()):
                proto_target = labels_clf - 1
                proto_correct += int((proto_pred[valid] == proto_target[valid]).sum().item())
                proto_total += int(valid.sum().item())

    global_metrics = metrics_from_cm(global_cm)
    global_errors = damage_error_rates(global_cm)

    loc_gt = np.concatenate(global_loc_gt)
    loc_pred = np.concatenate(global_loc_pred)
    global_loc = binary_metrics(loc_gt, loc_pred)

    proto_acc = 100.0 * proto_correct / max(proto_total, 1)

    event_results = []
    for event, cm in event_cm.items():
        m = metrics_from_cm(cm)
        e = damage_error_rates(cm)

        lg = np.concatenate(event_loc_gt[event])
        lp = np.concatenate(event_loc_pred[event])
        lm = binary_metrics(lg, lp)

        item = {
            "event": event,
            "num_images": event_count[event],
            **m,
            **e,
            **lm,
        }
        event_results.append(item)

    event_results = sorted(event_results, key=lambda x: x["miou"])

    output = {
        "split": split_name,
        "global": {
            **global_metrics,
            **global_errors,
            **global_loc,
            "prototype_class_accuracy": float(proto_acc),
        },
        "confusion_matrix": global_cm.tolist(),
        "confusion_row_percent": row_percent(global_cm).tolist(),
        "events": event_results,
        "event_mean_miou": float(np.mean([e["miou"] for e in event_results])) if event_results else None,
        "event_worst": event_results[0] if event_results else None,
    }

    return output


def print_summary(res):
    g = res["global"]

    print("\n---------------- SUMMARY:", res["split"], "----------------")
    print(f"Localization F1       : {g['loc_f1']:.4f}")
    print(f"Damage Harmonic F1    : {g['damage_harmonic_f1']:.4f}")
    print(
        "Damage F1 [I/D/Dest] : "
        f"{g['damage_f1_intact']:.4f} / "
        f"{g['damage_f1_damaged']:.4f} / "
        f"{g['damage_f1_destroyed']:.4f}"
    )
    print(f"OA                    : {g['oa']:.4f}")
    print(f"mIoU                  : {g['miou']:.4f}")
    print(f"target_3class_mIoU    : {g['target_3class_miou']:.4f}")
    print(f"Prototype class acc   : {g['prototype_class_accuracy']:.4f}")

    print("\nClass IoU:")
    for name, val in zip(CLASS_NAMES, g["class_iou"]):
        print(f"  {name:<10s}: {val:.4f}")

    print("\nKey damage errors:")
    print(f"  Damaged   -> BG        : {g['damaged_to_background']:.4f}%")
    print(f"  Damaged   -> Intact    : {g['damaged_to_intact']:.4f}%")
    print(f"  Damaged   -> Destroyed : {g['damaged_to_destroyed']:.4f}%")
    print(f"  Destroyed -> BG        : {g['destroyed_to_background']:.4f}%")
    print(f"  Destroyed -> Intact    : {g['destroyed_to_intact']:.4f}%")
    print(f"  Destroyed -> Damaged   : {g['destroyed_to_damaged']:.4f}%")

    print("\nWorst event:")
    if res["event_worst"] is not None:
        w = res["event_worst"]
        print(
            f"  {w['event']} | images={w['num_images']} | "
            f"mIoU={w['miou']:.4f} | "
            f"Damaged IoU={w['class_iou'][2]:.4f} | "
            f"Destroyed IoU={w['class_iou'][3]:.4f} | "
            f"Dmg->Intact={w['damaged_to_intact']:.2f}% | "
            f"Dest->Intact={w['destroyed_to_intact']:.2f}%"
        )

    print("\nDamage confusion row percentages:")
    r = np.array(res["confusion_row_percent"])
    header = "{:<12s} {:>10s} {:>10s} {:>10s} {:>10s}".format("GT \\ Pred", "BG", "Intact", "Damaged", "Destroyed")
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<12s} "
            f"{r[i,0]:10.4f} {r[i,1]:10.4f} "
            f"{r[i,2]:10.4f} {r[i,3]:10.4f}"
        )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Loading checkpoint:", args.checkpoint)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = strip_state_dict(raw)

    proto_key = "damage_proto_memory.prototypes"
    if proto_key in state:
        proto_shape = tuple(state[proto_key].shape)
        print("\nCheckpoint architecture inference:")
        print("  prototype key :", proto_key)
        print("  prototype bank:", proto_shape)
        print("  proto_dim     :", proto_shape[-1])
        args.proto_dim = int(proto_shape[-1])

        if "damage_proto_memory.prototype_valid" in state:
            valid = state["damage_proto_memory.prototype_valid"]
            k_list = valid.long().sum(dim=1).tolist()
            args.proto_k_list = ",".join(str(int(x)) for x in k_list)
            print("  proto_k_list  :", args.proto_k_list)
    else:
        print("WARNING: no prototype bank found in checkpoint.")

    model = DamageFormer_BGMP_Aux(
        roi_alpha=args.roi_alpha,
        use_pretrained_swin=False,
        proto_dim=args.proto_dim,
        proto_k_list=args.proto_k_list,
        proto_tau=args.proto_tau,
        proto_warmup_iters=args.proto_warmup_iters,
        proto_assign_mode=args.proto_assign_mode,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=True)
    print("\nSTRICT CHECKPOINT LOAD: PASS")

    model.eval()

    memory = model.damage_proto_memory
    print(f"\nEval proto_logit_weight: {args.proto_logit_weight}")

    print("\n================ PROTOTYPE DIAGNOSTICS ================")
    print("prototype shape :", list(memory.prototypes.shape))
    print("initialized     :", memory.prototype_initialized.tolist())
    print("valid mask      :", memory.prototype_valid.int().tolist())
    print("usage ratio     :", (
        memory.prototype_usage / memory.prototype_usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
    ).detach().cpu().tolist())

    results = {}

    if args.split in ("val", "both"):
        val_ds = load_dataset(args.dataset_path, args.val_data_list_path)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("\nValidation samples:", len(val_ds))
        res = evaluate(model, val_loader, device, "validation", proto_logit_weight=args.proto_logit_weight)
        print_summary(res)
        results["validation"] = res

    if args.split in ("test", "both"):
        test_ds = load_dataset(args.dataset_path, args.test_data_list_path)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("\nTest samples:", len(test_ds))
        res = evaluate(model, test_loader, device, "testing", proto_logit_weight=args.proto_logit_weight)
        print_summary(res)
        results["testing"] = res

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nEvaluation JSON saved to:", out)


if __name__ == "__main__":
    main()
