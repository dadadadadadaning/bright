# -*- coding: utf-8 -*-

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
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
    p.add_argument("--proto_tau", type=float, default=0.1)
    p.add_argument("--proto_warmup_iters", type=int, default=5000)
    p.add_argument("--proto_assign_mode", type=str, default="sinkhorn")

    p.add_argument("--model_module", default="", help="Example: model.DamageFormer_BGMP_AutoKLogit")
    p.add_argument("--model_class", default="", help="Example: DamageFormer_BGMP_AutoKLogit")
    p.add_argument("--model_file", default="", help="Import model class from a python file path, e.g. model/xxx.py or script/standard_ML/train_xxx.py")

    p.add_argument(
        "--lambda_sweep",
        default="0,0.05,0.1,0.2,-0.05,-0.1",
        help="Only used if proto logits are returned in aux.",
    )

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


def find_candidate_model_class():
    candidates = []

    model_dir = Path("model")
    patterns = [
        "*S3V4*.py",
        "*Auto*K*Logit*.py",
        "*AutoK*.py",
        "*Logit*.py",
        "*BGMP*.py",
    ]

    files = []
    for pat in patterns:
        files.extend(model_dir.glob(pat))

    files = sorted(set(files))

    for f in files:
        module_name = "model." + f.stem
        try:
            m = importlib.import_module(module_name)
        except Exception:
            continue

        for name, obj in m.__dict__.items():
            if inspect.isclass(obj):
                lname = name.lower()
                if (
                    "s3v4" in lname
                    or ("autok" in lname and "logit" in lname)
                    or ("bgmp" in lname and "logit" in lname)
                ):
                    candidates.append((module_name, name, obj))

    return candidates


def load_model_class(args):
    # Highest priority: import class from a given python file.
    if args.model_file and args.model_class:
        model_file = Path(args.model_file).resolve()
        if not model_file.exists():
            raise FileNotFoundError(f"model_file not found: {model_file}")

        module_name = "dynamic_s3v4_model"
        spec = importlib.util.spec_from_file_location(module_name, str(model_file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        cls = getattr(module, args.model_class)
        print(f"Using model class from file: {model_file}::{args.model_class}")
        return cls

    # Second priority: import from module path.
    if args.model_module and args.model_class:
        module = importlib.import_module(args.model_module)
        cls = getattr(module, args.model_class)
        print(f"Using model class: {args.model_module}.{args.model_class}")
        return cls

    # Fallback: auto detect in model/*.py.
    candidates = find_candidate_model_class()

    if len(candidates) == 1:
        module_name, name, cls = candidates[0]
        print(f"Auto-detected model class: {module_name}.{name}")
        return cls

    print("\nCould not uniquely auto-detect S3V4 model class.")
    print("Candidates:")
    for module_name, name, _ in candidates:
        print(f"  {module_name}.{name}")

    raise RuntimeError(
        "Please rerun with one of the following:\n"
        "  1) --model_module model.xxx --model_class Xxx\n"
        "  2) --model_file path/to/model_or_train_script.py --model_class Xxx"
    )



def infer_arch_from_state(state):
    info = {
        "proto_dim": 256,
        "proto_k": 5,
        "proto_kmax": 5,
        "proto_k_list": "5,5,5",
    }

    proto_keys = [k for k in state.keys() if k.endswith("prototypes")]
    proto_keys = [k for k in proto_keys if "damage_proto_memory" in k or "proto" in k.lower()]

    if proto_keys:
        key = proto_keys[0]
        shape = tuple(state[key].shape)
        print("\nCheckpoint architecture inference:")
        print("  prototype key :", key)
        print("  prototype bank:", shape)

        if len(shape) == 3:
            info["proto_k"] = int(shape[1])
            info["proto_kmax"] = int(shape[1])
            info["proto_dim"] = int(shape[2])
            info["proto_k_list"] = ",".join([str(int(shape[1]))] * 3)

    valid_keys = [k for k in state.keys() if k.endswith("prototype_valid")]
    if valid_keys:
        valid = state[valid_keys[0]]
        if valid.ndim == 2 and valid.shape[0] == 3:
            k_list = valid.long().sum(dim=1).tolist()
            info["proto_k_list"] = ",".join(str(int(x)) for x in k_list)
            info["proto_k"] = int(max(k_list))
            info["proto_kmax"] = int(max(k_list))
            print("  valid mask key:", valid_keys[0])
            print("  proto_k_list :", info["proto_k_list"])

    print("  inferred proto_dim :", info["proto_dim"])
    print("  inferred proto_kmax:", info["proto_kmax"])

    return info


def build_model(cls, args, arch):
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    possible_kwargs = {
        "roi_alpha": args.roi_alpha,
        "detach_building_prior": True,
        "use_pretrained_swin": False,
        "pretrained": False,
        "proto_dim": arch["proto_dim"],
        "proto_k": arch["proto_k"],
        "proto_kmax": arch["proto_kmax"],
        "proto_k_max": arch["proto_kmax"],
        "proto_max_k": arch["proto_kmax"],
        "kmax": arch["proto_kmax"],
        "proto_k_list": arch["proto_k_list"],
        "proto_tau": args.proto_tau,
        "tau": args.proto_tau,
        "proto_warmup_iters": args.proto_warmup_iters,
        "warmup_iters": args.proto_warmup_iters,
        "proto_assign_mode": args.proto_assign_mode,
        "assign_mode": args.proto_assign_mode,
    }

    kwargs = {}
    for k, v in possible_kwargs.items():
        if k in params:
            kwargs[k] = v

    print("\nModel init kwargs:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    return cls(**kwargs)


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

    damage_f1 = f1[1:4]
    harmonic = 3.0 / np.maximum(np.sum(1.0 / np.maximum(damage_f1, 1e-12)), 1e-12)

    return {
        "oa": float(oa * 100.0),
        "class_iou": [float(x * 100.0) for x in iou],
        "class_precision": [float(x * 100.0) for x in precision],
        "class_recall": [float(x * 100.0) for x in recall],
        "class_f1": [float(x * 100.0) for x in f1],
        "miou": float(iou.mean() * 100.0),
        "target_3class_miou": float(iou[1:4].mean() * 100.0),
        "damage_f1_intact": float(f1[1] * 100.0),
        "damage_f1_damaged": float(f1[2] * 100.0),
        "damage_f1_destroyed": float(f1[3] * 100.0),
        "damage_harmonic_f1": float(harmonic * 100.0),
    }


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

    damaged_total = max(cm[2].sum(), 1.0)
    destroyed_total = max(cm[3].sum(), 1.0)

    return {
        "damaged_to_background": float(cm[2, 0] / damaged_total * 100.0),
        "damaged_to_intact": float(cm[2, 1] / damaged_total * 100.0),
        "damaged_to_destroyed": float(cm[2, 3] / damaged_total * 100.0),
        "destroyed_to_background": float(cm[3, 0] / destroyed_total * 100.0),
        "destroyed_to_intact": float(cm[3, 1] / destroyed_total * 100.0),
        "destroyed_to_damaged": float(cm[3, 2] / destroyed_total * 100.0),
    }


def row_percent(cm):
    cm = cm.astype(np.float64)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) * 100.0


def infer_event_name(data_idx):
    if torch.is_tensor(data_idx):
        if data_idx.numel() == 1:
            data_idx = int(data_idx.item())
        else:
            data_idx = data_idx.detach().cpu().tolist()

    if isinstance(data_idx, (list, tuple)) and len(data_idx) > 0:
        data_idx = data_idx[0]

    s = str(data_idx).replace("\\", "/")
    parts = [p for p in s.split("/") if p]

    keywords = [
        "earthquake", "explosion", "wildfire", "flood", "hurricane",
        "storm", "conflict", "volcano", "volcanic", "typhoon",
        "bata", "beirut", "haiti", "turkey", "libya", "hawaii",
        "ukraine", "morocco", "noto", "myanmar", "mexico",
    ]

    for p in parts:
        low = p.lower()
        if any(k in low for k in keywords):
            return low

    if len(parts) >= 2:
        return parts[-2].lower()
    if len(parts) == 1:
        return Path(parts[0]).stem.lower()
    return "unknown"


def load_dataset(dataset_path, list_path):
    list_file = Path(list_path)

    if not list_file.is_absolute():
        candidates = [
            Path.cwd() / list_file,
            BDA_ROOT / list_file,
            PROJECT_ROOT / list_file,
        ]
        list_file = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[0],
        )

    if not list_file.exists():
        raise FileNotFoundError(
            f"Data-list file not found: {list_file}"
        )

    data_list = [
        line.strip()
        for line in list_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
        if line.strip()
    ]

    if len(data_list) == 0:
        raise RuntimeError(
            f"Data-list file is empty: {list_file}"
        )

    print(
        f"Loaded data list: {list_file} | "
        f"samples={len(data_list)}"
    )

    return MultimodalDamageAssessmentDatset(
        dataset_path=dataset_path,
        data_list=data_list,
        crop_size=None,
        max_iters=None,
        type="val",
    )



def call_model(model, pre, post, labels_clf):
    attempts = [
        lambda: model(pre, post, damage_target=labels_clf, current_iter=None, return_aux=True),
        lambda: model(pre, post, current_iter=None, return_aux=True),
        lambda: model(pre, post, return_aux=True),
        lambda: model(pre, post),
    ]

    last_error = None
    for fn in attempts:
        try:
            out = fn()
            break
        except TypeError as e:
            last_error = e
            out = None
    if out is None:
        raise last_error

    aux = {}

    if isinstance(out, (list, tuple)):
        if len(out) >= 3 and isinstance(out[2], dict):
            output_loc, output_dam, aux = out[0], out[1], out[2]
        elif len(out) >= 2:
            output_loc, output_dam = out[0], out[1]
        else:
            raise RuntimeError("Model output tuple too short.")
    else:
        raise RuntimeError("Model output should be tuple/list.")

    return output_loc, output_dam, aux


def find_proto_logits(aux):
    if not isinstance(aux, dict):
        return None, None

    candidates = [
        "proto_class_logits",
        "prototype_class_logits",
        "proto_logits",
        "damage_proto_logits",
        "prototype_logits",
        "auto_k_proto_logits",
    ]

    for k in candidates:
        v = aux.get(k, None)
        if torch.is_tensor(v) and v.ndim == 4:
            if v.shape[1] == 3:
                return k, v

    for k, v in aux.items():
        if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 3:
            return k, v

    return None, None


@torch.no_grad()
def evaluate(model, loader, device, split_name, lambda_values):
    model.eval()

    modes = ["final"]
    for lam in lambda_values:
        modes.append(f"lambda_{lam:g}")

    cms = {m: np.zeros((4, 4), dtype=np.int64) for m in modes}
    event_cms = {m: defaultdict(lambda: np.zeros((4, 4), dtype=np.int64)) for m in modes}
    event_count = defaultdict(int)

    loc_gt_all = []
    loc_pred_all = []

    proto_key_seen = None
    proto_available_count = 0

    for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
        if len(batch) >= 5:
            pre, post, labels_loc, labels_clf, data_idx = batch[:5]
        else:
            raise RuntimeError("Dataset batch must contain at least 5 items.")

        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels_loc = labels_loc.to(device, non_blocking=True).long()
        labels_clf = labels_clf.to(device, non_blocking=True).long()

        output_loc, output_dam, aux = call_model(model, pre, post, labels_clf)

        proto_key, proto_logits = find_proto_logits(aux)
        if proto_logits is not None:
            proto_key_seen = proto_key
            proto_available_count += 1
            proto_logits = F.interpolate(
                proto_logits,
                size=output_dam.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pred_loc = output_loc.argmax(dim=1)
        loc_gt_np = (labels_loc.detach().cpu().numpy() > 0).astype(np.uint8)
        loc_pred_np = (pred_loc.detach().cpu().numpy() > 0).astype(np.uint8)
        loc_gt_all.append(loc_gt_np.reshape(-1))
        loc_pred_all.append(loc_pred_np.reshape(-1))

        gt_np = labels_clf.detach().cpu().numpy()

        # Resolve numeric dataset indices back to sample identifiers.
        # This is required for correct per-event evaluation.
        batch_events = []

        for i in range(gt_np.shape[0]):
            if torch.is_tensor(data_idx):
                idx_item = data_idx[i]
            elif isinstance(data_idx, (list, tuple)):
                idx_item = data_idx[i]
            else:
                idx_item = data_idx

            if torch.is_tensor(idx_item):
                if idx_item.numel() == 1:
                    idx_item = int(idx_item.item())
                else:
                    idx_item = idx_item.detach().cpu().tolist()

            if (
                isinstance(idx_item, (int, np.integer))
                and hasattr(loader.dataset, "data_list")
                and 0 <= int(idx_item) < len(loader.dataset.data_list)
            ):
                sample_reference = loader.dataset.data_list[int(idx_item)]
            else:
                sample_reference = idx_item

            event = infer_event_name(sample_reference)
            batch_events.append(event)
            event_count[event] += 1

        logits_dict = {"final": output_dam}

        if proto_logits is not None:
            for lam in lambda_values:
                fused = output_dam.clone()
                fused[:, 1:4] = fused[:, 1:4] + float(lam) * proto_logits
                logits_dict[f"lambda_{lam:g}"] = fused

        for mode, logits in logits_dict.items():
            pred_np = logits.argmax(dim=1).detach().cpu().numpy()

            for i in range(gt_np.shape[0]):
                event = batch_events[i]
                cm = confusion_matrix(
                    gt_np[i],
                    pred_np[i],
                    num_classes=4,
                )
                cms[mode] += cm
                event_cms[mode][event] += cm

    loc_metrics = binary_metrics(np.concatenate(loc_gt_all), np.concatenate(loc_pred_all))

    results = {}
    for mode in modes:
        m = metrics_from_cm(cms[mode])
        e = damage_error_rates(cms[mode])

        event_results = []
        for event, cm in event_cms[mode].items():
            em = metrics_from_cm(cm)
            ee = damage_error_rates(cm)
            event_results.append({
                "event": event,
                "num_images": int(event_count[event]),
                **em,
                **ee,
            })
        event_results = sorted(event_results, key=lambda x: x["miou"])

        results[mode] = {
            "split": split_name,
            "mode": mode,
            "global": {
                **m,
                **e,
                **loc_metrics,
            },
            "confusion_matrix": cms[mode].tolist(),
            "confusion_row_percent": row_percent(cms[mode]).tolist(),
            "events": event_results,
            "event_mean_miou": float(np.mean([x["miou"] for x in event_results])) if event_results else None,
            "event_worst": event_results[0] if event_results else None,
        }

    return {
        "proto_key_seen": proto_key_seen,
        "proto_available_batches": int(proto_available_count),
        "results": results,
    }


def print_one_summary(res):
    g = res["global"]

    print("\n---------------- SUMMARY:", res["split"], "|", res["mode"], "----------------")
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

    if res["event_worst"] is not None:
        w = res["event_worst"]
        print("\nWorst event:")
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
    header = "{:<12s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "GT \\ Pred", "BG", "Intact", "Damaged", "Destroyed"
    )
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<12s} "
            f"{r[i,0]:10.4f} {r[i,1]:10.4f} "
            f"{r[i,2]:10.4f} {r[i,3]:10.4f}"
        )


def print_prototype_diagnostics(model):
    print("\n================ PROTOTYPE / LOGIT DIAGNOSTICS ================")

    for name, module in model.named_modules():
        if "proto" in name.lower() and hasattr(module, "prototypes"):
            proto = module.prototypes.detach()
            print(f"prototype module : {name}")
            print("prototype shape  :", list(proto.shape))
            print("norm range       :", float(proto.norm(dim=-1).min()), float(proto.norm(dim=-1).max()))

            if hasattr(module, "prototype_initialized"):
                print("initialized      :", module.prototype_initialized.detach().cpu().tolist())

            if hasattr(module, "prototype_valid"):
                print("valid mask       :", module.prototype_valid.detach().int().cpu().tolist())

            if hasattr(module, "prototype_usage"):
                usage = module.prototype_usage.detach().float()
                ratio = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
                print("usage ratio      :", ratio.cpu().tolist())

    print("\nScalar parameters related to proto/gamma/logit:")
    found = False
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in ["gamma", "proto_logit", "logit_weight", "fusion_weight", "lambda"]):
            if p.numel() <= 10:
                print(f"  {name}: {p.detach().cpu().flatten().tolist()}")
                found = True

    if not found:
        print("  No small scalar parameter found.")


def main():
    args = parse_args()
    device = torch.device(args.device)

    lambda_values = []
    for x in args.lambda_sweep.split(","):
        x = x.strip()
        if x:
            lambda_values.append(float(x))

    print("Loading checkpoint:", args.checkpoint)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = strip_state_dict(raw)

    arch = infer_arch_from_state(state)
    cls = load_model_class(args)
    model = build_model(cls, args, arch).to(device)

    try:
        model.load_state_dict(state, strict=True)
        print("\nSTRICT CHECKPOINT LOAD: PASS")
    except RuntimeError as e:
        print("\nSTRICT CHECKPOINT LOAD: FAIL")
        print(e)
        raise

    model.eval()
    print_prototype_diagnostics(model)

    results_all = {}

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
        pack = evaluate(model, val_loader, device, "validation", lambda_values)
        print("proto logits key seen:", pack["proto_key_seen"])
        print("proto available batches:", pack["proto_available_batches"])
        results_all["validation"] = pack

        print_one_summary(pack["results"]["final"])
        best = max(pack["results"].values(), key=lambda x: x["global"]["miou"])
        if best["mode"] != "final":
            print_one_summary(best)

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
        pack = evaluate(model, test_loader, device, "testing", lambda_values)
        print("proto logits key seen:", pack["proto_key_seen"])
        print("proto available batches:", pack["proto_available_batches"])
        results_all["testing"] = pack

        print_one_summary(pack["results"]["final"])
        best = max(pack["results"].values(), key=lambda x: x["global"]["miou"])
        if best["mode"] != "final":
            print("\nBest lambda-sweep mode on this split:")
            print_one_summary(best)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nEvaluation JSON saved to:", out)


if __name__ == "__main__":
    main()
