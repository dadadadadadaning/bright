# -*- coding: utf-8 -*-
"""Full active-vs-base evaluation for IARM-DamageFormer."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.DamageFormer_IARM import DamageFormer_IARM


CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate IARM-DamageFormer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_json", default=None)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_data_list_path", required=True)
    p.add_argument("--test_data_list_path", required=True)
    p.add_argument("--split", choices=("val", "test", "both"), default="both")
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--roi_alpha", type=float, default=None)
    p.add_argument("--residual_stages", default=None)
    p.add_argument("--residual_dim", type=int, default=None)
    p.add_argument("--flow_hidden_dim", type=int, default=64)
    p.add_argument("--max_offset_pixels", type=float, default=None)
    p.add_argument("--detach_backbone_for_residual", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--enable_residual_fusion", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--residual_gamma_max", type=float, default=None)
    p.add_argument("--num_events", type=int, default=14)
    p.add_argument("--gaussian_variance_floor", type=float, default=None)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def read_list(path: str) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_event_name(name: str) -> str:
    stem = Path(str(name).replace("\\", "/")).stem.lower()
    stem = re.sub(r"_(pre|post)_disaster$", "", stem)
    return re.sub(r"_\d+$", "", stem)


def clean_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object], Dict[str, object]]:
    obj = torch.load(path, map_location="cpu")
    metadata: Dict[str, object] = {}
    checkpoint_args: Dict[str, object] = {}
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
        metadata = {"format": "training_state", "optimizer_step": obj.get("optimizer_step")}
        raw_args = obj.get("args", {})
        checkpoint_args = vars(raw_args) if isinstance(raw_args, argparse.Namespace) else dict(raw_args) if isinstance(raw_args, dict) else {}
    else:
        state = obj
        metadata = {"format": "plain_state_dict"}
    return clean_state(state), checkpoint_args, metadata


def infer_stages(state: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    stages = sorted({int(key.split(".")[1]) for key in state if key.startswith("optical_projectors.")})
    if not stages:
        raise RuntimeError("Could not infer residual stages from checkpoint")
    return tuple(stages)


def resolve(args: argparse.Namespace, checkpoint_args: Dict[str, object], state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    config: Dict[str, object] = {}
    if args.config_json:
        config.update(json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    config.update(checkpoint_args)

    def value(name: str, cli, default):
        return cli if cli is not None else config.get(name, default)

    stages = tuple(int(x) for x in args.residual_stages.split(",")) if args.residual_stages else tuple(config.get("residual_stages_parsed", infer_stages(state)))
    if isinstance(config.get("residual_stages"), str) and args.residual_stages is None:
        stages = tuple(int(x.strip()) for x in str(config["residual_stages"]).split(",") if x.strip())
    residual_dim = int(value("residual_dim", args.residual_dim, state["class_distribution_memory.means"].shape[1]))
    return {
        "roi_alpha": float(value("roi_alpha", args.roi_alpha, 0.1)),
        "residual_stages": stages,
        "residual_dim": residual_dim,
        "flow_hidden_dim": int(value("flow_hidden_dim", args.flow_hidden_dim, 64)),
        "max_offset_pixels": float(value("max_offset_pixels", args.max_offset_pixels, 2.0)),
        "detach_backbone_for_residual": bool(value("detach_backbone_for_residual", args.detach_backbone_for_residual, True)),
        "enable_residual_fusion": bool(value("enable_residual_fusion", args.enable_residual_fusion, True)),
        "residual_gamma_max": float(value("residual_gamma_max", args.residual_gamma_max, 0.25)),
        "gaussian_variance_floor": float(value("gaussian_variance_floor", args.gaussian_variance_floor, 2.0e-3)),
    }


def disable_unused_swin_heads(model: nn.Module) -> None:
    for name in ("encoder_1", "encoder_2"):
        swin = getattr(getattr(model, name), "swin_transformer")
        swin.norm = nn.Identity(); swin.head = nn.Identity()


def fast_hist(gt: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    gt = gt.reshape(-1); pred = pred.reshape(-1)
    valid = (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    return np.bincount(classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64), minlength=classes**2).reshape(classes, classes)


def metric_report(hist: np.ndarray) -> Dict[str, object]:
    tp = np.diag(hist).astype(np.float64)
    denom = hist.sum(1) + hist.sum(0) - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    precision = np.divide(tp, hist.sum(0), out=np.zeros_like(tp), where=hist.sum(0) > 0)
    recall = np.divide(tp, hist.sum(1), out=np.zeros_like(tp), where=hist.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    harmonic = 3.0 / np.sum(1.0 / np.clip(f1[1:4], 1e-12, None))
    rows = np.divide(hist, hist.sum(1, keepdims=True), out=np.zeros_like(hist, dtype=np.float64), where=hist.sum(1, keepdims=True) > 0)
    return {
        "oa": float(tp.sum() / max(hist.sum(), 1)),
        "miou": float(iou.mean()),
        "target3_miou": float(iou[1:4].mean()),
        "iou": iou.tolist(),
        "f1": f1.tolist(),
        "damage_hf1": float(harmonic),
        "row_percent": (100.0 * rows).tolist(),
        "damaged_to_intact": float(100.0 * rows[2, 1]),
        "destroyed_to_intact": float(100.0 * rows[3, 1]),
        "intact_to_damaged": float(100.0 * rows[1, 2]),
        "intact_to_destroyed": float(100.0 * rows[1, 3]),
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataset_path: str, names: Sequence[str], batch_size: int, workers: int, device: torch.device, split: str) -> Dict[str, object]:
    dataset = MultimodalDamageAssessmentDatset(dataset_path, list(names), 1024, None, "test")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    active_hist = np.zeros((4,4), np.int64); base_hist = np.zeros((4,4), np.int64); loc_hist = np.zeros((2,2), np.int64); exist_hist = np.zeros((2,2), np.int64)
    event_active = defaultdict(lambda: np.zeros((4,4), np.int64)); event_base = defaultdict(lambda: np.zeros((4,4), np.int64))
    residual_norm_sum = np.zeros(3, np.float64); residual_norm_count = np.zeros(3, np.int64)
    flow_magnitude = defaultdict(list)

    for pre, post, loc, damage, sample_names in tqdm(loader, desc=split):
        pre=pre.to(device); post=post.to(device); loc=loc.to(device).long(); damage=damage.to(device).long()
        out_loc, out_damage, out_exist, aux = model(pre, post, return_aux=True)
        base_logits = F.interpolate(aux["main_damage_logits_feat"], size=out_damage.shape[-2:], mode="bilinear", align_corners=False)
        pred_active=out_damage.argmax(1).cpu().numpy(); pred_base=base_logits.argmax(1).cpu().numpy(); gt=damage.cpu().numpy()
        pred_loc=out_loc.argmax(1).cpu().numpy(); gt_loc=loc.cpu().numpy(); valid_loc=gt_loc!=255
        active_hist += fast_hist(gt,pred_active,4); base_hist += fast_hist(gt,pred_base,4)
        loc_hist += fast_hist((gt_loc>0)[valid_loc].astype(np.int64),(pred_loc>0)[valid_loc].astype(np.int64),2)
        exist_gt=np.full_like(gt,255); exist_gt[gt==1]=0; exist_gt[(gt==2)|(gt==3)]=1; exist_pred=out_exist.argmax(1).cpu().numpy(); valid=exist_gt!=255
        exist_hist += fast_hist(exist_gt[valid],exist_pred[valid],2)
        embedding=aux["metric_embedding"]
        gt_small=F.interpolate(damage.unsqueeze(1).float(),size=embedding.shape[-2:],mode="nearest")[:,0].long()
        norm=embedding.norm(dim=1)
        for idx,label in enumerate((1,2,3)):
            mask=gt_small==label
            residual_norm_sum[idx]+=float(norm[mask].sum().item()); residual_norm_count[idx]+=int(mask.sum().item())
        for stage in model.residual_stages:
            flow_magnitude[str(stage)].append(float(aux[f"flow_s{stage}"].abs().mean().item()))
        for i,name in enumerate(sample_names):
            event=infer_event_name(str(name)); event_active[event]+=fast_hist(gt[i],pred_active[i],4); event_base[event]+=fast_hist(gt[i],pred_base[i],4)

    loc_tp=float(loc_hist[1,1]); loc_p=loc_tp/max(loc_tp+loc_hist[0,1],1); loc_r=loc_tp/max(loc_tp+loc_hist[1,0],1); loc_f1=2*loc_p*loc_r/max(loc_p+loc_r,1e-12)
    tp=float(exist_hist[1,1]); p=tp/max(tp+exist_hist[0,1],1); r=tp/max(tp+exist_hist[1,0],1); exist_f1=2*p*r/max(p+r,1e-12)
    active=metric_report(active_hist); base=metric_report(base_hist)
    events=[]
    for event in sorted(event_active):
        a=metric_report(event_active[event]); b=metric_report(event_base[event])
        events.append({"event":event,"active":a,"base":b,"delta_miou":a["miou"]-b["miou"]})
    event_macro=float(np.mean([item["active"]["target3_miou"] for item in events])) if events else float("nan")
    worst=min(events,key=lambda x:x["active"]["target3_miou"]) if events else None
    result={
        "split":split,"samples":len(dataset),"loc_f1":loc_f1,"damage_exist_f1":exist_f1,
        "active":active,"base":base,"active_minus_base_miou":active["miou"]-base["miou"],
        "event_macro_target3":event_macro,"worst_event":worst,"events":events,
        "mean_metric_norm_by_class":{name: float(residual_norm_sum[i]/max(residual_norm_count[i],1)) for i,name in enumerate(CLASS_NAMES[1:])},
        "mean_abs_flow_by_stage":{stage:float(np.mean(values)) for stage,values in flow_magnitude.items()},
    }
    return result


def print_summary(result: Dict[str, object]) -> None:
    print(f"\n================ {result['split']} ================")
    print(f"LocF1={100*result['loc_f1']:.4f}  DamageExistF1={100*result['damage_exist_f1']:.4f}")
    for mode in ("active","base"):
        m=result[mode]
        print(f"{mode:<8s}: mIoU={100*m['miou']:.4f}, target3={100*m['target3_miou']:.4f}, HF1={100*m['damage_hf1']:.4f}, IoU={100*np.asarray(m['iou'])}")
        print(f"           D->I={m['damaged_to_intact']:.4f}%, X->I={m['destroyed_to_intact']:.4f}%, I->D={m['intact_to_damaged']:.4f}%")
    print(f"active-base mIoU={100*result['active_minus_base_miou']:+.4f}")
    print(f"event macro target3={100*result['event_macro_target3']:.4f}")
    if result["worst_event"]:
        w=result["worst_event"]
        print(f"worst event={w['event']}, target3={100*w['active']['target3_miou']:.4f}")


def main() -> None:
    args=parse_args(); device=torch.device(args.device)
    state, checkpoint_args, meta=load_checkpoint(args.checkpoint); resolved=resolve(args,checkpoint_args,state)
    model=DamageFormer_IARM(
        roi_alpha=resolved["roi_alpha"],use_pretrained_swin=False,residual_stages=resolved["residual_stages"],residual_dim=resolved["residual_dim"],
        flow_hidden_dim=resolved["flow_hidden_dim"],max_offset_pixels=resolved["max_offset_pixels"],detach_backbone_for_residual=resolved["detach_backbone_for_residual"],
        enable_residual_fusion=resolved["enable_residual_fusion"],residual_gamma_max=resolved["residual_gamma_max"],num_events=args.num_events,
        gaussian_variance_floor=resolved["gaussian_variance_floor"],
    ).to(device)
    disable_unused_swin_heads(model); model.load_state_dict(state,strict=True); model.eval()
    print("Checkpoint:",args.checkpoint); print("Metadata:",meta); print("Resolved:",resolved); print("STRICT CHECKPOINT LOAD: PASS")
    print("Residual gamma:",float(model.residual_gamma().detach().item()))
    diag=model.class_distribution_memory.diagnostics()
    print("Gaussian initialized:",diag["gaussian_initialized"].cpu().tolist())
    print("Gaussian centre cosine:\n",diag["gaussian_centre_cosine"].cpu().numpy())
    print("Gaussian variance mean:",diag["gaussian_variance_mean"].cpu().tolist())
    output={"checkpoint":args.checkpoint,"metadata":meta,"resolved":resolved,"splits":{}}
    if args.split in ("val","both"):
        r=evaluate(model,args.dataset_path,read_list(args.val_data_list_path),args.eval_batch_size,args.num_workers,device,"validation"); output["splits"]["validation"]=r; print_summary(r)
    if args.split in ("test","both"):
        r=evaluate(model,args.dataset_path,read_list(args.test_data_list_path),args.eval_batch_size,args.num_workers,device,"testing"); output["splits"]["testing"]=r; print_summary(r)
    path=Path(args.output_json); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8"); print("Saved:",path)


if __name__ == "__main__":
    main()
