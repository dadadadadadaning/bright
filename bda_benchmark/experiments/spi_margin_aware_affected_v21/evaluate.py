from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.event_balanced_sampling import event_from_sample_name
from bda_benchmark.model.DamageFormer_MarginAwareAO_V21 import DamageFormerMarginAwareAOV21
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, fast_hist, metrics_from_hist, parse_batch


def miou(hist):
    hist = hist.double(); diagonal = hist.diag(); union = hist.sum(0) + hist.sum(1) - diagonal
    return float((diagonal[1:4] / union[1:4].clamp_min(1)).mean())


def bootstrap(parent_rows, output_rows, iterations, seed):
    rng = np.random.default_rng(seed); values = []; count = len(parent_rows)
    for _ in range(iterations):
        chosen = rng.integers(0, count, count)
        ph = torch.stack([parent_rows[i] for i in chosen]).sum(0)
        oh = torch.stack([output_rows[i] for i in chosen]).sum(0)
        values.append(100 * (miou(oh) - miou(ph)))
    return {"mean_pp": float(np.mean(values)), "ci95_low_pp": float(np.quantile(values, .025)),
            "ci95_high_pp": float(np.quantile(values, .975)), "iterations": iterations}


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser("Evaluate one frozen Margin-Aware AO V2.1 policy")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1); parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("protocol") != "SPI_MARGIN_AWARE_AO_TRAIN_ONLY_v21": raise RuntimeError("checkpoint protocol mismatch")
    model = DamageFormerMarginAwareAOV21(threshold=args.threshold); model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device(args.device); model.to(device).eval(); model.margin_residual.set_threshold(args.threshold)
    dataset = build_dataset(args.dataset_path, args.list_path, 1024, "test")
    if args.expected_images and len(dataset) != args.expected_images: raise RuntimeError("image-count mismatch")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device); ph = torch.zeros(4, 4, dtype=torch.float64, device=device)
    oh = torch.zeros_like(ph); parent_rows=[]; output_rows=[]; events={}; accepted=candidates=0; severity_error=0.
    for index, batch in enumerate(loader, 1):
        pre, post, location, damage, names = parse_batch(batch, device)
        output_loc, output, aux = model(pre, post, return_aux=True)
        parent = F.interpolate(aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False)
        loc_hist += fast_hist(output_loc.argmax(1), location, 2, 255)
        prow=fast_hist(parent.argmax(1), damage, 4, 255); orow=fast_hist(output.argmax(1), damage, 4, 255)
        ph += prow; oh += orow; parent_rows.append(prow.cpu()); output_rows.append(orow.cpu())
        accepted += int(aux["accepted_mask"].sum()); candidates += int(aux["candidate_mask"].sum())
        severity_error=max(severity_error,float(((output[:,3]-output[:,2])-(parent[:,3]-parent[:,2])).abs().max()))
        event=event_from_sample_name(str(names[0])); pair=events.setdefault(event,[torch.zeros_like(prow),torch.zeros_like(orow)])
        pair[0]+=prow; pair[1]+=orow
        if index % 10 == 0: print(f"Margin-Aware AO evaluate: {index}/{len(loader)}", flush=True)
    parent_metrics=metrics_from_hist(loc_hist,ph); metrics=metrics_from_hist(loc_hist,oh)
    gain=100*(metrics["valid_miou"]-parent_metrics["valid_miou"]); dchange=100*(metrics["damaged_f1"]-parent_metrics["damaged_f1"])
    event_rows={e:{"gain_pp":100*(miou(x[1])-miou(x[0]))} for e,x in sorted(events.items())}; nonnegative=sum(x["gain_pp"]>=0 for x in event_rows.values())
    boot=bootstrap(parent_rows,output_rows,args.bootstrap_iterations,args.seed)
    checks={"gain_ge_0p30pp":gain>=.30,"damaged_f1_not_below_parent":dchange>=0,
            "nonnegative_events_ge_10":nonnegative>=10,"bootstrap_ci_low_gt_zero":boot["ci95_low_pp"]>0,
            "severity_preserved":severity_error<=1e-5,"nontrivial_acceptance":0<accepted<candidates}
    payload={"protocol":"SPI_MARGIN_AWARE_AO_INTERNAL_EVAL_v21","threshold":args.threshold,
             "parent":parent_metrics,"variant":metrics,"gain_valid_miou_pp":gain,"damaged_f1_change_pp":dchange,
             "accepted_fraction":accepted/max(candidates,1),"events":event_rows,"nonnegative_events":f"{nonnegative}/{len(events)}",
             "bootstrap":boot,"maximum_severity_difference_error":severity_error,"checks":checks,
             "decision":"ADMIT_MARGIN_AWARE_AO_V21_LOEO" if all(checks.values()) else "STOP_MARGIN_AWARE_AO_V21_AFTER_PILOT",
             "meta_confirm_accessed":False,"val450_accessed":False,"test_accessed":False}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); print(f"Saved: {path}",flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
