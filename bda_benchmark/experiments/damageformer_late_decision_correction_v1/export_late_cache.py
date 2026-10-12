#!/usr/bin/env python3
"""Export all predicted-building evidence plus exact pixel-CM contributions.

Unlike the earlier diagnostic cache, this exporter NEVER uses GT purity to
remove validation candidates. GT labels/purity are attached only for training
or retrospective audit. Thus late-decision candidate selection can be done
without GT leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    load_a1_parent, load_v2_checkpoint, parse_batch, seed_worker,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import (
    DamageFormerBuildingInternalChangePartitionV2,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import read_list
from bda_benchmark.experiments.damageformer_causal_research_chain_v1.cache_features import building_feature_from_regions

from .late_core import confusion4, event_name, metrics_from_cm, sha256_file


def main() -> None:
    p = argparse.ArgumentParser("Late-decision all-building cache exporter")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--list_file", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", default="p_sc_residual")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min_region_mass", type=float, default=0.50)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--ignore_index", type=int, default=255)
    args = p.parse_args()

    if "test" in Path(args.list_file).name.lower() or "test" in Path(args.output).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stage = str(payload.get("stage", "change"))
    if stage != "change":
        raise RuntimeError(f"late-decision evidence must come from frozen Stage-1(change), got {stage}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingInternalChangePartitionV2(args.variant).to(device)
    parent_report = load_a1_parent(model, args.a1_ckpt)
    ckpt_report = load_v2_checkpoint(model, args.checkpoint)
    model.configure_stage("change")
    model.eval()

    ds = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.list_file), args.crop_size, type="val"
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
    )

    features: List[torch.Tensor] = []
    labels: List[int] = []
    purities: List[float] = []
    names: List[str] = []
    events: List[str] = []
    pred_building_ids: List[int] = []
    base_probs: List[torch.Tensor] = []
    uncertainty: List[float] = []
    peak_mean_delta: List[float] = []
    region_count: List[int] = []
    building_pixel_cm: List[torch.Tensor] = []
    all_region_tokens: List[torch.Tensor] = []
    all_region_critical: List[torch.Tensor] = []
    all_region_logits: List[torch.Tensor] = []
    region_offsets = [0]
    base_pixel_cm = torch.zeros(4, 4, dtype=torch.long)
    images = 0
    skipped_empty_region = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            pre, post, _, damage, batch_names = parse_batch(batch, device)
            _, output_damage, aux = model(pre, post, return_aux=True)
            pred_full = output_damage.argmax(1)
            if damage.ndim == 4 and damage.shape[1] == 1:
                damage_full = damage[:, 0]
            else:
                damage_full = damage
            base_pixel_cm += confusion4(pred_full, damage_full, args.ignore_index)

            b_ids = aux["building_ids"]
            h, w = b_ids.shape[-2:]
            H, W = damage_full.shape[-2:]
            b_ids_full = F.interpolate(
                b_ids[:, None].float(), size=(H, W), mode="nearest"
            )[:, 0].long()
            gt_small = F.interpolate(
                damage_full[:, None].float(), size=(h, w), mode="nearest"
            )[:, 0].long()

            # Base building probs are the frozen A1 parent probabilities at Stage-1 feature resolution.
            probs3 = aux["parent_damage_probability"][:, 1:4].float()
            owners = aux["token_building_ids"]
            mass = aux["region_mass"]
            region_tok = aux["region_tokens"].float()
            rel_mean = aux["relation_token_mean"].float()
            rel_peak = aux["relation_token_peak"].float()
            rel_std = aux["relation_token_std"].float()
            region_logits = aux["region_logits"].float()

            for bi in range(pre.shape[0]):
                sample_name = str(batch_names[bi])
                for bid_raw in torch.unique(b_ids[bi]).tolist():
                    bid = int(bid_raw)
                    if bid <= 0:
                        continue
                    pmask = b_ids[bi] == bid
                    y = gt_small[bi][pmask]
                    y = y[(y >= 1) & (y <= 3)]
                    if y.numel() == 0:
                        dom, purity = -1, 0.0
                    else:
                        counts = torch.stack([(y == c).sum() for c in (1, 2, 3)]).float()
                        dom = int(counts.argmax())  # 0/1/2 = I/D/X
                        purity = float(counts[dom] / counts.sum().clamp_min(1.0))

                    tmask = (owners[bi] == bid) & (mass[bi] >= args.min_region_mass)
                    tidx = torch.nonzero(tmask, as_tuple=False).flatten()
                    if tidx.numel() == 0:
                        skipped_empty_region += 1
                        continue
                    tok = region_tok[bi, tidx]
                    critical = (
                        (rel_peak[bi, tidx] - rel_mean[bi, tidx]).abs().mean(-1)
                        + rel_std[bi, tidx].abs().mean(-1)
                    )
                    feat = building_feature_from_regions(tok, critical)
                    bp = probs3[bi, :, pmask].mean(1)
                    bp = bp / bp.sum().clamp_min(1e-8)

                    full_mask = b_ids_full[bi] == bid
                    bcm = confusion4(
                        pred_full[bi][full_mask], damage_full[bi][full_mask], args.ignore_index
                    )

                    features.append(feat.cpu().half())
                    labels.append(dom)
                    purities.append(purity)
                    names.append(sample_name)
                    events.append(event_name(sample_name))
                    pred_building_ids.append(bid)
                    base_probs.append(bp.cpu().half())
                    uncertainty.append(float(1.0 - bp.max()))
                    peak_mean_delta.append(float((rel_peak[bi, tidx] - rel_mean[bi, tidx]).abs().mean()))
                    region_count.append(int(tidx.numel()))
                    building_pixel_cm.append(bcm)
                    all_region_tokens.append(tok.cpu().half())
                    all_region_critical.append(critical.cpu().half())
                    all_region_logits.append(region_logits[bi, tidx].cpu().half())
                    region_offsets.append(region_offsets[-1] + int(tidx.numel()))
            images += int(pre.shape[0])

    if not features:
        raise RuntimeError("no predicted-building records exported")

    cache: Dict[str, object] = {
        "protocol": "DAMAGEFORMER_LATE_DECISION_CORRECTION_V1_CACHE",
        "split": args.split,
        "variant": args.variant,
        "stage": "change",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "a1_checkpoint": str(args.a1_ckpt),
        "a1_checkpoint_sha256": sha256_file(args.a1_ckpt),
        "parent_report": parent_report,
        "checkpoint_report": ckpt_report,
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_purity": torch.tensor(purities, dtype=torch.float32),
        "names": names,
        "events": events,
        "pred_building_ids": torch.tensor(pred_building_ids, dtype=torch.long),
        "base_probs": torch.stack(base_probs),
        "base_pred": torch.stack(base_probs).argmax(1),
        "uncertainty": torch.tensor(uncertainty, dtype=torch.float32),
        "peak_mean_delta": torch.tensor(peak_mean_delta, dtype=torch.float32),
        "region_count": torch.tensor(region_count, dtype=torch.long),
        "region_offsets": torch.tensor(region_offsets, dtype=torch.long),
        "region_tokens": torch.cat(all_region_tokens, dim=0),
        "region_critical": torch.cat(all_region_critical, dim=0),
        "region_logits": torch.cat(all_region_logits, dim=0),
        "base_pixel_cm": base_pixel_cm,
        "building_pixel_cm": torch.stack(building_pixel_cm),
        "feature_definition": "concat(region_mean, region_std, top1_critical_region, top2_critical_region)",
        "critical_definition": "mean_abs(relation_peak-relation_mean)+mean_abs(relation_std)",
        "validation_candidate_filter_uses_gt": False,
        "gt_used_in_forward_partition": False,
        "processed_images": images,
        "building_records": len(features),
        "skipped_empty_region": skipped_empty_region,
        "test_accessed": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    summary = {
        "output": str(out), "split": args.split, "images": images,
        "buildings": len(features), "regions": int(region_offsets[-1]),
        "source_step": ckpt_report.get("checkpoint_step"),
        "source_sha256": cache["source_checkpoint_sha256"],
        "base_pixel_metrics": metrics_from_cm(base_pixel_cm),
        "label_counts_I_D_X_unknown": [
            int((cache["labels"] == 0).sum()), int((cache["labels"] == 1).sum()),
            int((cache["labels"] == 2).sum()), int((cache["labels"] < 0).sum()),
        ],
        "test_accessed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
