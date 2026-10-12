# -*- coding: utf-8 -*-
"""Validation-only audit of common/unique evidence for A2 gate errors."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (  # noqa: E402
    build_h6_sdcr_damageformer,
)
from bda_benchmark.script.standard_ML.audit_CCU_local_correspondence import (  # noqa: E402
    binary_auroc,
)
from bda_benchmark.script.standard_ML.h12_dgb_runtime import binary_auprc  # noqa: E402
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import set_seed  # noqa: E402


def event_name(value: object) -> str:
    return re.sub(r"_\d+$", "", str(value))


def append_values(
    store: Dict[str, Dict[str, List[np.ndarray]]],
    signal: str,
    event: str,
    score: torch.Tensor,
    label: torch.Tensor,
) -> None:
    eligible = label >= 0
    if not eligible.any():
        return
    store[signal][event].append(score[eligible].detach().float().cpu().numpy())
    store[signal][event + "::__labels__"].append(
        label[eligible].detach().to(torch.uint8).cpu().numpy()
    )


def concatenate(items: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(items) if items else np.empty((0,), dtype=np.float32)


def summarize_signal(events: Mapping[str, List[np.ndarray]]) -> Dict[str, object]:
    names = sorted(key for key in events if not key.endswith("::__labels__"))
    all_scores = concatenate([array for name in names for array in events[name]])
    all_labels = concatenate(
        [array for name in names for array in events[name + "::__labels__"]]
    ).astype(np.uint8)
    raw_auc = binary_auroc(all_scores, all_labels)
    if raw_auc is None:
        return {"global_auroc": None, "events": {}, "positive": 0, "negative": 0}
    direction = 1.0 if raw_auc >= 0.5 else -1.0
    oriented = direction * all_scores
    global_auc = binary_auroc(oriented, all_labels)
    global_ap = binary_auprc(oriented, all_labels)
    prevalence = float(all_labels.mean()) if all_labels.size else 0.0
    event_results = {}
    event_aucs = []
    for name in names:
        score = direction * concatenate(events[name])
        label = concatenate(events[name + "::__labels__"]).astype(np.uint8)
        auc = binary_auroc(score, label)
        ap = binary_auprc(score, label)
        if auc is not None:
            event_aucs.append(float(auc))
        event_results[name] = {
            "auroc": auc,
            "auprc": ap,
            "positive": int((label == 1).sum()),
            "negative": int((label == 0).sum()),
        }
    return {
        "direction": direction,
        "global_auroc": global_auc,
        "global_auprc": global_ap,
        "positive_prevalence": prevalence,
        "auprc_lift": (float(global_ap) / prevalence) if global_ap and prevalence else None,
        "event_macro_auroc": float(np.mean(event_aucs)) if event_aucs else None,
        "events_above_chance": sum(value > 0.5 for value in event_aucs),
        "comparable_events": len(event_aucs),
        "positive": int((all_labels == 1).sum()),
        "negative": int((all_labels == 0).sum()),
        "events": event_results,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("H14 unique-evidence correction audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant", "")).lower() != "a2":
        raise RuntimeError("H14 audit requires the locked H6-A2 checkpoint")
    model = build_h6_sdcr_damageformer("a2", pretrained_rgb=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.data_list_path,
        args.crop_size,
        split="val",
        variant="h3",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    store: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    processed = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        rgb, sar, _location, damage, names = batch
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        outputs = model(rgb, sar, return_dict=True)
        base_prediction = outputs["output_dam"].argmax(1)
        for stage in ("stage2", "stage3"):
            rc = outputs[f"{stage}_rgb_common"].float()
            sc = outputs[f"{stage}_sar_common"].float()
            ru = outputs[f"{stage}_rgb_unique"].float()
            su = outputs[f"{stage}_sar_unique"].float()
            size = rc.shape[-2:]
            target = F.interpolate(
                damage.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1).long()
            prediction = F.interpolate(
                base_prediction.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1).long()
            positive = (target == 2) & (prediction == 1)
            negative = (target == 1) & (prediction == 1)
            label = torch.full_like(target, -1)
            label[negative] = 0
            label[positive] = 1
            signals = {
                f"{stage}_common_gap": (rc - sc).abs().mean(1),
                f"{stage}_sar_unique": su.abs().mean(1),
                f"{stage}_rgb_unique": ru.abs().mean(1),
                f"{stage}_unique_energy": 0.5 * (
                    ru.abs().mean(1) + su.abs().mean(1)
                ),
            }
            for sample_index, name in enumerate(names):
                event = event_name(name)
                for signal, score in signals.items():
                    append_values(
                        store,
                        signal,
                        event,
                        score[sample_index],
                        label[sample_index],
                    )
        processed += int(rgb.shape[0])
        if processed % 50 == 0:
            print(f"H14 evidence audit: {processed}/{len(dataset)}", flush=True)

    summaries = {signal: summarize_signal(events) for signal, events in store.items()}
    ranked = sorted(
        summaries.items(),
        key=lambda item: float(item[1].get("event_macro_auroc") or 0.0),
        reverse=True,
    )
    best_name, best = ranked[0] if ranked else ("none", {})
    decision = "STOP_UNIQUE_EVIDENCE"
    if (
        float(best.get("global_auroc") or 0.0) >= 0.60
        and float(best.get("event_macro_auroc") or 0.0) >= 0.58
        and int(best.get("events_above_chance") or 0) >= 9
        and float(best.get("auprc_lift") or 0.0) >= 1.5
    ):
        decision = "PROCEED_TO_U1"
    payload = {
        "protocol": "H14_UWDC_unique_evidence_audit_v1",
        "processed_images": processed,
        "positive_definition": "GT Damaged and H6-A2 predicts Intact",
        "negative_definition": "GT Intact and H6-A2 predicts Intact",
        "signals": summaries,
        "best_signal": best_name,
        "decision": decision,
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=" * 88)
    print(f"Best signal: {best_name}")
    print(
        "global AUROC={:.4f} | event macro={:.4f} | AUPRC lift={:.2f} | events={}/{}".format(
            float(best.get("global_auroc") or 0.0),
            float(best.get("event_macro_auroc") or 0.0),
            float(best.get("auprc_lift") or 0.0),
            int(best.get("events_above_chance") or 0),
            int(best.get("comparable_events") or 0),
        )
    )
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
