from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.event_balanced_sampling import event_from_sample_name
from bda_benchmark.model.DamageFormer_SafeAO_V2 import DamageFormerSafeAOV2
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, parse_batch


def rank_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    score, label = score.double().flatten(), label.bool().flatten()
    positive, negative = int(label.sum()), int((~label).sum())
    if not positive or not negative:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=score.dtype)
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(score: torch.Tensor, label: torch.Tensor) -> float:
    order = torch.argsort(score, descending=True)
    y = label.bool().flatten()[order].double()
    if not y.sum():
        return float("nan")
    precision = y.cumsum(0) / torch.arange(1, y.numel() + 1, dtype=torch.double)
    return float((precision * y).sum() / y.sum())


@torch.no_grad()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Audit whether an existing AO score identifies P3 affected misses")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--maximum-per-image", type=int, default=4096)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("protocol") != "SPI_SAFE_AFFECTED_MODEL_TRAIN_ONLY_v2":
        raise RuntimeError("D0 expects an existing Safe AO V2 checkpoint")
    model = DamageFormerSafeAOV2(saved["variant"])
    model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = build_dataset(args.dataset_path, args.list_path, 1024, "test")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    by_event: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for index, batch in enumerate(loader, 1):
        pre, post, location, damage, names = parse_batch(batch, device)
        _, _, aux = model(pre, post, return_aux=True)
        score = F.interpolate(aux["occurrence_logits"], damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        parent = F.interpolate(aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False)
        source = parent.argmax(1)
        loc = F.interpolate(aux["p_building"], damage.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        candidate = (damage != 255) & (source <= 1) & (loc >= 0.5)
        label = damage >= 2
        nonaffected = parent[:, 0:2].amax(1)
        affected = parent[:, 2:4].amax(1)
        required = (nonaffected - affected).clamp_min(0)
        flat = torch.nonzero(candidate.flatten(), as_tuple=False).flatten()
        if flat.numel() > args.maximum_per_image:
            pick = torch.linspace(0, flat.numel() - 1, args.maximum_per_image, device=flat.device).long()
            flat = flat[pick]
        event = event_from_sample_name(str(names[0]))
        by_event.setdefault(event, []).append((score.flatten()[flat].cpu(), label.flatten()[flat].cpu(), required.flatten()[flat].cpu()))
        if index % 10 == 0:
            print(f"Candidate D0: {index}/{len(loader)}", flush=True)
    events = {}
    all_score, all_label, all_required = [], [], []
    for event, rows in sorted(by_event.items()):
        score = torch.cat([r[0] for r in rows]); label = torch.cat([r[1] for r in rows]); required = torch.cat([r[2] for r in rows])
        auc = rank_auc(score, label)
        events[event] = {"auroc": auc, "positive": int(label.sum()), "negative": int((~label).sum())}
        all_score.append(score); all_label.append(label); all_required.append(required)
    score, label, required = torch.cat(all_score), torch.cat(all_label), torch.cat(all_required)
    prevalence = float(label.float().mean())
    top = {}
    for fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
        count = max(1, round(score.numel() * fraction))
        chosen = torch.topk(score, count).indices
        precision = float(label[chosen].float().mean())
        top[f"top_{fraction:g}"] = {
            "count": count, "precision": precision, "lift": precision / max(prevalence, 1e-12),
            "mean_required_shift": float(required[chosen].mean()),
        }
    event_values = [row["auroc"] for row in events.values() if row["auroc"] == row["auroc"]]
    payload = {
        "protocol": "SPI_CANDIDATE_SCORE_AUDIT_D0_v1",
        "source_variant": saved["variant"], "candidate_definition": "valid & P3 in {background,intact} & p_building>=0.5",
        "samples": int(score.numel()), "positive": int(label.sum()), "negative": int((~label).sum()),
        "prevalence": prevalence, "auroc": rank_auc(score, label), "average_precision": average_precision(score, label),
        "event_macro_auroc": sum(event_values) / max(len(event_values), 1),
        "correct_events": sum(v > 0.5 for v in event_values), "event_total": len(event_values),
        "top_fraction": top, "events": events,
        "checks": {}, "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    payload["checks"] = {
        "auroc_ge_0p65": payload["auroc"] >= 0.65,
        "event_macro_ge_0p60": payload["event_macro_auroc"] >= 0.60,
        "correct_events_ge_10": payload["correct_events"] >= 10,
        "top_1pct_lift_ge_2": payload["top_fraction"]["top_0.01"]["lift"] >= 2.0,
    }
    payload["decision"] = "ADMIT_MARGIN_AWARE_AO_V21" if all(payload["checks"].values()) else "STOP_SHALLOW_AO_SCORE_NO_CANDIDATE_SIGNAL"
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
