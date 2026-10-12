#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
for value in (str(ROOT), str(ROOT / "bda_benchmark"), str(ROOT / "bda_benchmark/script/standard_ML")):
    if value not in sys.path:
        sys.path.insert(0, value)

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import augment_metrics
from bda_benchmark.experiments.p3_damageformer_ablation_v1.train import format_metrics
from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import (
    event_name, parse_batch, read_list, sha256_file, write_json,
)
from bda_benchmark.model.DamageFormer_BuildingPrototypeMemory_V1 import DamageFormerBuildingPrototypeMemoryV1
from train_S5_S8_GPU import fast_hist, metrics_from_hist


@torch.inference_mode()
def evaluate_with_diagnostics(model, loader, device):
    model.eval()
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    dam_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    event_hists = {}
    diagnostic_sums = {"tokens": 0, "delta_abs": 0.0, "modified": 0, "high_conf_tokens": 0, "high_conf_modified": 0, "memory_confidence": 0.0}
    processed = 0
    for batch in loader:
        pre, post, location, damage, names = parse_batch(batch, device)
        output_loc, output_damage, aux = model(pre, post, names=names, return_aux=True)
        pred_loc, pred_damage = output_loc.argmax(1), output_damage.argmax(1)
        loc_hist += fast_hist(pred_loc, location, 2, 255)
        dam_hist += fast_hist(pred_damage, damage, 4, 255)
        for sample_index, name in enumerate(names):
            key = event_name(str(name))
            hist = fast_hist(pred_damage[sample_index:sample_index + 1], damage[sample_index:sample_index + 1], 4, 255)
            event_hists[key] = event_hists.get(key, torch.zeros_like(hist)) + hist
        delta = aux["delta"].abs()
        parent_conf = torch.softmax(aux["parent_logits_feature"].float(), dim=1).amax(dim=1, keepdim=True)
        high_conf = parent_conf >= 0.8
        modified = delta >= 0.01
        diagnostic_sums["tokens"] += int(delta.numel())
        diagnostic_sums["delta_abs"] += float(delta.sum().item())
        diagnostic_sums["modified"] += int(modified.sum().item())
        diagnostic_sums["high_conf_tokens"] += int(high_conf.sum().item())
        diagnostic_sums["high_conf_modified"] += int((high_conf & modified).sum().item())
        diagnostic_sums["memory_confidence"] += float(aux["memory_confidence"].sum().item())
        processed += int(pre.shape[0])
    metrics = augment_metrics(metrics_from_hist(loc_hist, dam_hist), loc_hist, dam_hist)
    metrics["intact_to_affected"] = float((dam_hist[1, 2:4].sum() / dam_hist[1].sum().clamp_min(1.0)).item())
    per_event = {}
    for key, hist in event_hists.items():
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - intersection
        per_event[key] = float((intersection / union.clamp_min(1.0)).mean().item())
    metrics["event_count"] = len(per_event)
    metrics["event_mean_miou4"] = sum(per_event.values()) / max(1, len(per_event))
    metrics["event_worst_miou4"] = min(per_event.values()) if per_event else 0.0
    metrics["event_miou4_by_name"] = per_event
    metrics["processed_images"] = processed
    tokens = max(1, diagnostic_sums["tokens"])
    high_conf_tokens = max(1, diagnostic_sums["high_conf_tokens"])
    diagnostics = {
        "delta_abs_mean": diagnostic_sums["delta_abs"] / tokens,
        "modified_token_fraction_at_abs_delta_ge_0_01": diagnostic_sums["modified"] / tokens,
        "high_confidence_modified_fraction": diagnostic_sums["high_conf_modified"] / high_conf_tokens,
        "memory_confidence_mean": diagnostic_sums["memory_confidence"] / tokens,
        "learned_gain": float(torch.tanh(model.gain).item()),
    }
    return metrics, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate frozen prototype-memory variants on validation only")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--parent_ckpt", required=True)
    parser.add_argument("--memory_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in Path(args.val_list).name.lower():
        raise RuntimeError("validation evaluator refuses test-list access")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingPrototypeMemoryV1(args.variant).to(device)
    parent = model.load_parent(args.parent_ckpt)
    memory = model.load_memory(args.memory_path)
    model.freeze_parent()
    checkpoint_step = 0
    if args.variant != "mm0_frozen_a0":
        if not args.checkpoint or not Path(args.checkpoint).is_file():
            raise FileNotFoundError(f"missing trained checkpoint for {args.variant}: {args.checkpoint}")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("variant") != args.variant:
            raise RuntimeError("evaluation checkpoint variant mismatch")
        if saved.get("parent_sha256") != sha256_file(args.parent_ckpt):
            raise RuntimeError("evaluation parent differs from the training parent")
        if saved.get("memory_sha256") != sha256_file(args.memory_path):
            raise RuntimeError("evaluation Memory differs from the training Memory")
        model.head.load_state_dict(saved["head"])
        model.projection.load_state_dict(saved["projection"])
        model.gain.data.copy_(saved["gain"].to(device))
        checkpoint_step = int(saved.get("step", -1))
    model.eval()
    names = read_list(args.val_list)
    dataset = MultimodalDamageAssessmentDatset(args.dataset_path, names, crop_size=640, type="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    memory_sha_before = sha256_file(args.memory_path)
    metrics, diagnostics = evaluate_with_diagnostics(model, loader, device)
    memory_sha_after = sha256_file(args.memory_path)
    if memory_sha_before != memory_sha_after:
        raise RuntimeError("Memory file changed during validation")
    payload = {
        "protocol": "SPI_BUILDING_PROTOTYPE_MEMORY_V1", "variant": args.variant,
        "checkpoint_step": checkpoint_step, "metrics": metrics, "diagnostics": diagnostics,
        "parent": parent, "memory": memory,
        "validation_list": str(Path(args.val_list).resolve()),
        "validation_list_sha256": sha256_file(args.val_list),
        "memory_sha256_before": memory_sha_before, "memory_sha256_after": memory_sha_after,
        "memory_updated_during_validation": False, "model_updated_during_validation": False,
        "test_accessed": False,
    }
    write_json(args.output_json, payload)
    print(format_metrics(metrics))
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
