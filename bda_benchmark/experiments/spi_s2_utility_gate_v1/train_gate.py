from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.experiments.spi_selective_context_v1.core import SelectiveContextConfig
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import p3_sar_view
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .core import (
    UtilityGate,
    balanced_selector_bce,
    fixed_s2_candidate_from_labels,
    predicted_building_label_map,
    selector_targets,
    utility_features,
)
from .runtime import load_frozen_p3


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train only the fixed-S2 utility gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, provenance = load_frozen_p3(args.checkpoint, device)
    gate = UtilityGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gate_last.pth"
    step = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("p3_sha256") != provenance["p3_sha256"]:
            raise RuntimeError("existing utility-gate checkpoint P3 provenance mismatch")
        gate.load_state_dict(saved["gate_state"], strict=True)
        step = int(saved["optimizer_step"])
        if step < args.optimizer_steps:
            optimizer.load_state_dict(saved["optimizer_state"])
        else:
            print(f"Utility gate already complete at step={step}")
            return 0

    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.train_list, args.sar_stats_json)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
                        generator=generator)
    config = SelectiveContextConfig()
    manifest = {
        "protocol": "SPI_S2_UTILITY_GATE_TRAIN_ONLY_v1",
        **provenance,
        "s2_config": config.__dict__,
        "train_list": str(Path(args.train_list).resolve()),
        "train_images": len(dataset),
        "args": vars(args),
        "validation_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("=" * 100)
    print(f"Fixed-S2 utility gate | trainable={sum(p.numel() for p in gate.parameters()):,} | P3 frozen")
    print(f"steps={args.optimizer_steps} accumulation={args.accumulation_steps} full-resolution batch=1")
    print("Validation/official instances/test accessed: NEVER / NEVER / NEVER")
    print("=" * 100, flush=True)
    gate.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.perf_counter()
    while step < args.optimizer_steps:
        for rgb, sar, names in loader:
            rgb = rgb.to(device, non_blocking=True)
            sar = p3_sar_view(sar.to(device, non_blocking=True))
            with torch.no_grad():
                loc_logits, damage_logits = model(rgb, sar)
                base_cpu = damage_logits[0].float().softmax(0).cpu()
                localization = loc_logits[0].argmax(0).cpu()
                region_map = predicted_building_label_map(localization, args.minimum_area)
                candidate = fixed_s2_candidate_from_labels(base_cpu, region_map, config)
            base = base_cpu[None].to(device)
            candidate_prob = candidate.probabilities[None].to(device)
            support = candidate.support[None, None].to(device)
            region_features = candidate.region_features[None].to(device)
            inputs, changed = utility_features(base, candidate_prob, support, region_features)
            gate_values = gate(inputs, changed)
            target = load_semantic_target(args.dataset_path, str(names[0]))[None].to(device)
            selector_target, decisive, positive, negative = selector_targets(
                base.argmax(1), candidate_prob.argmax(1), target
            )
            loss = balanced_selector_bce(gate_values, selector_target, decisive)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite utility loss at step={step}")
            (loss / args.accumulation_steps).backward()
            micro_step += 1
            if micro_step < args.accumulation_steps:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            step += 1
            if step == 1 or step % args.log_every == 0:
                row = {
                    "optimizer_step": step,
                    "loss": float(loss.detach()),
                    "gate_on_changed_mean": float(
                        (gate_values.sum() / changed.sum().clamp_min(1)).detach()
                    ),
                    "changed": int(changed.sum()),
                    "positive": int(positive.sum()),
                    "negative": int(negative.sum()),
                    "decisive": int(decisive.sum()),
                    "gradient_norm_before_clip": float(grad_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.optimizer_steps:
                atomic_save({
                    **manifest,
                    "optimizer_step": step,
                    "gate_state": gate.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, checkpoint_path)
            if step >= args.optimizer_steps:
                break
    print(f"Saved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
