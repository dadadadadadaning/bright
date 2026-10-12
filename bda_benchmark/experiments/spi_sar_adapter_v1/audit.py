from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    checkpoint_state,
    fast_histogram,
    file_sha256,
    metrics_from_histogram,
    p3_sar_view,
)
from bda_benchmark.script.standard_ML.audit_H9_building_oracle import H9RGBSARInferenceDataset
from bda_benchmark.script.standard_ML.spi_t1.audit import load_semantic_target

from .model import SARFeatureAdapter, forward_frozen_p3_with_sar_adapter


PROTOCOL = "SPI_SAR_ADAPTER_INTERNAL_CALIBRATION_v1"


def paired_bootstrap(rows, iterations: int, seed: int):
    grouped = defaultdict(list)
    for event, base, candidate in rows:
        grouped[event].append((base.numpy(), candidate.numpy()))
    rng = np.random.default_rng(seed)
    values = np.zeros(iterations, dtype=np.float64)
    for iteration in range(iterations):
        base_total = np.zeros((4, 4), dtype=np.float64)
        candidate_total = np.zeros((4, 4), dtype=np.float64)
        for event_rows in grouped.values():
            indices = rng.integers(0, len(event_rows), size=len(event_rows))
            for index in indices:
                base, candidate = event_rows[int(index)]
                base_total += base
                candidate_total += candidate
        def miou(hist):
            intersection = np.diag(hist)
            union = hist.sum(0) + hist.sum(1) - intersection
            return float((intersection / np.maximum(union, 1.0))[1:4].mean())
        values[iteration] = 100.0 * (miou(candidate_total) - miou(base_total))
    return {
        "mean_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": iterations,
    }


@torch.inference_mode()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Audit one SAR adapter on internal calibration")
    parser.add_argument("--variant", choices=("fsa", "rcm"), required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=291)
    parser.add_argument("--expected-steps", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    base_sha = file_sha256(args.base_checkpoint)
    base_saved = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(base_saved), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)

    adapter_saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    if adapter_saved.get("variant") != args.variant:
        raise RuntimeError("adapter variant mismatch")
    if adapter_saved.get("base_checkpoint_sha256") != base_sha:
        raise RuntimeError("adapter parent checkpoint mismatch")
    if int(adapter_saved.get("optimizer_step", 0)) != args.expected_steps:
        raise RuntimeError("adapter optimizer-step mismatch")
    adapter = SARFeatureAdapter(args.variant).to(device)
    adapter.load_state_dict(adapter_saved["adapter_state"], strict=True)
    adapter.eval().requires_grad_(False)

    dataset = H9RGBSARInferenceDataset(args.dataset_path, args.list_path, args.sar_stats_json)
    if len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    totals = {
        "p3": torch.zeros(4, 4, dtype=torch.float64),
        args.variant: torch.zeros(4, 4, dtype=torch.float64),
    }
    event_totals = defaultdict(
        lambda: {
            "p3": torch.zeros(4, 4, dtype=torch.float64),
            args.variant: torch.zeros(4, 4, dtype=torch.float64),
        }
    )
    rows = []
    started = time.perf_counter()
    for index, (rgb, sar, names) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        sar = p3_sar_view(sar.to(device, non_blocking=True))
        _, candidate, auxiliary = forward_frozen_p3_with_sar_adapter(p3, adapter, rgb, sar)
        base = auxiliary["base_output_damage"]
        name = str(names[0])
        target = load_semantic_target(args.dataset_path, name)
        event = name.rsplit("_", 1)[0]
        base_hist = fast_histogram(base[0].float().argmax(0).cpu(), target)
        candidate_hist = fast_histogram(candidate[0].float().argmax(0).cpu(), target)
        totals["p3"] += base_hist
        totals[args.variant] += candidate_hist
        event_totals[event]["p3"] += base_hist
        event_totals[event][args.variant] += candidate_hist
        rows.append((event, base_hist, candidate_hist))
        if index % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"SAR {args.variant} audit: {index}/{len(dataset)} | {index/max(elapsed, 1e-9):.2f} img/s", flush=True)

    reports = {name: metrics_from_histogram(hist) for name, hist in totals.items()}
    events = {
        event: {name: metrics_from_histogram(hist) for name, hist in table.items()}
        for event, table in sorted(event_totals.items())
    }
    gain_miou = 100.0 * (reports[args.variant]["valid_miou"] - reports["p3"]["valid_miou"])
    gain_damaged_f1 = 100.0 * (reports[args.variant]["damaged_f1"] - reports["p3"]["damaged_f1"])
    foreground = {
        key: 100.0 * (reports[args.variant][key] - reports["p3"][key])
        for key in ("intact_iou", "damaged_iou", "destroyed_iou")
    }
    event_deltas = {
        event: 100.0 * (table[args.variant]["valid_miou"] - table["p3"]["valid_miou"])
        for event, table in events.items()
    }
    bootstrap = paired_bootstrap(rows, args.bootstrap_iterations, args.seed)
    minimum_events = min(10, math.ceil(0.70 * len(events)))
    checks = {
        "miou_gain_ge_0_30pp": gain_miou >= 0.30,
        "damaged_f1_gain_ge_0_50pp": gain_damaged_f1 >= 0.50,
        "no_foreground_iou_drop_gt_0_50pp": min(foreground.values()) >= -0.50,
        "event_nonnegative_ge_10": sum(value >= 0 for value in event_deltas.values()) >= minimum_events,
        "bootstrap_ci_low_gt_zero": bootstrap["ci95_low_pp"] > 0,
    }
    decision = "PROCEED_TO_FULL_CONFIRMATION" if all(checks.values()) else "STOP_SAR_ADAPTER_AFTER_PILOT"
    payload = {
        "protocol": PROTOCOL,
        "variant": args.variant,
        "processed_images": len(dataset),
        "base_checkpoint_sha256": base_sha,
        "adapter_checkpoint": args.adapter_checkpoint,
        "optimizer_steps": args.expected_steps,
        "reports": reports,
        "event_reports": events,
        "gains": {
            "miou_pp": gain_miou,
            "damaged_f1_pp": gain_damaged_f1,
            "foreground_iou_pp": foreground,
            "event_miou_pp": event_deltas,
        },
        "bootstrap": bootstrap,
        "checks": checks,
        "decision": decision,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 110)
    print(
        f"p3 mIoU={100*reports['p3']['valid_miou']:.4f} D-F1={100*reports['p3']['damaged_f1']:.4f}"
    )
    print(
        f"{args.variant} mIoU={100*reports[args.variant]['valid_miou']:.4f} "
        f"gain={gain_miou:+.4f}pp D-F1={100*reports[args.variant]['damaged_f1']:.4f}"
    )
    print(f"Checks: {checks}")
    print(f"Decision: {decision}")
    print(f"Saved: {output}")
    print("Official instances/test accessed: NEVER / NEVER")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
