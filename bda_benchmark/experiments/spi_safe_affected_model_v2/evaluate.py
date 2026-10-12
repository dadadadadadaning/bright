from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_SafeAO_V2 import DamageFormerSafeAOV2
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset, fast_hist, metrics_from_hist, parse_batch,
)
from bda_benchmark.dataset.event_balanced_sampling import event_from_sample_name


def shift_without_wrap(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    output = torch.zeros_like(x)
    height, width = x.shape[-2:]
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    output[..., target_y0:target_y1, target_x0:target_x1] = x[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return output


def valid_miou(histogram: torch.Tensor) -> float:
    hist = histogram.double()
    intersection = torch.diag(hist)
    union = hist.sum(0) + hist.sum(1) - intersection
    return float((intersection[1:4] / union[1:4].clamp_min(1)).mean())


def bootstrap_gain(parent_rows, variant_rows, iterations: int, seed: int):
    if not iterations:
        return {"iterations": 0}
    generator = np.random.default_rng(seed)
    count = len(parent_rows)
    values = []
    for _ in range(iterations):
        indices = generator.integers(0, count, size=count)
        parent = torch.stack([parent_rows[index] for index in indices]).sum(0)
        variant = torch.stack([variant_rows[index] for index in indices]).sum(0)
        values.append(100.0 * (valid_miou(variant) - valid_miou(parent)))
    return {
        "mean_pp": float(np.mean(values)),
        "ci95_low_pp": float(np.quantile(values, 0.025)),
        "ci95_high_pp": float(np.quantile(values, 0.975)),
        "iterations": iterations,
    }


@torch.no_grad()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Internal Safe AO V2 evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--shift-x", type=int, default=0)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("protocol") != "SPI_SAFE_AFFECTED_MODEL_TRAIN_ONLY_v2":
        raise RuntimeError("checkpoint protocol mismatch")
    model = DamageFormerSafeAOV2(saved["variant"])
    model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = build_dataset(args.dataset_path, args.list_path, 1024, "test")
    if args.expected_images and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    parent_hist = torch.zeros(4, 4, dtype=torch.float64, device=device)
    variant_hist = torch.zeros_like(parent_hist)
    parent_rows, variant_rows = [], []
    events = {}
    severity_error = 0.0
    confidence_sum = confidence_count = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        pre, post, location, damage, names = parse_batch(batch, device)
        post = shift_without_wrap(post, args.shift_x, args.shift_y)
        output_loc, output, aux = model(pre, post, return_aux=True)
        parent = F.interpolate(
            aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False
        )
        loc_hist += fast_hist(output_loc.argmax(1), location, 2, 255)
        parent_hist += fast_hist(parent.argmax(1), damage, 4, 255)
        variant_hist += fast_hist(output.argmax(1), damage, 4, 255)
        severity_error = max(
            severity_error,
            float(((output[:, 3] - output[:, 2]) - (parent[:, 3] - parent[:, 2])).abs().max()),
        )
        if "match_confidence" in aux:
            confidence_sum += float(aux["match_confidence"].double().sum())
            confidence_count += int(aux["match_confidence"].numel())
        for sample in range(damage.shape[0]):
            ph = fast_hist(parent[sample : sample + 1].argmax(1), damage[sample : sample + 1], 4, 255).cpu()
            vh = fast_hist(output[sample : sample + 1].argmax(1), damage[sample : sample + 1], 4, 255).cpu()
            parent_rows.append(ph)
            variant_rows.append(vh)
            event = event_from_sample_name(str(names[sample]))
            if event not in events:
                events[event] = [torch.zeros_like(ph), torch.zeros_like(vh)]
            events[event][0] += ph
            events[event][1] += vh
        if batch_index % 10 == 0:
            print(f"Safe AO evaluate: {batch_index}/{len(loader)}", flush=True)
    parent_metrics = metrics_from_hist(loc_hist, parent_hist)
    metrics = metrics_from_hist(loc_hist, variant_hist)
    event_results = {}
    nonnegative = 0
    for event, (parent_event, variant_event) in sorted(events.items()):
        gain = 100.0 * (valid_miou(variant_event) - valid_miou(parent_event))
        event_results[event] = {"gain_valid_miou_pp": gain}
        nonnegative += int(gain >= 0)
    gain = 100.0 * (metrics["valid_miou"] - parent_metrics["valid_miou"])
    damaged_change = 100.0 * (metrics["damaged_f1"] - parent_metrics["damaged_f1"])
    bootstrap = bootstrap_gain(parent_rows, variant_rows, args.bootstrap_iterations, args.seed)
    checks = {
        "gain_ge_0p30pp": gain >= 0.30,
        "damaged_f1_not_below_parent": damaged_change >= 0.0,
        "nonnegative_events_ge_10": nonnegative >= 10,
        "bootstrap_ci_low_gt_zero": bootstrap.get("ci95_low_pp", -1.0) > 0.0,
        "severity_difference_preserved": severity_error <= 1e-5,
        "localization_is_reported": float(loc_hist.sum()) > 0,
    }
    decision = "ADMIT_SAFE_AO_V2_NEXT_STAGE" if all(checks.values()) else "STOP_SAFE_AO_V2_AFTER_PILOT"
    payload = {
        "protocol": "SPI_SAFE_AFFECTED_MODEL_INTERNAL_EVAL_v2",
        "variant": saved["variant"], "shift": {"x": args.shift_x, "y": args.shift_y},
        "parent": parent_metrics, "variant_metrics": metrics,
        "gain_valid_miou_pp": gain, "damaged_f1_change_pp": damaged_change,
        "affected_gain": float(model.affected_residual.gain),
        "mean_match_confidence": confidence_sum / max(confidence_count, 1),
        "maximum_severity_difference_error": severity_error,
        "events": event_results, "nonnegative_events": f"{nonnegative}/{len(events)}",
        "bootstrap": bootstrap, "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
