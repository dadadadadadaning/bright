from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bda_benchmark.dataset.event_balanced_sampling import event_from_sample_name
from bda_benchmark.model.DamageFormer_AffectedOccurrence_V1 import DamageFormerAffectedOccurrenceV1
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, fast_hist, metrics_from_hist, parse_batch
from . import PROTOCOL
from .spatial import boundary_and_small_masks, local_sar_noise


def valid_miou(histogram: torch.Tensor) -> float:
    hist = histogram.double(); intersection = torch.diag(hist)
    union = hist.sum(0) + hist.sum(1) - intersection
    return float((intersection[1:4] / union[1:4].clamp_min(1)).mean())


def bootstrap(parent_rows, variant_rows, iterations: int, seed: int) -> dict:
    generator = np.random.default_rng(seed); values = []
    for _ in range(iterations):
        index = generator.integers(0, len(parent_rows), size=len(parent_rows))
        parent = torch.stack([parent_rows[i] for i in index]).sum(0)
        variant = torch.stack([variant_rows[i] for i in index]).sum(0)
        values.append(100. * (valid_miou(variant) - valid_miou(parent)))
    return {"iterations": iterations, "mean_pp": float(np.mean(values)), "ci95_low_pp": float(np.quantile(values, .025)), "ci95_high_pp": float(np.quantile(values, .975))}


def recall(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@torch.no_grad()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Evaluate frozen-P3 Affected Occurrence Head")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--small-area", type=float, required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("protocol") != PROTOCOL: raise RuntimeError("checkpoint protocol mismatch")
    model = DamageFormerAffectedOccurrenceV1(saved["variant"]); model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device(args.device); model.to(device).eval()
    dataset = build_dataset(args.dataset_path, args.list_path, 1024, "test")
    if len(dataset) != args.expected_images: raise RuntimeError(f"expected {args.expected_images} images, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)
    parent_hist = torch.zeros(4, 4, dtype=torch.float64, device=device); variant_hist = torch.zeros_like(parent_hist)
    parent_rows, variant_rows = [], []; events = defaultdict(lambda: [torch.zeros(4, 4), torch.zeros(4, 4)])
    areas = {name: [0, 0, 0, 0] for name in ("all_affected", "ring", "small_damage", "high_sar_noise")}
    severity_error = 0.; quality_sum = quality_count = 0.
    for index, batch in enumerate(loader, start=1):
        pre, post, location, damage, names = parse_batch(batch, device)
        output_loc, output, aux = model(pre, post, return_aux=True)
        parent = F.interpolate(aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False)
        parent_prediction, prediction = parent.argmax(1), output.argmax(1)
        loc_hist += fast_hist(output_loc.argmax(1), location, 2, 255)
        parent_hist += fast_hist(parent_prediction, damage, 4, 255); variant_hist += fast_hist(prediction, damage, 4, 255)
        severity_error = max(severity_error, float(((output[:, 3] - output[:, 2]) - (parent[:, 3] - parent[:, 2])).abs().max()))
        quality_sum += float(aux["sar_quality"].double().sum()); quality_count += int(aux["sar_quality"].numel())
        boundary, small = boundary_and_small_masks(location, damage, args.small_area)
        affected = (damage >= 2) & (damage != 255)
        noise = local_sar_noise(post)
        high_noise = noise >= torch.quantile(noise.flatten(1), .75, dim=1)[:, None, None, None]
        masks = {"all_affected": affected, "ring": affected & boundary, "small_damage": affected & small,
                 "high_sar_noise": affected & high_noise[:, 0]}
        for name, mask in masks.items():
            areas[name][0] += int(mask.sum()); areas[name][1] += int((mask & (parent_prediction >= 2)).sum()); areas[name][2] += int((mask & (prediction >= 2)).sum())
            areas[name][3] += int((mask & (parent_prediction <= 1) & (prediction >= 2)).sum())
        for sample in range(damage.shape[0]):
            ph = fast_hist(parent_prediction[sample:sample+1], damage[sample:sample+1], 4, 255).cpu()
            vh = fast_hist(prediction[sample:sample+1], damage[sample:sample+1], 4, 255).cpu()
            parent_rows.append(ph); variant_rows.append(vh)
            event = event_from_sample_name(str(names[sample])); events[event][0] += ph; events[event][1] += vh
        if index % 10 == 0 or index == len(loader): print(f"AO-H evaluate: {index}/{len(loader)}", flush=True)
    parent_metrics, metrics = metrics_from_hist(loc_hist, parent_hist), metrics_from_hist(loc_hist, variant_hist)
    event_rows = {}; nonnegative = 0
    for event, (ph, vh) in sorted(events.items()):
        gain = 100. * (valid_miou(vh) - valid_miou(ph)); event_rows[event] = {"gain_valid_miou_pp": gain}; nonnegative += int(gain >= 0)
    diagnostics = {name: {"affected_pixels": values[0], "p3_affected_recall": recall(values[1], values[0]),
                           "variant_affected_recall": recall(values[2], values[0]),
                           "recall_gain_pp": 100. * ((recall(values[2], values[0]) or 0.) - (recall(values[1], values[0]) or 0.)),
                           "newly_recovered_pixels": values[3]} for name, values in areas.items()}
    gain = 100. * (metrics["valid_miou"] - parent_metrics["valid_miou"])
    damaged_gain = 100. * (metrics["damaged_f1"] - parent_metrics["damaged_f1"])
    destroyed_gain = 100. * (metrics["destroyed_f1"] - parent_metrics["destroyed_f1"])
    boot = bootstrap(parent_rows, variant_rows, args.bootstrap_iterations, args.seed)
    checks = {"gain_ge_0p30pp": gain >= .30, "damaged_f1_not_below_parent": damaged_gain >= 0.,
              "destroyed_f1_drop_within_0p10pp": destroyed_gain >= -.10, "nonnegative_events_ge_10": nonnegative >= 10,
              "bootstrap_ci_low_gt_zero": boot["ci95_low_pp"] > 0., "severity_difference_preserved": severity_error <= 1e-5,
              "ring_or_small_recall_improved": diagnostics["ring"]["recall_gain_pp"] > 0. or diagnostics["small_damage"]["recall_gain_pp"] > 0.}
    payload = {"protocol": PROTOCOL + "_INTERNAL_EVAL", "variant": saved["variant"], "parent": parent_metrics,
               "variant_metrics": metrics, "gain_valid_miou_pp": gain, "damaged_f1_gain_pp": damaged_gain,
               "destroyed_f1_gain_pp": destroyed_gain, "affected_diagnostics": diagnostics, "events": event_rows,
               "nonnegative_events": f"{nonnegative}/{len(events)}", "bootstrap": boot, "maximum_severity_difference_error": severity_error,
               "mean_sar_quality": quality_sum / max(quality_count, 1), "checks": checks,
               "decision": "ADMIT_NEXT_ABLATION" if all(checks.values()) else "STOP_THIS_VARIANT",
               "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); print(f"Saved: {path}", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
