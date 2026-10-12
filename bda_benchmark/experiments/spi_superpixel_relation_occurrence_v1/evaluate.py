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
from bda_benchmark.model.DamageFormer_SuperpixelRelationOccurrence_V1 import DamageFormerSuperpixelRelationOccurrenceV1
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import build_dataset, fast_hist, metrics_from_hist, parse_batch
from . import PROTOCOL


def valid_miou(hist: torch.Tensor) -> float:
    hist = hist.double(); union = hist.sum(0) + hist.sum(1) - torch.diag(hist)
    return float((torch.diag(hist)[1:4] / union[1:4].clamp_min(1)).mean())


def bootstrap(parent_rows, proposal_rows, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed); values = []
    for _ in range(iterations):
        ids = rng.integers(0, len(parent_rows), len(parent_rows))
        left = torch.stack([proposal_rows[i] for i in ids]).sum(0); right = torch.stack([parent_rows[i] for i in ids]).sum(0)
        values.append(100. * (valid_miou(left) - valid_miou(right)))
    return {"iterations": iterations, "mean_pp": float(np.mean(values)), "ci95_low_pp": float(np.quantile(values, .025)), "ci95_high_pp": float(np.quantile(values, .975))}


def event_macro_bootstrap(events, iterations: int, seed: int) -> dict:
    """Equal-event bootstrap so large disasters cannot dominate admission."""
    rng = np.random.default_rng(seed); names = sorted(events); gains = []
    observed = np.asarray([
        100. * (valid_miou(events[name][1]) - valid_miou(events[name][0]))
        for name in names
    ], dtype=np.float64)
    for _ in range(iterations):
        ids = rng.integers(0, len(names), len(names))
        gains.append(float(observed[ids].mean()))
    return {"iterations": iterations, "event_count": len(names), "observed_mean_pp": float(observed.mean()),
            "mean_pp": float(np.mean(gains)), "ci95_low_pp": float(np.quantile(gains, .025)),
            "ci95_high_pp": float(np.quantile(gains, .975))}


@torch.no_grad()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Internal evaluation for superpixel relation occurrence")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--dataset-path", required=True); parser.add_argument("--list-path", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--bootstrap-iterations", type=int, default=2000); parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv); saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_protocol = saved.get("protocol")
    if checkpoint_protocol != PROTOCOL:
        raise RuntimeError(
            "checkpoint protocol mismatch: "
            f"checkpoint={checkpoint_protocol!r}, evaluator={PROTOCOL!r}. "
            "Do not rename or evaluate a V1 checkpoint as corrected V2; "
            "synchronise every file in SERVER_SYNC_FILES.txt and retrain."
        )
    init = saved["args"]; model = DamageFormerSuperpixelRelationOccurrenceV1(saved["variant"], segments=init["segments"], compactness=init["compactness"])
    model.load_state_dict(saved["model_state"], strict=True); device = torch.device(args.device); model.to(device).eval()
    dataset = build_dataset(args.dataset_path, args.list_path, 1024, "test")
    if len(dataset) != args.expected_images: raise RuntimeError(f"expected {args.expected_images}, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    loc_hist = torch.zeros(2, 2, dtype=torch.float64, device=device); parent_hist = torch.zeros(4, 4, dtype=torch.float64, device=device); proposal_hist = torch.zeros_like(parent_hist)
    parent_rows, proposal_rows = [], []; events = defaultdict(lambda: [torch.zeros(4, 4), torch.zeros(4, 4)])
    severity_error = 0.; relation_boundary_sum = relation_count = 0
    for index, batch in enumerate(loader, 1):
        pre, post, location, damage, names = parse_batch(batch, device)
        loc, output, aux = model(pre, post, return_aux=True)
        parent = F.interpolate(aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False)
        pp, op = parent.argmax(1), output.argmax(1)
        loc_hist += fast_hist(loc.argmax(1), location, 2, 255); parent_hist += fast_hist(pp, damage, 4, 255); proposal_hist += fast_hist(op, damage, 4, 255)
        severity_error = max(severity_error, float(((output[:, 3] - output[:, 2]) - (parent[:, 3] - parent[:, 2])).abs().max()))
        relation_boundary_sum += int(aux["region_boundary"].sum()); relation_count += int(aux["region_boundary"].numel())
        for row in range(damage.shape[0]):
            ph = fast_hist(pp[row:row+1], damage[row:row+1], 4, 255).cpu(); oh = fast_hist(op[row:row+1], damage[row:row+1], 4, 255).cpu()
            parent_rows.append(ph); proposal_rows.append(oh); event = event_from_sample_name(str(names[row])); events[event][0] += ph; events[event][1] += oh
        if index % 10 == 0 or index == len(loader): print(f"SPR evaluate: {index}/{len(loader)}", flush=True)
    parent_metrics, proposal_metrics = metrics_from_hist(loc_hist, parent_hist), metrics_from_hist(loc_hist, proposal_hist)
    gain = 100. * (proposal_metrics["valid_miou"] - parent_metrics["valid_miou"]); dmg = 100. * (proposal_metrics["damaged_f1"] - parent_metrics["damaged_f1"]); des = 100. * (proposal_metrics["destroyed_f1"] - parent_metrics["destroyed_f1"])
    event_rows, nonnegative = {}, 0
    for event, (left, right) in sorted(events.items()):
        event_gain = 100. * (valid_miou(right) - valid_miou(left)); event_rows[event] = {"gain_valid_miou_pp": event_gain}; nonnegative += int(event_gain >= 0.)
    boot = bootstrap(parent_rows, proposal_rows, args.bootstrap_iterations, args.seed)
    event_boot = event_macro_bootstrap(events, args.bootstrap_iterations, args.seed + 1)
    checks = {"gain_ge_0p30pp": gain >= .30, "damaged_f1_not_below_parent": dmg >= 0., "destroyed_f1_drop_within_0p10pp": des >= -.10, "nonnegative_events_ge_10": nonnegative >= 10, "event_macro_bootstrap_ci_low_gt_zero": event_boot["ci95_low_pp"] > 0., "severity_difference_preserved": severity_error <= 1e-5}
    payload = {"protocol": PROTOCOL + "_INTERNAL_EVAL", "variant": saved["variant"], "parent": parent_metrics, "variant_metrics": proposal_metrics, "gain_valid_miou_pp": gain, "damaged_f1_gain_pp": dmg, "destroyed_f1_gain_pp": des, "events": event_rows, "nonnegative_events": f"{nonnegative}/{len(events)}", "image_bootstrap": boot, "equal_event_bootstrap": event_boot, "maximum_severity_difference_error": severity_error, "mean_superpixel_boundary_fraction": relation_boundary_sum / max(relation_count, 1), "checks": checks, "decision": "ADMIT_NEXT_ABLATION" if all(checks.values()) else "STOP_THIS_VARIANT", "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); print(f"Saved: {destination}", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
