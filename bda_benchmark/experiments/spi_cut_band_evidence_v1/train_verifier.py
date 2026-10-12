from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram

from .audit_e0 import labels


class CutBandVerifier(nn.Module):
    def __init__(self, inputs: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def _matrix(cache: dict, groups: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([cache["features"][name].float() for name in groups], dim=1)


def _event_balanced_weights(cache: dict, valid: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
    image_index = cache["image_index"].long()[valid]
    events = [cache["image_events"][int(index)] for index in image_index]
    counts = Counter(events)
    weights = torch.tensor([1.0 / counts[event] for event in events])
    weights *= utility[valid].abs().clamp_min(0.05).sqrt()
    return weights / weights.mean()


def _stable_holdout(cache: dict, valid: torch.Tensor, seed: int) -> torch.Tensor:
    rows = cache["image_index"].long()[valid]
    selected = []
    for index in rows:
        name = cache["image_names"][int(index)]
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        selected.append(int.from_bytes(digest[:4], "little") % 5 == 0)
    return torch.tensor(selected, dtype=torch.bool)


def _histogram(cache: dict, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    selected = scores >= threshold
    return cache["r0_histograms"].sum(0).double() + cache["delta_hist"][selected].sum(0).double()


def _event_report(cache: dict, scores: torch.Tensor, threshold: float) -> dict:
    rows = {}
    row_images = cache["image_index"].long()
    for event in sorted(set(cache["image_events"])):
        image_ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        image_mask = torch.isin(torch.arange(len(cache["image_names"])), image_ids)
        candidate_mask = torch.isin(row_images, image_ids) & (scores >= threshold)
        fixed_hist = cache["r0_histograms"][image_mask].sum(0).double()
        selective_hist = fixed_hist + cache["delta_hist"][candidate_mask].sum(0).double()
        fixed = metrics_from_histogram(fixed_hist)
        selective = metrics_from_histogram(selective_hist)
        rows[event] = 100.0 * (selective["valid_miou"] - fixed["valid_miou"])
    return {
        "gains_miou_pp": rows,
        "macro_gain_pp": float(np.mean(list(rows.values()))) if rows else float("nan"),
        "nonnegative": sum(value >= 0 for value in rows.values()),
        "total": len(rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train one low-capacity cut-band verifier")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--admission-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", default="geometry,rgb_edge,sar_edge,same_position,local_matching")
    parser.add_argument("--utility-margin", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)
    admission = json.loads(Path(args.admission_json).read_text())
    if admission.get("decision") != "ADMIT_CUT_BAND_VERIFIER_E1":
        raise RuntimeError("E1 training refused: CB-E0 did not admit local evidence")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    def file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if admission.get("train_cache_sha256") != file_sha256(args.train_cache):
        raise RuntimeError("admission/train cache fingerprint mismatch")
    if admission.get("calibration_cache_sha256") != file_sha256(args.calibration_cache):
        raise RuntimeError("admission/calibration cache fingerprint mismatch")
    groups = tuple(value.strip() for value in args.groups.split(",") if value.strip())
    unknown = [name for name in groups if name not in train["features"]]
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")
    y_all = labels(train, args.utility_margin); valid = y_all >= 0
    x_all, y = _matrix(train, groups)[valid], y_all[valid].float()
    holdout = _stable_holdout(train, valid, args.seed)
    if min(int(holdout.sum()), int((~holdout).sum())) == 0:
        raise RuntimeError("deterministic train holdout is empty")
    mean, std = x_all[~holdout].mean(0), x_all[~holdout].std(0, unbiased=False).clamp_min(1e-4)
    x = ((x_all - mean) / std).clamp(-8, 8)
    weights = _event_balanced_weights(train, valid, train["utility"]["utility_ratio"].float())
    torch.manual_seed(args.seed)
    model = CutBandVerifier(x.shape[1], args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positives, negatives = y[~holdout].sum(), (~y[~holdout].bool()).sum()
    pos_weight = (negatives / positives.clamp_min(1)).detach()
    for step in range(1, args.steps + 1):
        model.train(); logits = model(x[~holdout])
        raw = F.binary_cross_entropy_with_logits(logits, y[~holdout], reduction="none", pos_weight=pos_weight)
        loss = (raw * weights[~holdout]).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"optimizer_step": step, "loss": float(loss)}), flush=True)
    model.eval()
    with torch.no_grad():
        holdout_score = model(x[holdout]).sigmoid()
    thresholds = torch.linspace(0.20, 0.80, 61)
    utility = train["utility"]["net"][valid][holdout].float()
    threshold = max(thresholds, key=lambda t: float(utility[holdout_score >= t].sum()))
    cal_x = ((_matrix(calibration, groups) - mean) / std).clamp(-8, 8)
    with torch.no_grad():
        cal_score = model(cal_x).sigmoid()
    selective = metrics_from_histogram(_histogram(calibration, cal_score, float(threshold)))
    fixed = metrics_from_histogram(calibration["r0_histograms"].sum(0))
    all_split = metrics_from_histogram(calibration["r1_histograms"].sum(0))
    gain = 100.0 * (selective["valid_miou"] - fixed["valid_miou"])
    event_report = _event_report(calibration, cal_score, float(threshold))
    checks = {
        "gain_vs_fixed_s2_ge_0_10pp": gain >= 0.10,
        "damaged_f1_not_below_fixed_s2": selective["damaged_f1"] >= fixed["damaged_f1"],
        "nonnegative_events_ge_10": event_report["nonnegative"] >= 10,
    }
    result = {
        "protocol": "SPI_CUT_BAND_VERIFIER_E1_v1", "groups": groups,
        "threshold_source": "deterministic_20pct_meta_train_image_holdout",
        "threshold": float(threshold), "reports": {"fixed_s2": fixed, "all_rgid_s2": all_split, "selective": selective},
        "selective_gain_vs_fixed_s2_miou_pp": gain, "event_report": event_report,
        "checks": checks,
        "decision": "ADMIT_CUT_BAND_E1_META_CONFIRM" if all(checks.values()) else "STOP_CUT_BAND_E1_NO_DEPLOYABLE_GAIN",
        "meta_confirm_accessed": False, "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "input_dim": x.shape[1], "hidden": args.hidden,
                "groups": groups, "mean": mean, "std": std, "threshold": float(threshold)}, output / "verifier.pth")
    (output / "calibration_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gain_miou_pp": gain, "decision": result["decision"]}, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
