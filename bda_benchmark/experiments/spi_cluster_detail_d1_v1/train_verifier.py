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

from bda_benchmark.experiments.spi_s2_cluster_evidence_v1.audit_e0 import utility_labels
from bda_benchmark.experiments.spi_s2_evidence_v2.audit_evidence import average_precision
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import PROTOCOL
from .core import selective_histogram_from_fixed


class HighResolutionClusterVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(cache: dict, groups: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([cache["features"][name].float() for name in groups], 1)


def _weights(cache: dict, valid: torch.Tensor) -> torch.Tensor:
    events = [cache["image_events"][int(i)] for i in cache["image_index"].long()[valid]]
    counts = Counter(events)
    event_weight = torch.tensor([1.0 / counts[event] for event in events])
    impact = cache["utility"]["changed"].float()[valid].sqrt().clamp(1.0, 12.0)
    result = event_weight * impact
    return result / result.mean().clamp_min(1e-8)


def _image_selected_histograms(cache: dict, accepted: torch.Tensor) -> torch.Tensor:
    rows = []
    for image_id in range(len(cache["image_names"])):
        belongs = cache["image_index"].long() == image_id
        rejected = belongs & ~accepted
        hist = cache["fixed_s2_histograms"][image_id].double()
        if bool(rejected.any()):
            hist = hist - cache["delta_hist"][rejected].sum(0).double()
        rows.append(hist)
    return torch.stack(rows)


def _event_report(cache: dict, selected_by_image: torch.Tensor) -> tuple[int, int, dict]:
    rows = {}
    for event in sorted(set(cache["image_events"])):
        ids = torch.tensor([i for i, value in enumerate(cache["image_events"]) if value == event])
        selected = metrics_from_histogram(selected_by_image[ids].sum(0))
        fixed = metrics_from_histogram(cache["fixed_s2_histograms"][ids].sum(0))
        rows[event] = {"gain_vs_fixed_s2_miou_pp": 100.0 * (selected["valid_miou"] - fixed["valid_miou"])}
    return sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in rows.values()), len(rows), rows


def _bootstrap(cache: dict, selected: torch.Tensor, iterations: int, seed: int) -> dict:
    fixed = cache["fixed_s2_histograms"].double()
    generator = torch.Generator().manual_seed(seed)
    gains = []
    for _ in range(iterations):
        ids = torch.randint(len(fixed), (len(fixed),), generator=generator)
        a = metrics_from_histogram(selected[ids].sum(0))["valid_miou"]
        b = metrics_from_histogram(fixed[ids].sum(0))["valid_miou"]
        gains.append(100.0 * (a - b))
    return {"mean_pp": float(np.mean(gains)), "ci95_low_pp": float(np.quantile(gains, .025)),
            "ci95_high_pp": float(np.quantile(gains, .975)), "iterations": iterations}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train the single D1 high-resolution cluster verifier")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--admission-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--utility-margin", type=float, default=0.10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.25)
    parser.add_argument("--minimum-oracle-recovery", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    admission = json.loads(Path(args.admission_json).read_text(encoding="utf-8"))
    if admission.get("decision") != "ADMIT_CLUSTER_DETAIL_D1_E1_SINGLE_VERIFIER":
        raise RuntimeError("D1 E1 refused: parameter-free D1 E0 did not pass")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, split in ((train, "meta_train"), (calibration, "meta_calibration")):
        if cache.get("protocol") != PROTOCOL or cache.get("split") != split:
            raise RuntimeError("unexpected D1 cache")
        if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError("D1 E1 received a forbidden cache")
    if admission["train_cache_sha256"] != _sha256(args.train_cache) or admission["calibration_cache_sha256"] != _sha256(args.calibration_cache):
        raise RuntimeError("D1 E0/cache fingerprint mismatch")

    selected_group = admission.get("selected_group")
    if selected_group not in admission.get("admitted_groups", ()):
        raise RuntimeError("D1 E0 did not freeze one admitted evidence group")
    # First learned experiment uses exactly one new evidence family. This keeps
    # the causal comparison interpretable and prevents opportunistic stacking.
    groups = ("spatial", str(selected_group))
    train_labels_all = utility_labels(train, args.utility_margin)
    valid = train_labels_all >= 0
    matrix = _matrix(train, groups)
    mean = matrix[valid].mean(0); std = matrix[valid].std(0, unbiased=False).clamp_min(1e-4)
    x = ((matrix[valid] - mean) / std).clamp(-8, 8)
    y = train_labels_all[valid].float(); weights = _weights(train, valid)

    torch.manual_seed(args.seed)
    model = HighResolutionClusterVerifier(x.shape[1], args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    pos_weight = ((y == 0).sum() / y.sum().clamp_min(1)).float()
    generator = torch.Generator().manual_seed(args.seed + 1)
    batch_size = min(args.batch_size, len(x))
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(x), (batch_size,), generator=generator)
        raw = F.binary_cross_entropy_with_logits(model(x[ids]), y[ids], reduction="none", pos_weight=pos_weight)
        loss = (raw * weights[ids]).mean()
        optimizer.zero_grad(); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"optimizer_step": step, "loss": float(loss), "gradient_norm": float(gradient)}), flush=True)

    calibration_x = ((_matrix(calibration, groups) - mean) / std).clamp(-8, 8)
    model.eval()
    with torch.no_grad():
        scores = model(calibration_x).sigmoid()
    fixed_report = metrics_from_histogram(calibration["fixed_s2_histograms"].sum(0))
    candidates = []
    for threshold in torch.linspace(0.05, 0.95, 91):
        accepted = scores >= threshold
        reject_fraction = float((~accepted).float().mean())
        if reject_fraction > args.maximum_reject_fraction:
            continue
        report = metrics_from_histogram(selective_histogram_from_fixed(
            calibration["fixed_s2_histograms"], calibration["delta_hist"], accepted
        ))
        candidates.append({"threshold": float(threshold), "reject_fraction": reject_fraction,
                           "miou": report["valid_miou"], "damaged_f1": report["damaged_f1"]})
    if not candidates:
        raise RuntimeError("no threshold satisfied the rejection cap")
    selected = max(candidates, key=lambda row: (row["miou"], row["damaged_f1"], -row["reject_fraction"]))
    accepted = scores >= selected["threshold"]
    selected_by_image = _image_selected_histograms(calibration, accepted)
    gain_pp = 100.0 * (selected["miou"] - fixed_report["valid_miou"])
    oracle_accept = calibration["utility"]["net"].long() > 0
    oracle_report = metrics_from_histogram(selective_histogram_from_fixed(
        calibration["fixed_s2_histograms"], calibration["delta_hist"], oracle_accept
    ))
    oracle_gain = 100.0 * (oracle_report["valid_miou"] - fixed_report["valid_miou"])
    recovery = gain_pp / max(oracle_gain, 1e-9)
    nonnegative, event_total, event_rows = _event_report(calibration, selected_by_image)
    bootstrap = _bootstrap(calibration, selected_by_image, args.bootstrap_iterations, args.seed + 99)
    cal_labels = utility_labels(calibration, args.utility_margin)
    cal_valid = cal_labels >= 0
    checks = {
        "gain_vs_fixed_s2_ge_threshold": gain_pp >= args.minimum_gain_pp,
        "oracle_recovery_ge_threshold": recovery >= args.minimum_oracle_recovery,
        "bootstrap_ci_low_gt_zero": bootstrap["ci95_low_pp"] > 0,
        "damaged_f1_not_below_fixed_s2": selected["damaged_f1"] >= fixed_report["damaged_f1"],
        "nonnegative_events_ge_10": nonnegative >= min(10, event_total),
        "reject_fraction_within_cap": selected["reject_fraction"] <= args.maximum_reject_fraction,
    }
    decision = "FREEZE_CLUSTER_DETAIL_D1_FOR_META_CONFIRM" if all(checks.values()) else "STOP_CLUSTER_DETAIL_D1_E1_AFTER_CALIBRATION"
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"protocol": "SPI_CLUSTER_DETAIL_D1_VERIFIER_v1", "model_state_dict": model.state_dict(),
                  "input_dim": x.shape[1], "hidden": args.hidden, "groups": groups,
                  "feature_mean": mean, "feature_std": std, "threshold": selected["threshold"],
                  "minimum_cluster_area": 33, "p3_sha256": train["p3_sha256"], "seed": args.seed,
                  "admission_sha256": _sha256(args.admission_json), "meta_confirm_accessed": False,
                  "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False}
    torch.save(checkpoint, output / "verifier_frozen.pth")
    report = {"protocol": "SPI_CLUSTER_DETAIL_D1_E1_CALIBRATION_v1", "configuration": vars(args),
              "groups": groups, "calibration_auroc": binary_auroc(cal_labels[cal_valid], scores[cal_valid]),
              "calibration_average_precision": average_precision(cal_labels[cal_valid], scores[cal_valid]),
              "fixed_s2": fixed_report, "eligible_oracle": oracle_report, "selected": selected,
              "gain_vs_fixed_s2_miou_pp": gain_pp, "eligible_oracle_gain_pp": oracle_gain,
              "oracle_recovery_ratio": recovery, "bootstrap": bootstrap,
              "event_nonnegative": nonnegative, "event_total": event_total, "event_rows": event_rows,
              "checks": checks, "decision": decision, "meta_confirm_accessed": False,
              "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False}
    (output / "calibration_screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "selected": selected, "gain_pp": gain_pp,
                      "eligible_oracle_gain_pp": oracle_gain, "oracle_recovery": recovery,
                      "bootstrap": bootstrap, "event_nonnegative": f"{nonnegative}/{event_total}",
                      "checks": checks, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {output / 'verifier_frozen.pth'}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
