from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import metrics_from_histogram
from bda_benchmark.script.standard_ML.im4cd_matching_runtime import binary_auroc

from . import AUDIT_PROTOCOL, TARGET_PROTOCOL
from .audit_oracle import bootstrap, event_rows
from .core import metric_labels, per_image_selective_histograms, selective_histogram


DEFAULT_GROUPS = ("spatial", "stage2_role", "rgb_edge", "sar_texture", "structure_relation")


class MetricUtilityVerifier(nn.Module):
    def __init__(self, input_dim: int, model_type: str, hidden: int) -> None:
        super().__init__()
        if model_type == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif model_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
                nn.Linear(hidden, 1),
            )
        else:
            raise ValueError("model_type must be linear or mlp")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(cache: dict, groups: tuple[str, ...], split: str) -> torch.Tensor:
    missing = [name for name in groups if name not in cache["features"]]
    if missing:
        raise RuntimeError(f"missing feature groups: {missing}")
    rows = []
    repaired = {}
    half_max = float(torch.finfo(torch.float16).max)
    for name in groups:
        values = cache["features"][name].float()
        nan_count = int(torch.isnan(values).sum())
        if nan_count:
            raise RuntimeError(
                f"{split} feature group {name!r} contains {nan_count} NaN values; "
                "the source cache is corrupted"
            )
        positive_inf = int(torch.isposinf(values).sum())
        negative_inf = int(torch.isneginf(values).sum())
        if positive_inf or negative_inf:
            # D1 validates values in float32 and stores them in float16.  A raw
            # area-like value above 65504 therefore becomes +/-inf on disk.
            # Saturating it at the float16 boundary preserves its ordering and
            # allows the train-only standardizer to handle the column safely.
            values = torch.nan_to_num(
                values, nan=0.0, posinf=half_max, neginf=-half_max
            )
            repaired[name] = {
                "positive_inf": positive_inf,
                "negative_inf": negative_inf,
            }
        rows.append(values)
    result = torch.cat(rows, 1)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"{split} feature matrix remains non-finite after repair")
    if repaired:
        print(json.dumps({"feature_overflow_repair": split, "groups": repaired}), flush=True)
    return result


def sample_weights(cache: dict, targets: dict, valid: torch.Tensor) -> torch.Tensor:
    image_ids = cache["image_index"].long()[valid]
    events = [cache["image_events"][int(index)] for index in image_ids]
    counts = Counter(events)
    event_weight = torch.tensor([1.0 / counts[event] for event in events])
    impact = targets["metric_utility_pp"].abs().float()[valid]
    impact = impact / impact.median().clamp_min(1e-12)
    impact = impact.sqrt().clamp(0.25, 8.0)
    result = event_weight * impact
    result = result / result.mean().clamp_min(1e-8)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("sample weights contain non-finite values")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Train metric-aligned low-capacity cluster verifier")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--calibration-targets", required=True)
    parser.add_argument("--oracle-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--model-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon-pp", type=float, default=0.0)
    parser.add_argument("--maximum-reject-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-gain-pp", type=float, default=0.30)
    parser.add_argument("--minimum-oracle-recovery", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args(argv)

    oracle = json.loads(Path(args.oracle_json).read_text(encoding="utf-8"))
    if oracle.get("protocol") != AUDIT_PROTOCOL or oracle.get("decision") != "ADMIT_METRIC_ALIGNED_LINEAR_VERIFIER_E1":
        raise RuntimeError("MACU-E1 refused: metric Oracle M0 did not pass")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    train_targets = torch.load(args.train_targets, map_location="cpu", weights_only=False)
    cal_targets = torch.load(args.calibration_targets, map_location="cpu", weights_only=False)
    for target, split, source in ((train_targets, "meta_train", args.train_cache), (cal_targets, "meta_calibration", args.calibration_cache)):
        if target.get("protocol") != TARGET_PROTOCOL or target.get("split") != split:
            raise RuntimeError("unexpected metric target cache")
        if target["source_cache_sha256"] != sha256(source):
            raise RuntimeError("metric targets/source cache mismatch")
    for cache, split in ((train, "meta_train"), (calibration, "meta_calibration")):
        if cache.get("split") != split or cache.get("meta_confirm_accessed", True):
            raise RuntimeError(f"unexpected or contaminated {split} cache")
        if cache.get("val450_accessed", True) or cache.get("official_instances_accessed", True) or cache.get("test_accessed", True):
            raise RuntimeError(f"{split} cache accessed a forbidden split")
    groups = tuple(dict.fromkeys(name.strip() for name in args.groups.split(",") if name.strip()))
    train_labels_all = metric_labels(train_targets["metric_utility_pp"], args.epsilon_pp)
    valid = train_labels_all >= 0
    x_all = matrix(train, groups, "meta_train")
    mean = x_all[valid].mean(0); std = x_all[valid].std(0, unbiased=False).clamp_min(1e-4)
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise RuntimeError("train-only feature normalization is non-finite")
    x = ((x_all[valid] - mean) / std).clamp(-8, 8)
    y = train_labels_all[valid].float()
    weights = sample_weights(train, train_targets, valid)

    torch.manual_seed(args.seed)
    model = MetricUtilityVerifier(x.shape[1], args.model_type, args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    pos_weight = ((y == 0).sum() / y.sum().clamp_min(1.0)).float()
    generator = torch.Generator().manual_seed(args.seed + 1)
    batch_size = min(args.batch_size, len(x))
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(x), (batch_size,), generator=generator)
        loss_rows = F.binary_cross_entropy_with_logits(
            model(x[ids]), y[ids], reduction="none", pos_weight=pos_weight
        )
        loss = (loss_rows * weights[ids]).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite verifier loss at optimizer step {step}")
        optimizer.zero_grad(); loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite gradient norm at optimizer step {step}")
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"optimizer_step": step, "model": args.model_type,
                              "loss": float(loss.detach()),
                              "gradient_norm": float(gradient_norm.detach())}), flush=True)

    cal_x = ((matrix(calibration, groups, "meta_calibration") - mean) / std).clamp(-8, 8)
    model.eval()
    with torch.no_grad():
        scores = model(cal_x).sigmoid()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("calibration verifier scores contain non-finite values")
    fixed = metrics_from_histogram(calibration["fixed_s2_histograms"].sum(0))
    candidates = []
    for threshold in torch.linspace(0.05, 0.95, 91):
        accept = scores >= threshold
        reject_fraction = float((~accept).float().mean())
        if reject_fraction > args.maximum_reject_fraction:
            continue
        report = metrics_from_histogram(selective_histogram(
            calibration["fixed_s2_histograms"], calibration["delta_hist"], accept
        ))
        candidates.append({"threshold": float(threshold), "reject_fraction": reject_fraction,
                           "miou": report["valid_miou"], "damaged_f1": report["damaged_f1"]})
    if not candidates:
        raise RuntimeError("no threshold satisfied the rejection cap")
    selected = max(candidates, key=lambda row: (row["miou"], row["damaged_f1"], -row["reject_fraction"]))
    accept = scores >= selected["threshold"]
    selected_by_image = per_image_selective_histograms(calibration, accept)
    gain_pp = 100.0 * (selected["miou"] - fixed["valid_miou"])
    metric_oracle_gain = oracle["variants"]["metric_linear_oracle"]["gain_vs_fixed_s2_miou_pp"]
    recovery = gain_pp / max(metric_oracle_gain, 1e-9)
    events = event_rows(calibration, selected_by_image)
    nonnegative = sum(row["gain_vs_fixed_s2_miou_pp"] >= 0 for row in events.values())
    bootstrap_row = bootstrap(calibration, selected_by_image, args.bootstrap_iterations, args.seed + 99)
    cal_labels = metric_labels(cal_targets["metric_utility_pp"], args.epsilon_pp)
    cal_valid = cal_labels >= 0
    checks = {
        "gain_vs_fixed_s2_ge_threshold": gain_pp >= args.minimum_gain_pp,
        "oracle_recovery_ge_threshold": recovery >= args.minimum_oracle_recovery,
        "bootstrap_ci_low_gt_zero": bootstrap_row["ci95_low_pp"] > 0,
        "damaged_f1_not_below_fixed_s2": selected["damaged_f1"] >= fixed["damaged_f1"],
        "nonnegative_events_ge_10": nonnegative >= min(10, len(events)),
        "reject_fraction_within_cap": selected["reject_fraction"] <= args.maximum_reject_fraction,
    }
    decision = "FREEZE_MACU_E1_FOR_META_CONFIRM" if all(checks.values()) else "STOP_MACU_E1_AFTER_CALIBRATION"
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"protocol": "SPI_METRIC_ALIGNED_CLUSTER_VERIFIER_E1_v1",
                  "model_state_dict": model.state_dict(), "model_type": args.model_type,
                  "input_dim": x.shape[1], "hidden": args.hidden, "groups": groups,
                  "feature_mean": mean, "feature_std": std, "threshold": selected["threshold"],
                  "nonfinite_feature_policy": "reject_nan_saturate_float16_infinity",
                  "minimum_cluster_area": 33, "p3_sha256": train["p3_sha256"], "seed": args.seed,
                  "oracle_sha256": sha256(args.oracle_json), "meta_confirm_accessed": False,
                  "val450_accessed": False, "official_instances_accessed": False, "test_accessed": False}
    checkpoint_name = (
        "verifier_frozen.pth"
        if decision == "FREEZE_MACU_E1_FOR_META_CONFIRM"
        else "verifier_rejected.pth"
    )
    torch.save(checkpoint, output / checkpoint_name)
    report = {"protocol": "SPI_METRIC_ALIGNED_CLUSTER_VERIFIER_E1_CALIBRATION_v1",
              "configuration": vars(args), "groups": groups,
              "nonfinite_feature_policy": "reject_nan_saturate_float16_infinity",
              "calibration_metric_target_auroc": binary_auroc(cal_labels[cal_valid], scores[cal_valid]),
              "fixed_s2": fixed, "selected": selected, "gain_vs_fixed_s2_miou_pp": gain_pp,
              "metric_oracle_gain_pp": metric_oracle_gain, "oracle_recovery_ratio": recovery,
              "events": events, "event_nonnegative": nonnegative, "event_total": len(events),
              "bootstrap": bootstrap_row, "checks": checks, "decision": decision,
              "meta_confirm_accessed": False, "val450_accessed": False,
              "official_instances_accessed": False, "test_accessed": False}
    (output / "calibration_screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "gain_pp": gain_pp,
                      "metric_oracle_gain_pp": metric_oracle_gain,
                      "oracle_recovery_ratio": recovery, "bootstrap": bootstrap_row,
                      "event_nonnegative": f"{nonnegative}/{len(events)}",
                      "checks": checks, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {output / checkpoint_name}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
